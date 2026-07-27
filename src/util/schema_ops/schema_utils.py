from typing import Type, Any, Dict, Optional, Set, Union, get_origin, get_args
from pydantic import BaseModel


def generate_hierarchical_schema_description(
    model: Type[BaseModel],
    indent: int = 0,
    _described: Optional[Set[Type[BaseModel]]] = None,
    _described_unions: Optional[Set[tuple]] = None,
) -> str:
    """
    Recursively generates a hierarchical Markdown description of a Pydantic model's
    structure, including field names, types, descriptions, and nested fields.

    _described is internal (recursive calls only) -- a SHARED, mutable set
    of every model class already fully expanded anywhere in this call tree
    (not just on the current path). Each distinct model class is described
    in full exactly ONCE, on its first occurrence; every later occurrence
    anywhere else (a repeated union member, or a genuine self-/mutually-
    referential cycle like RArithmetic nesting itself via left/right) gets a
    short "-> see <Name> above" pointer instead of a full re-expansion.
    Without this, a moderately deep discriminated-union schema (e.g. this
    project's ON-tree/R-AST models) re-expands the same handful of leaf
    types at every occurrence and blows up combinatorially -- confirmed via
    a real measurement: fixing the discriminator-tag-value gap below without
    this dedup made Stage 3's UnifiedExtractionOutput description alone
    balloon to ~230k tokens, which would have gone straight into every
    single agent call's prompt.

    _described_unions is the same idea one level up: a discriminated
    union's FULL signature (`{tag: 'x'} -> ClassName | ...` for every
    variant) is reprinted at every field that uses it, even though the
    variant bodies themselves are already deduped -- a union like this
    project's RPredicate (~10 variants) or RExprUnion appears at 20+
    distinct field positions in UnifiedExtractionOutput alone, each one
    re-paying the full tag-mapping cost. Keyed by (discriminator property
    name, sorted variant class names) since JSON schema has no name for
    an anonymous Union[...] type alias -- only its shape. First occurrence
    prints the full `{tag}->Class` mapping; later occurrences print just
    the bare class list (`ClassA | ClassB | ...`), since the tag-to-class
    association was already established earlier in the same document.
    """
    described = _described if _described is not None else set()
    described_unions = _described_unions if _described_unions is not None else set()
    if model in described:
        return f"{'  ' * indent}-> see **{model.__name__}** above (already fully described)"
    described.add(model)

    lines = []
    prefix = "  " * indent

    # Get JSON schema to easily access field metadata
    schema = model.model_json_schema()
    # A self-/mutually-referential model (e.g. RArithmetic, which nests
    # itself via left/right) gets hoisted into $defs by Pydantic, with the
    # top-level schema left as a bare {"$ref": ..., "$defs": {...}} pointer
    # -- "properties" lives on the $defs entry, not the root, in that case.
    # Without this resolution the model's OWN fields silently render as an
    # empty section (the same OUTPUT FORMAT blind spot as the discriminator
    # tag gap fixed above -- both leave the LLM guessing).
    def_map = schema.get("$defs", {})
    if "$ref" in schema and "properties" not in schema:
        ref_name = schema["$ref"].split("/")[-1]
        schema = def_map.get(ref_name, schema)
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    def resolve_type_info(prop: Dict[str, Any]) -> str:
        if "anyOf" in prop:
            return " | ".join(resolve_type_info(p) for p in prop["anyOf"])
        if "allOf" in prop:
            # Handle cases where Enum is nested in allOf (common in Pydantic 2)
            parts = []
            for p in prop["allOf"]:
                if "$ref" in p:
                    ref_name = p["$ref"].split("/")[-1]
                    ref_def = def_map.get(ref_name, {})
                    if "enum" in ref_def:
                        vals = ", ".join(
                            [
                                f"'{v}'" if isinstance(v, str) else str(v)
                                for v in ref_def["enum"]
                            ]
                        )
                        parts.append(f"Enum[{vals}]")
                    else:
                        parts.append(ref_name)
                elif "enum" in p:
                    vals = ", ".join(
                        [f"'{v}'" if isinstance(v, str) else str(v) for v in p["enum"]]
                    )
                    parts.append(f"Enum[{vals}]")
                else:
                    parts.append(p.get("type", "any"))
            return " & ".join(parts)
        if "$ref" in prop:
            ref_name = prop["$ref"].split("/")[-1]
            ref_def = def_map.get(ref_name, {})
            if "enum" in ref_def:
                vals = ", ".join(
                    [
                        f"'{v}'" if isinstance(v, str) else str(v)
                        for v in ref_def["enum"]
                    ]
                )
                return f"Enum[{vals}]"
            return ref_name
        if "discriminator" in prop and "oneOf" in prop:
            # A Field(discriminator=...) union: the model MUST see the actual
            # tag VALUE each variant requires (e.g. 'table'), not just the
            # class name -- the class name is what it guesses instead
            # otherwise, which is a real value json_mode providers (no
            # server-side schema enforcement) will actually emit verbatim.
            prop_name = prop["discriminator"].get("propertyName", "type")
            mapping = prop["discriminator"].get("mapping", {})
            class_names = tuple(
                sorted(ref_path.split("/")[-1] for ref_path in mapping.values())
            )
            union_key = (prop_name, class_names)
            if union_key in described_unions:
                # Tag-to-class mapping already shown at this union's first
                # occurrence -- repeat only the (already-deduped) class
                # list, not the full {tag}->Class pairing again.
                return "Discriminated[" + " | ".join(class_names) + "]"
            described_unions.add(union_key)
            parts = [
                f"{{{prop_name}: '{tag}'}} -> {ref_path.split('/')[-1]}"
                for tag, ref_path in mapping.items()
            ]
            return "Discriminated[" + " | ".join(parts) + "]"
        if "const" in prop:
            # A single-value Literal (every discriminator tag field, e.g.
            # `type: Literal["table"] = "table"`) -- previously fell through
            # to a bare "string"/"any", the actual gap that let an LLM using
            # a provider with no server-side schema enforcement (json_mode)
            # emit the wrong tag value with nothing in the prompt to catch it.
            const_val = prop["const"]
            return (
                f"Literal['{const_val}']"
                if isinstance(const_val, str)
                else f"Literal[{const_val}]"
            )
        if "enum" in prop:
            vals = ", ".join(
                [f"'{v}'" if isinstance(v, str) else str(v) for v in prop["enum"]]
            )
            return f"Enum[{vals}]"
        if prop.get("type") == "array" and "items" in prop:
            return f"List[{resolve_type_info(prop['items'])}]"
        return prop.get("type", "any")

    for field_name in sorted(properties.keys()):
        prop = properties[field_name]
        field_type_str = resolve_type_info(prop)
        field_desc = prop.get("description", "No description provided.")
        is_required = field_name in required

        req_label = "(Required)" if is_required else "(Optional)"
        lines.append(
            f"{prefix}- **{field_name}** (`{field_type_str}`): {field_desc} {req_label}"
        )

        # Check for nested models to recurse
        field_info = model.model_fields[field_name]
        actual_type = field_info.annotation

        # Handle Union, Optional, List
        origin = get_origin(actual_type)
        args = get_args(actual_type)

        target_models = []
        if origin is list:
            if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
                target_models.append(args[0])
        elif origin is Union:
            for arg in args:
                if isinstance(arg, type) and issubclass(arg, BaseModel):
                    target_models.append(arg)
        elif isinstance(actual_type, type) and issubclass(actual_type, BaseModel):
            target_models.append(actual_type)

        # When recursing into a Union's member models, label each variant's
        # field list with its class name -- without this, the model sees an
        # unlabeled, flattened concatenation of every variant's fields (e.g.
        # ONBaseTable's + ONJoin's + ONAggregate's, run together with no
        # indication which fields belong together), compounding the missing-
        # discriminator-value gap fixed above.
        multi = len(target_models) > 1
        for m in target_models:
            if multi:
                lines.append(f"{prefix}  - Variant **{m.__name__}**:")
                lines.append(
                    generate_hierarchical_schema_description(
                        m, indent + 2, described, described_unions
                    )
                )
            else:
                lines.append(
                    generate_hierarchical_schema_description(
                        m, indent + 1, described, described_unions
                    )
                )

    return "\n".join(lines)
