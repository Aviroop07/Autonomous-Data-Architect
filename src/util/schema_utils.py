import json
from typing import Type, Any, Dict, List, Union, get_origin, get_args
from pydantic import BaseModel

def generate_hierarchical_schema_description(model: Type[BaseModel], indent: int = 0) -> str:
    """
    Recursively generates a hierarchical Markdown description of a Pydantic model's 
    structure, including field names, types, descriptions, and nested fields.
    """
    lines = []
    prefix = "  " * indent
    
    # Get JSON schema to easily access field metadata
    schema = model.model_json_schema()
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    
    # Map for recursive lookups
    def_map = schema.get("$defs", {})

    def resolve_type_info(prop: Dict[str, Any]) -> str:
        if "anyOf" in prop:
            return " | ".join(resolve_type_info(p) for p in prop["anyOf"])
        if "allOf" in prop:
            return " & ".join(resolve_type_info(p) for p in prop["allOf"])
        if "$ref" in prop:
            ref_name = prop["$ref"].split("/")[-1]
            return ref_name
        if prop.get("type") == "array" and "items" in prop:
            return f"List[{resolve_type_info(prop['items'])}]"
        return prop.get("type", "any")

    for field_name in sorted(properties.keys()):
        prop = properties[field_name]
        field_type_str = resolve_type_info(prop)
        field_desc = prop.get("description", "No description provided.")
        is_required = field_name in required
        
        req_label = "(Required)" if is_required else "(Optional)"
        lines.append(f"{prefix}- **{field_name}** (`{field_type_str}`): {field_desc} {req_label}")
        
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
            
        for m in target_models:
            lines.append(generate_hierarchical_schema_description(m, indent + 1))

    return "\n".join(lines)
