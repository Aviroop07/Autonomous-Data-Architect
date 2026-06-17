import base64
import graphviz
import html
import networkx as nx
from typing import List, Any, Union
from src.pipeline.stage2.models.schema import Schema
from src.util.schema_ops.schema_patch import (
    SchemaPatch, CritiqueReport, MergeTablesPatch
)
from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.pipeline.stage1.models.integrity_report import IntegrityReport
from src.pipeline.stage2.models.corrections import FixHistoryStep, CorrectionStatus
try:
    from src.pipeline.stage3.models import LogicNode, NodeType
    _HAS_STAGE3_MODELS = True
except ImportError:
    _HAS_STAGE3_MODELS = False

def render_schema_to_base64(schema: Schema) -> str:
    """
    Renders a Schema object to a Graphviz ER diagram and returns a base64 Data URI.
    Standard rendering without highlighting patches.
    """
    dot = graphviz.Digraph(comment='ER Diagram', format='png')
    dot.attr(rankdir='LR', overlap='false', splines='true', bgcolor='white', fontname='Helvetica')

    # Modern Color Palette
    HEADER_BG = '#1E293B' # Dark Slate
    TABLE_BG = '#F8FAFC'
    TEXT_COLOR = '#0F172A'
    HEADER_TEXT = '#F1F5F9'
    BORDER_COLOR = '#CBD5E1'
    PK_COLOR = '#F59E0B' # Amber for PK

    for table in schema.tables:
        html_label = f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="10" BGCOLOR="{TABLE_BG}" PORT="p" STYLE="ROUNDED">'
        html_label += f'<TR><TD BGCOLOR="{HEADER_BG}" COLSPAN="2"><B><FONT COLOR="{HEADER_TEXT}" POINT-SIZE="11">{table.name}</FONT></B></TD></TR>'

        for col in table.columns:
            is_pk = col.name == table.pk
            icon = "🔑 " if is_pk else "  "
            color = PK_COLOR if is_pk else TEXT_COLOR
            font_weight = "B" if is_pk else "I" if "id" in col.name.lower() else ""

            label_tag = f"<{font_weight}>{icon}{col.name}</{font_weight}>" if font_weight else f"{icon}{col.name}"

            html_label += f'<TR><TD ALIGN="LEFT" BORDER="0"><FONT COLOR="{color}" POINT-SIZE="10">{label_tag}</FONT></TD>'
            html_label += f'<TD ALIGN="RIGHT" BORDER="0"><FONT COLOR="#64748B" POINT-SIZE="9">{col.data_type}</FONT></TD></TR>'

        html_label += '</TABLE>>'
        dot.node(table.name, label=html_label, shape='none')

    if schema.relationships:
        for rel in schema.relationships:
            dot.edge(rel.referencing_table, rel.referred_table,
                     label=f"{rel.referencing_column}",
                     fontsize='9',
                     color='#94A3B8',
                     fontcolor='#475569',
                     arrowsize='0.7',
                     penwidth='1.5')

    try:
        img_data = dot.pipe()
        encoded = base64.b64encode(img_data).decode('utf-8')
        return f"data:image/png;base64,{encoded}"
    except Exception as e:
        return f"IMAGE_RENDER_FAILED: {str(e)}"

def render_logic_tree_to_base64(root_node: "LogicNode") -> str:
    """
    Renders a recursive LogicNode tree to a Graphviz visualization.
    Returns a base64 Data URI.
    """
    dot = graphviz.Digraph(comment='Logic Tree', format='png')
    dot.attr(rankdir='TB', overlap='false', splines='true', bgcolor='white', fontname='Helvetica')

    # Modern Logic Palette
    OP_BG = '#F3F4F6'
    BOOLEAN_BG = '#DCFCE7'  # Greenish
    NUMERIC_BG = '#DBEAFE'  # Blueish
    TEXT_COLOR = '#1F2937'
    BORDER_COLOR = '#9CA3AF'

    node_count = 0

    def add_node(node: LogicNode) -> str:
        nonlocal node_count
        id = f"node_{node_count}"
        node_count += 1

        # Determine label and style
        header = node.type.value
        body = ""
        bg_color = OP_BG

        if node.type == NodeType.CONSTANT:
            body = f'[{node.value}]'
            bg_color = NUMERIC_BG if isinstance(node.value, (int, float)) else BOOLEAN_BG if isinstance(node.value, bool) else OP_BG
        elif node.type == NodeType.COLUMN_REF:
            body = f'{node.table}.{node.column}'
            bg_color = '#FEF3C7' # Yellowish
        elif node.op:
            body = node.op

        label = f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4" BGCOLOR="{bg_color}">'
        label += f'<TR><TD><B><FONT POINT-SIZE="10">{html.escape(header)}</FONT></B></TD></TR>'
        if body:
            label += f'<TR><TD><FONT POINT-SIZE="12">{html.escape(str(body))}</FONT></TD></TR>'
        label += '</TABLE>>'

        dot.node(id, label=label, shape='none')

        # Recursive children
        if node.left:
            child_id = add_node(node.left)
            dot.edge(id, child_id, label="left")
        if node.right:
            child_id = add_node(node.right)
            dot.edge(id, child_id, label="right")
        if node.condition:
            child_id = add_node(node.condition)
            dot.edge(id, child_id, label="if")
        if node.then_branch:
            child_id = add_node(node.then_branch)
            dot.edge(id, child_id, label="then")
        if node.else_branch:
            child_id = add_node(node.else_branch)
            dot.edge(id, child_id, label="else")
        if node.on_condition:
            child_id = add_node(node.on_condition)
            dot.edge(id, child_id, label="on")
        if node.filter_condition:
            child_id = add_node(node.filter_condition)
            dot.edge(id, child_id, label="filter")
        if node.children:
            for i, child in enumerate(node.children):
                child_id = add_node(child)
                dot.edge(id, child_id, label=f"child_{i}")

        return id

    try:
        add_node(root_node)
        img_data = dot.pipe()
        encoded = base64.b64encode(img_data).decode('utf-8')
        return f"data:image/png;base64,{encoded}"
    except Exception as e:
        return f"IMAGE_RENDER_FAILED: {str(e)}"
def render_dag_to_base64(graph_data: Union[nx.DiGraph, dict, str]) -> str:
    """
    Renders a NetworkX DiGraph (or its node_link_data) to a Graphviz visualization.
    Returns a base64 Data URI.
    """
    import json
    if isinstance(graph_data, str):
        graph_data = json.loads(graph_data)
    if isinstance(graph_data, dict):
        graph = nx.node_link_graph(graph_data)
    else:
        graph = graph_data

    dot = graphviz.Digraph(comment='Dependency DAG', format='png')
    dot.attr(rankdir='LR', overlap='false', splines='true', bgcolor='white', fontname='Helvetica')

    # Modern DAG Palette
    FK_COLOR = '#334155'
    LOGIC_COLOR = '#059669'
    NODE_BG = '#F1F5F9'
    NODE_BORDER = '#94A3B8'
    TEXT_COLOR = '#1E293B'

    for node in graph.nodes():
        node_label = f"{node[0]}.{node[1]}" if isinstance(node, (list, tuple)) else str(node)
        dot.node(str(node), label=node_label, shape='rect', style='filled,rounded',
                 fillcolor=NODE_BG, color=NODE_BORDER, fontname='Helvetica-Bold',
                 fontsize='10', fontcolor=TEXT_COLOR)

    for u, v, data in graph.edges(data=True):
        edge_type = data.get('type', 'FK')
        color = LOGIC_COLOR if edge_type == 'LOGIC' else FK_COLOR
        style = 'solid' if edge_type == 'FK' else 'dashed'
        label = "Logic" if edge_type == 'LOGIC' else "FK"

        dot.edge(str(u), str(v), color=color, style=style, penwidth='1.5',
                 arrowhead='vee', label=label, fontsize='8', fontcolor=color)

    try:
        img_data = dot.pipe()
        encoded = base64.b64encode(img_data).decode('utf-8')
        return f"data:image/png;base64,{encoded}"
    except Exception as e:
        return f"IMAGE_RENDER_FAILED: {str(e)}"

def format_schema_patch(patch: SchemaPatch) -> str:
    """
    Returns a high-quality Markdown block describing a single patch.
    Uses GitHub Alerts for visual emphasis.
    """
    # Extract action as string and normalize
    action = str(patch.action)
    if "ActionTag." in action:
        action = action.split(".")[-1]

    lines = []

    emoji = "➕" if "ADD" in action else "🗑️" if "DELETE" in action else "⚙️"
    color = "TIP" if "ADD" in action else "CAUTION" if "DELETE" in action else "IMPORTANT"

    lines.append(f"> [!{color}]")
    lines.append(f"> **{emoji} {action}**")
    lines.append(f"> {patch.reason}")
    lines.append("> ")

    details = []
    if hasattr(patch, 'table_name'):
        details.append(f"Table: `{patch.table_name}`")
    if hasattr(patch, 'column_name'):
        details.append(f"Column: `{patch.column_name}`")

    if hasattr(patch, 'new_name'):
        details.append(f"New Name: `{patch.new_name}`")
    elif isinstance(patch, MergeTablesPatch):
        details.append(f"Source: `{patch.source_table}`")
        details.append(f"Target: `{patch.target_table}`")
    elif hasattr(patch, 'table_definition'):
        details.append(f"Table: `{patch.table_definition.name}`")
        details.append(f"Columns: `{', '.join(patch.table_definition.columns)}` (PK: `{patch.table_definition.pk}`)")
    elif hasattr(patch, 'fk_definition') and patch.fk_definition:
        rk = patch.fk_definition
        details.append(f"Relation: `{rk.referencing_table}.{rk.referencing_column} -> {rk.referred_table}`")
    elif hasattr(patch, 'unique_definition'):
        details.append(f"Unique Relation: `{', '.join(patch.unique_definition.columns)}`")

    for d in details:
        lines.append(f"> - {d}")

    return "\n".join(lines)

def format_critique_report(report: CritiqueReport) -> str:
    """
    Formats a full CritiqueReport into Markdown with premium visual grouping.
    """
    lines = [f"### Assessment by: `{report.agent_name}`"]
    if report.observations:
        lines.append(f"\n**Strategic Observations**:\n{report.observations}\n")

    if report.patches:
        lines.append("**Proposed Architecture Patches**:\n")
        for p in report.patches:
            lines.append(format_schema_patch(p))
            lines.append("\n")
    else:
        lines.append("\n> [!NOTE]\n> *No structural patches required for this shard requirement.*")

    return "\n".join(lines)

def format_atomic_facts(facts: List[Any]) -> str:
    """Formats a list of AtomicFact or RawFact objects into a Markdown list."""
    if not facts: return "No facts found."
    lines = []
    for f in facts:
        snippet = f' (Source: "{f.origin}")' if f.origin else ""
        tags = getattr(f, "tags", [])
        tags_str = ", ".join(str(t) for t in tags) if tags else "UNTAGGED"

        refs = ""
        if hasattr(f, "referenced_fact_ids") and f.referenced_fact_ids:
            refs = f" [Refs: {f.referenced_fact_ids}]"

        lines.append(f"{f.id}. **[{tags_str}]** {f.fact}{refs}{snippet}")
    return "\n".join(lines)

def format_integrity_report(report: IntegrityReport) -> str:
    """Formats an IntegrityReport into Markdown details."""
    return str(report)

def format_fix_history(history: List[FixHistoryStep]) -> str:
    """Formats the hardened FixHistoryStep list."""
    if not history: return "No correction history."

    lines = []
    for step in history:
        lines.append(f"#### Attempt {step.attempt}")
        if step.errors:
            lines.append("\n**Errors Detected**:")
            for err in step.errors:
                lines.append(f"- `{err}`")

        if step.corrections:
            lines.append("\n**Corrections Applied**:")
            for c in step.corrections:
                status_icon = "✅" if c.status == CorrectionStatus.FIXED else "❌" if c.status == CorrectionStatus.NOT_FIXED else "⏳"
                msg = f"{status_icon} **{c.status.upper()}**: {c.error_message}"
                if c.description:
                    msg += f"\n   - *Note*: {c.description}"
                lines.append(f"- {msg}")

        if hasattr(step, "schema_state") and step.schema_state:
            try:
                img_uri = render_schema_to_base64(step.schema_state)
                lines.append(f"\n![Attempt {step.attempt} Schema]({img_uri})\n")
            except Exception as e:
                lines.append(f"\n*Image rendering failed: {str(e)}*")

    return "\n".join(lines)

def format_patch_repair_history(history: List[Any]) -> str:
    """Formats the Stage 3 automated repair history."""
    if not history: return "*No repair attempts.*"

    lines = ["**Automated Repair Audit Trail:**"]
    for step in history:
        lines.append(str(step))

    return "\n".join(lines)
