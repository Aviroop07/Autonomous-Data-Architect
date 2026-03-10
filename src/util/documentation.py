import json
import base64
import graphviz
from typing import List, Dict, Any, Optional
from src.pipeline.stage2.models.schema import Schema
from src.pipeline.stage3.models.patch import (
    SchemaPatch, CritiqueReport, ActionTag,
    AddTablePatch, AddRelationshipPatch, DeleteRelationshipPatch,
    ColumnPatch, RenameColumnPatch, MergeTablesPatch, UpsertUniquePatch,
    DeleteTablePatch, UpdatePKPatch
)
from src.pipeline.stage1.models.rephrased_nl import AtomicFact, IntegrityReport
from src.pipeline.stage2.models.corrections import FixHistoryStep

def render_schema_to_base64(schema: Schema) -> str:
    """
    Renders a Schema object to a Graphviz ER diagram and returns a base64 Data URI.
    Standard rendering without highlighting patches.
    """
    dot = graphviz.Digraph(comment='ER Diagram', format='png')
    dot.attr(rankdir='LR', overlap='false', splines='true', bgcolor='white')
    
    for table in schema.tables:
        table_bg = "#e9ecef"
        html_label = f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4" BGCOLOR="#f8f9fa">'
        html_label += f'<TR><TD BGCOLOR="{table_bg}" COLSPAN="2"><B>{table.name}</B></TD></TR>'
        
        for col in table.columns:
            pk_marker = " (PK)" if col.name == table.pk else ""
            html_label += f'<TR><TD ALIGN="LEFT">{col.name}{pk_marker}</TD></TR>'
        
        if table.unique:
            for uq in table.unique:
                html_label += f'<TR><TD ALIGN="LEFT" COLOR="#6c757d"><I>UQ({", ".join(uq.columns)})</I></TD></TR>'
                
        html_label += '</TABLE>>'
        dot.node(table.name, label=html_label, shape='none')

    if schema.relationships:
        for rel in schema.relationships:
            dot.edge(rel.referencing_table, rel.referred_table, 
                     label=rel.referencing_column, 
                     fontsize='10', 
                     color='#4a4a4a')

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
    action = patch.action
    lines = []
    
    emoji = "➕" if "ADD" in action else "➖" if "DELETE" in action else "🔄"
    color = "TIP" if "ADD" in action else "CAUTION" if "DELETE" in action else "IMPORTANT"
    
    lines.append(f"> [!{color}]")
    lines.append(f"> **{emoji} {action}**")
    lines.append(f"> {patch.reason}")
    lines.append("> ")
    
    # Type-safe attribute extraction
    details = []
    
    # Common 'table_name' access
    if isinstance(patch, (ColumnPatch, UpsertUniquePatch, DeleteTablePatch, UpdatePKPatch)):
        details.append(f"Table: `{patch.table_name}`")
        
    # Specific 'column_name' access
    if isinstance(patch, (ColumnPatch, UpdatePKPatch)):
        details.append(f"Column: `{patch.column_name}`")
            
    if isinstance(patch, RenameColumnPatch):
        details.append(f"New Name: `{patch.new_name}`")
    elif isinstance(patch, MergeTablesPatch):
        details.append(f"Source: `{patch.source_table}`")
        details.append(f"Target: `{patch.target_table}`")
    elif isinstance(patch, AddTablePatch):
        details.append(f"Table: `{patch.table_definition.name}`")
        details.append(f"Columns: `{', '.join(patch.table_definition.columns)}` (PK: `{patch.table_definition.pk}`)")
    elif isinstance(patch, (AddRelationshipPatch, DeleteRelationshipPatch)):
        rk = patch.fk_definition
        details.append(f"Relation: `{rk.referencing_table}.{rk.referencing_column} -> {rk.referred_table}`")
    elif isinstance(patch, UpsertUniquePatch):
        details.append(f"Unique Constraint: `{', '.join(patch.unique_definition.columns)}`")
    
    for d in details:
        lines.append(f"> - {d}")
        
    return "\n".join(lines)

def format_critique_report(report: CritiqueReport) -> str:
    """
    Formats a full CritiqueReport into Markdown.
    """
    lines = []
    lines.append(f"### Critique by: {report.agent_name}")
    if report.observations:
        lines.append(f"\n**Observations**:\n{report.observations}\n")
    
    if report.patches:
        lines.append("**Suggested Patches**:")
        for patch in report.patches:
            lines.append(format_schema_patch(patch))
            lines.append("")
    else:
        lines.append("*No patches suggested.*")
        
    return "\n".join(lines)

def format_atomic_facts(facts: List[AtomicFact]) -> str:
    """Formats a list of AtomicFact objects into a Markdown list."""
    if not facts: return "No facts found."
    lines = []
    for f in facts:
        lines.append(f"{f.id}. **[{f.tag}]** {f.fact}")
    return "\n".join(lines)

def format_integrity_report(report: IntegrityReport) -> str:
    """Formats an IntegrityReport into Markdown details."""
    if not report: return "No report."
    
    lines = []
    status = "✅ SAFE" if report.is_safe else "❌ ISSUES DETECTED"
    lines.append(f"#### Integrity Status: {status}")
    
    def format_issues(title, issues):
        if not issues: return
        lines.append(f"\n**{title}:**")
        for iss in issues:
            f_id = f" (Fact {iss.fact_id})" if iss.fact_id else ""
            lines.append(f"- [{iss.severity.upper()}] {iss.description}{f_id}")

    format_issues("Missing Information", report.missing_information)
    format_issues("Introduced Information", report.introduced_information)
    format_issues("Changed Constraints", report.changed_constraints)
    
    return "\n".join(lines)

def format_fix_history(history: List[FixHistoryStep]) -> str:
    """Formats the hardened FixHistoryStep list."""
    if not history: return "No correction history."
    
    lines = []
    for step in history:
        lines.append(f"#### Attempt {step.attempt}")
        if step.errors:
            lines.append("**Errors Found:**")
            for err in step.errors:
                lines.append(f"- `{err}`")
        
        if step.corrections:
            lines.append("\n**Corrections Applied:**")
            for c in step.corrections:
                status_emoji = "✅" if c.status == "fixed" else "⚠️" if c.status == "deferred" else "❌"
                lines.append(f"- {status_emoji} **{c.status.upper()}**: {c.description or c.error_message}")
    
    return "\n".join(lines)

def format_patch_repair_history(history: List[Any]) -> str:
    """
    Formats a list of repair steps (e.g. PatchRepairStep).
    Logic is kept generic to avoid circular imports with orchestration models.
    """
    if not history: return ""
    
    lines = []
    lines.append("\n### Automated Patch Repair Audit")
    lines.append("The following patches failed validation and were automatically repaired by the Repair Agent.")
    
    for i, step in enumerate(history):
        lines.append(f"\n#### Repair Attempt {i+1}")
        
        # Access attributes generically (works for PatchRepairStep)
        errors = getattr(step, 'original_errors', [])
        if errors:
            lines.append("**Validation Errors:**")
            for err in errors:
                lines.append(f"- `[{getattr(err, 'error_type', 'Error')}]` {', '.join(getattr(err, 'errors', ['Unknown error']))}")
        
        repaired = getattr(step, 'repaired_patches', [])
        if repaired:
            lines.append("\n**Repaired Patches (Summary):**")
            for p in repaired:
                lines.append(f"- {p.get('action', 'Patch')}: {p.get('reason', 'No reason provided')}")
                
    return "\n".join(lines)
