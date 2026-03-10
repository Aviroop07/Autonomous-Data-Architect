from src.orchestration.stage5.models import GenerationPlan, TableGenerationStrategy

def _format_flags(table: TableGenerationStrategy) -> str:
    if not table.structural_flags:
        return ""
    return " · ".join(f"`{f.value}`" for f in table.structural_flags)

def document(plan: GenerationPlan) -> str:
    """Renders the generation plan into markdown documentation."""
    doc = ["## Stage 5: Simulation Strategy & Dependency Resolution\n"]
    doc.append(f"**Total Estimated Dataset Size**: {plan.total_expected_volume:,} rows\n")
    doc.append(f"### Generation Justification\n{plan.generation_sequence_justification}\n")
    
    doc.append("### 1. Generation Sequence & Strategy")
    doc.append("| Order | Table | Strategy | Flags | Est. Rows | Logic |")
    doc.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for table in sorted(plan.ordered_tables, key=lambda x: x.order):
        flags = _format_flags(table)
        doc.append(
            f"| {table.order} | `{table.table_name}` | `{table.strategy_type.value}` | {flags} "
            f"| {table.target_row_count:,} | {table.logic_summary} |"
        )
        
        # Add nested details for Support Columns & Conditional Logic
        if table.support_columns or table.conditional_logics:
            doc.append("\n<details><summary>View Behavioral Implementation Details</summary>\n")
            if table.support_columns:
                doc.append("**Support Columns (Latent Variables)**:")
                for sc in table.support_columns:
                    doc.append(f"- `{sc.column_name}`: {sc.description} *(Logic: {sc.logic})*")
            if table.conditional_logics:
                doc.append("\n**Conditional Logic Mappings**:")
                for cl in table.conditional_logics:
                    doc.append(f"- **If `{cl.support_column_name}`**: Affects `{', '.join(cl.target_column_names)}`. {cl.effect_description}")
            doc.append("\n</details>\n")
    
    doc.append("\n### 2. Dependency Graph")
    # ... (Rest remains the same)
    doc.append("```mermaid")
    doc.append("graph TD")
    for table in plan.ordered_tables:
        if not table.dependencies:
            doc.append(f"    ROOT --> {table.table_name}")
        else:
            for dep in table.dependencies:
                doc.append(f"    {dep} --> {table.table_name}")
    doc.append("```\n")
    
    return "\n".join(doc)
