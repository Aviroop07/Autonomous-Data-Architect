from src.orchestration.stage2.models import Output
from src.util.observability.documentation import render_schema_to_base64, format_atomic_facts, format_fix_history

def document(output: Output) -> str:
    """
    Converts Unified Stage 2 Output (Generation + Refinement) into formatted Markdown.
    """
    lines = []
    lines.append("## [Stage 2] Unified Schema Modeling")

    # PHASE 1: INITIAL GENERATION
    lines.append("\n### Phase 1: Structural Chunking & Initial Generation")
    lines.append(f"Generated **{len(output.plan.chunks)}** initial logical shards.")

    for i, (segment, chunk) in enumerate(zip(output.segments, output.plan.chunks)):
        lines.append(f"\n#### Initial Shard {i+1}")
        lines.append(f"<details><summary>View Atomic Facts (Shard {i+1})</summary>\n")
        lines.append(format_atomic_facts(chunk))
        lines.append("\n</details>")

        # Initial Generation Fix History
        if i < len(output.fix_history) and output.fix_history[i]:
            lines.append(f"\n<details><summary>View Initial Shard {i+1} Correction History</summary>\n")
            lines.append(format_fix_history(output.fix_history[i]))
            lines.append("\n</details>")

        img_uri = render_schema_to_base64(segment)
        lines.append(f"\n![Initial Shard {i+1}]({img_uri})\n")

    if output.merged_schema:
        lines.append("\n---")
        lines.append("\n### Phase 2: Shard Merging & Initial Integration")

        merged_img_uri = render_schema_to_base64(output.merged_schema)
        lines.append(f"\n![Initial Merged Schema]({merged_img_uri})\n")

    # PHASE 3: FINAL GLOBAL SCHEMA
    if output.final_global_schema:
        lines.append("\n---")
        lines.append("\n### Final Refined Global Schema")
        lines.append("The definitive unified schema after iterative refinement and global linking.")

        if output.final_fix_history:
            lines.append("\n<details><summary>View Final Schema Correction History</summary>\n")
            lines.append(format_fix_history(output.final_fix_history))
            lines.append("\n</details>")

        final_img_uri = render_schema_to_base64(output.final_global_schema)
        lines.append(f"\n![Final Refined Schema]({final_img_uri})\n")

        lines.append("\n<details><summary>View Final SQL Schema</summary>")
        lines.append(f"\n```sql\n{str(output.final_global_schema)}\n```")
        lines.append("\n</details>")

    return "\n".join(lines)
