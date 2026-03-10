from src.orchestration.stage3.models import Output
from src.util.documentation import (
    format_critique_report,
    format_patch_repair_history
)

def document(output: Output) -> str:
    """
    Converts Stage 3 Output into formatted Markdown with Tiered Refinement passes.
    """
    lines = []
    lines.append("## Stage 3: Distributed Precision 2.0")
    
    lines.append("\n### Domain Style Guide")
    lines.append(f"<details><summary>View Style Guide</summary>\n\n{output.style_guide}\n\n</details>")
    
    lines.append("\n### Shard-Level Refinement History (Tiered Passes)")
    for step in output.shard_steps:
        lines.append(f"\n#### Chunk {step.chunk_index + 1}")
        
        # Pass A: Minimalism
        lines.append(f"\n##### Pass A: Minimalism (Fact Fidelity)")
        lines.append(format_critique_report(step.pass_a_report))
        if step.pass_a_image_uri:
            lines.append(f"\n<details><summary>View Minimalist State</summary>\n\n![]({step.pass_a_image_uri})\n\n```sql\n{str(step.pass_a_schema)}\n```\n</details>")
            
        # Pass B: Realism & Workload
        lines.append(f"\n##### Pass B: Realism & Workload (Industry Hardening)")
        lines.append(format_critique_report(step.pass_b_report))
        if step.pass_b_image_uri:
            lines.append(f"\n<details><summary>View Realistic State</summary>\n\n![]({step.pass_b_image_uri})\n\n```sql\n{str(step.pass_b_schema)}\n```\n</details>")

    lines.append("\n### Global Stitching & Linking")
    if output.merge_image_uri:
        lines.append(f"\n<details><summary>View Initial Merged Schema</summary>\n\n![]({output.merge_image_uri})\n\n</details>")
        
    lines.append("\n## Final Global Schema Result")
    if output.final_image_uri:
        lines.append(f"![Final Global Schema]({output.final_image_uri})")
    lines.append(f"\n<details><summary>View Final SQL</summary>\n\n```sql\n{str(output.global_schema)}\n```\n</details>")
    
    status = "✅ No relational cycles detected." if not output.cycles else f"⚠️ **Cycles Detected/Fixed**: {output.cycles}"
    lines.append(f"\n**Cyclic Dependency Check**: {status}")
    
    return "\n".join(lines)
