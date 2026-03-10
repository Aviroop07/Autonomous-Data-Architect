from src.orchestration.stage2.models import Output
from src.util.documentation import render_schema_to_base64

def document(output: Output) -> str:
    """
    Converts Stage 2 Output into formatted Markdown.
    """
    lines = []
    lines.append("## Stage 2: Structural Chunking & Schema Generation")
    
    lines.append("\n### Chunked Generation Plan")
    lines.append(f"Generated **{len(output.plan.chunks)}** logical shards based on atomic facts.")
    
    from src.util.documentation import format_atomic_facts
    for i, (segment, chunk) in enumerate(zip(output.segments, output.plan.chunks)):
        lines.append(f"\n#### Schema Shard {i+1}")
        
        # Include the subset of Atomic Facts
        lines.append(f"<details><summary>View Atomic Facts (Shard {i+1})</summary>\n")
        lines.append(format_atomic_facts(chunk))
        lines.append("\n</details>\n")
        
        img_uri = render_schema_to_base64(segment)
        lines.append(f"![Schema Shard {i+1}]({img_uri})")
        
        lines.append(f"\n<details><summary>View SQL (Shard {i+1})</summary>")
        lines.append(f"\n```sql\n{str(segment)}\n```")
        lines.append("\n</details>")
        
    return "\n".join(lines)
