from src.orchestration.stage1.models import Output
from src.util.documentation import format_atomic_facts, format_integrity_report

def document(output: Output) -> str:
    """
    Converts Stage 1 Output into formatted Markdown.
    """
    lines = []
    lines.append("## Stage 1: Technical Enrichment & Atomization")
    lines.append(f"\n**Identified Domain**: {output.domain}")
    lines.append(f"**Analytical Goal**: {output.analytical_goal}")
    
    lines.append("\n### Enrichment Iterations")
    for i, it in enumerate(output.iterations):
        lines.append(f"\n#### Iteration {i+1}")
        assert it.integrity_report is not None
        lines.append(format_integrity_report(it.integrity_report))
        
        lines.append(f"\n<details><summary>View Atomic Facts (Iteration {i+1})</summary>")
        lines.append(f"\n{format_atomic_facts(it.rephrased_output.rephrased_text)}")
        lines.append("\n</details>")
            
    lines.append("\n### Final Atomized Facts")
    lines.append(format_atomic_facts(output.final_facts))
    
    return "\n".join(lines)
