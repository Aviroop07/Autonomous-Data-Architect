from src.orchestration.stage1.models import Output
from src.util.documentation import format_atomic_facts

def document(output: Output) -> str:
    """
    Converts Stage 1 Output into formatted Markdown.
    """
    lines = []
    lines.append("## Stage 1: Technical Enrichment & Atomization")
    lines.append(f"\n**Identified Domain**: {output.domain}")
    lines.append(f"**Analytical Goal**: {output.analytical_goal}")

    lines.append("\n<details><summary>View Original Natural Language Description</summary>")
    lines.append(f"\n{output.original_nl}")
    lines.append("\n</details>")

    lines.append("\n### Enrichment Iterations")
    for i, it in enumerate(output.iterations):
        lines.append(f"\n#### Iteration {i+1}")

        # Format integrity report with markdown
        report = it.integrity_report
        if report:
            status = "✅ SAFE" if report.is_safe else "❌ ISSUES DETECTED"
            lines.append(f"#### Integrity Status: {status}")

            def format_issues_local(title, issues):
                if not issues: return
                lines.append(f"\n**{title}:**")
                for iss in issues:
                    f_id = f" (Fact {iss.fact_id})" if iss.fact_id else ""
                    lines.append(f"- [{iss.severity.upper()}] {iss.description}{f_id}")

            format_issues_local("Missing Information", report.missing_information)
            format_issues_local("Introduced Information", report.introduced_information)
            format_issues_local("Changed Constraints", report.changed_constraints)
        else:
            lines.append("No integrity report.")

        facts_str = format_atomic_facts(it.extracted_output.facts)
        lines.append(f"\n<details><summary>View Atomic Facts ({len(it.extracted_output.facts)})</summary>\n")
        lines.append(facts_str)
        lines.append("\n</details>")

        if it.extracted_output.definitions:
            lines.append(f"\n<details><summary>View Technical Definitions ({len(it.extracted_output.definitions)})</summary>\n")
            for term in it.extracted_output.definitions:
                lines.append(f"- **{term.term}**: {term.definition} (Source: {term.citation or 'Research Agent'})")
            lines.append("\n</details>")

    lines.append("\n### Final Atomized Facts")
    lines.append(format_atomic_facts(output.final_facts))

    return "\n".join(lines)
