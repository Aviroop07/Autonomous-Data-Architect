from src.orchestration.stage6.models import GeneratedGenerator

def document(generator: GeneratedGenerator) -> str:
    """Renders the generated code information into markdown."""
    doc = ["## Stage 6: Rule Synthesis & Code Generation\n"]
    doc.append(f"**Generated File**: `{generator.filename}`\n")
    
    doc.append("### External Dependencies")
    for dep in generator.external_dependencies:
        doc.append(f"- `{dep}`")
    
    doc.append(f"\n### Execution Instructions\n{generator.execution_instructions}\n")
    
    # NEW: Columnar Interface Info
    doc.append("### create_db() Interface")
    doc.append("Returns a `Dict[str, pd.DataFrame]`. Each table is a fully vectorized DataFrame.")
    
    # NEW: Full Source Code (As requested by user)
    doc.append("### Complete Source Code")
    doc.append("```python")
    doc.append(generator.python_code)
    doc.append("```\n")
    
    return "\n".join(doc)
