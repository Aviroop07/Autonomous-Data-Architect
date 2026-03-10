import json
from src.pipeline.stage6.agents.code_generator import agent as generator_agent
from src.orchestration.stage6 import documentation

def orchestrate(global_schema_obj, global_registry_obj, gen_plan_obj):
    """Orchestrates Stage 6: Rule Synthesis & Code Generation."""
    
    # 1. Build rich schema representation that shows every table's columns explicitly
    schema_str = str(global_schema_obj)  # Uses Schema.__str__ which shows TABLE(col1, col2, ...)
    schema_json = json.dumps(global_schema_obj.model_dump(), indent=2)
    registry_json = json.dumps(global_registry_obj.model_dump(), indent=2)
    plan_json = json.dumps(gen_plan_obj.model_dump(), indent=2)
    
    # Combine the human-readable schema with JSON for maximum context
    combined_schema = f"### SCHEMA (Human-Readable)\n{schema_str}\n\n### SCHEMA (Structured JSON)\n{schema_json}"
    
    code_artifact, tokens = generator_agent.generate_code(
        schema_json=combined_schema,
        registry_json=registry_json,
        plan_json=plan_json
    )
    
    # 2. Render Documentation
    report = documentation.document(code_artifact)
    
    return code_artifact, report, tokens
