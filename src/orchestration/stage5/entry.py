import json
from typing import Optional
from src.pipeline.stage5.agents.strategy_architect import agent as strategy_agent
from src.orchestration.stage5 import documentation

def orchestrate(global_schema_obj, global_registry_obj, strategy_prompt: Optional[str] = None):
    """Orchestrates Stage 5: Generation Planning."""
    
    # 1. Run StrategyArchitect
    # We pass the JSON representation of the schema and registry
    schema_json = json.dumps(global_schema_obj.model_dump(), indent=2)
    registry_json = json.dumps(global_registry_obj.model_dump(), indent=2)
    
    gen_plan, tokens = strategy_agent.plan_generation(
        schema_json=schema_json,
        registry_json=registry_json,
        prompt=strategy_prompt
    )
    
    # 2. Render Documentation
    report = documentation.document(gen_plan)
    
    return gen_plan, report, tokens
