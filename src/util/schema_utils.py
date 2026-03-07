import json
from typing import Type, Any, Dict
from pydantic import BaseModel

def generate_pydantic_description(model: Type[BaseModel]) -> str:
    """
    Generates a deterministic and beautiful representation of a Pydantic model's 
    structure, including field names, types, and descriptions.
    """
    schema = model.model_json_schema()
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    description_dict = {}
    
    # Sort keys for determinism
    for field_name in sorted(properties.keys()):
        field_info = properties[field_name]
        field_type = field_info.get("type", "unknown")
        field_desc = field_info.get("description", "No description provided.")
        
        is_optional = field_name not in required
        type_str = f"{field_type} {'(Optional)' if is_optional else '(Required)'}"
        
        description_dict[field_name] = {
            "type": type_str,
            "description": field_desc
        }

    return json.dumps(description_dict, indent=4)
