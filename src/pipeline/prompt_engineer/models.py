from pydantic import BaseModel, Field
from typing import List, Optional, Any, Generic, TypeVar

T = TypeVar('T', bound=BaseModel)

class PromptExample(BaseModel, Generic[T]):
    scenario: str = Field(description="A description of the situation or context for this example.")
    instance: T = Field(description="An instance of the expected output object (Pydantic model).")
    reasoning: Optional[str] = Field(description="Optional reasoning for why this output is correct/appropriate.")

class PromptStructure(BaseModel, Generic[T]):
    role: str = Field(description="The persona the agent should adopt.")
    task: str = Field(description="The primary objective the agent must achieve.")
    input_data: str = Field(description="Description of the input the agent will process.")
    output_format: str = Field(..., description="Detailed narrative of expected output fields and their semantic meanings.")
    guidelines: List[str] = Field(..., description="List of at least 5 strict rules or domain best practices.")
    examples: List[PromptExample[T]] = Field(..., min_length=2, description="At least 2 high-quality scenario-based examples. ABSOLUTELY MANDATORY.")

    def format_as_text(self) -> str:
        """
        Formats the structured prompt into a standard string block with markdown patterns.
        """
        guidelines_str = "\n".join([f"{i+1}. {g}" for i, g in enumerate(self.guidelines)])
        
        examples_str = ""
        if self.examples:
            examples_str = "\n### EXAMPLES\n"
            for ex in self.examples:
                examples_str += f"\n**Scenario:** {ex.scenario}\n"
                # Use str(instance) to capture custom __str__ logic
                examples_str += f"**Output Instance:**\n{str(ex.instance)}\n"
                if ex.reasoning:
                    examples_str += f"**Reasoning:** {ex.reasoning}\n"
                examples_str += "---\n"

        return f"""### ROLE
**{self.role}**

### TASK
{self.task}

### INPUT
{self.input_data}

### OUTPUT
{self.output_format}

### GUIDELINES
{guidelines_str}
{examples_str}
"""
