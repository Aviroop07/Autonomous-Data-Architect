from pydantic import BaseModel, Field
from typing import List


class GeneratedGenerator(BaseModel):
    """The final synthetic data generator code artifact."""
    filename: str = Field(..., description="Name of the generated file, e.g. 'generator.py'")
    python_code: str = Field(..., description="Complete, self-contained Python source code for the generator.")
    external_dependencies: List[str] = Field(
        ...,
        description="List of pip-installable package names required (e.g. ['numpy', 'pandas', 'scipy'])."
    )
    execution_instructions: str = Field(
        ...,
        description="Brief instructions for how to run the generated script."
    )
