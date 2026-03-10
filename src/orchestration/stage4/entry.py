from typing import List, Optional
from src.orchestration.stage1.models import Output as Stage1Output
from src.orchestration.stage3.models import Output as Stage3Output
from src.orchestration.stage4.models import GlobalDistributionRegistry
from src.pipeline.stage4.agents.cardinality_extractor import agent as cardinality_extractor
from src.pipeline.stage4.agents.distribution_extractor import agent as distribution_extractor
from src.pipeline.stage4.agents.statistical_mapper import agent as statistical_mapper

def orchestrate(
    stage1_output: Stage1Output,
    stage3_output: Stage3Output,
    model: Optional[str] = None
) -> GlobalDistributionRegistry:
    """
    Executes the tiered Stage 4 extraction process:
    Pass 1: Explicit Cardinalities
    Pass 2: Univariate Kernels
    Pass 3: Statistical Mapping to Schema
    """
    
    # 1. Pass 1: Global Cardinality Extraction
    # We use all final facts from Stage 1 and the global schema for mapping
    cardinalities, card_tokens = cardinality_extractor.extract_cardinalities(
        facts=stage1_output.final_facts,
        schema_json=stage3_output.global_schema.model_dump_json(),
        model=model
    )
    
    # 2. Pass 2: Global Behavioral & Distributional Extraction
    extracted_data, kernel_tokens = distribution_extractor.extract_distributions(
        facts=stage1_output.final_facts,
        model=model
    )
    
    # 3. Pass 3: Mapping to Refined Global Schema
    mappings, mapping_tokens = statistical_mapper.map_distributions(
        global_schema=stage3_output.global_schema,
        cardinalities=cardinalities,
        kernels=extracted_data.kernels,
        transitions=extracted_data.transitions,
        processes=extracted_data.processes,
        policies=extracted_data.policies,
        model=model
    )
    
    # 4. Aggregation
    return GlobalDistributionRegistry(
        all_cardinalities=cardinalities,
        all_kernels=extracted_data.kernels,
        all_transitions=extracted_data.transitions,
        all_processes=extracted_data.processes,
        all_policies=extracted_data.policies,
        all_mappings=mappings
    )
