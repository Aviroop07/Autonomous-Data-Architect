from src.orchestration.stage4.models import GlobalDistributionRegistry

def document(registry: GlobalDistributionRegistry) -> str:
    """
    Renders the Stage 4 extraction registry into markdown format.
    """
    md = "## Stage 4: Data Distribution Modelling\n\n"
    
    # 1. Explicit Cardinalities
    md += "### 1. Explicit Cardinalities (Row Volumes)\n"
    if not registry.all_cardinalities:
        md += "_No explicit cardinalities found._\n\n"
    else:
        md += "| Entity | Table | Value | Source Context |\n"
        md += "| :--- | :--- | :--- | :--- |\n"
        for c in registry.all_cardinalities:
            table_val = f"`{c.target_table}`" if c.target_table else "_unmapped_"
            count_val = f"`{c.raw_count_description}`"
            if c.approximate_count is not None:
                count_val += f" (~{c.approximate_count:,})"
            md += f"| {c.entity_name} | {table_val} | {count_val} | {c.context} |\n"
        md += "\n"
        
    # 2. Univariate Kernels
    md += "### 2. Univariate Distributions (Kernels)\n"
    if not registry.all_kernels:
        md += "_No univariate kernels identified._\n\n"
    else:
        for k in registry.all_kernels:
            md += f"- **Fact {k.fact_id}**: `{k.kernel_type}`\n"
            md += f"  - Parameters: `{k.parameters}`\n"
            md += f"  - Description: {k.description}\n"
        md += "\n"

    # 3. State Transitions
    md += "### 3. State Transitions (Life-cycles)\n"
    if not registry.all_transitions:
        md += "_No state transitions identified._\n\n"
    else:
        for t in registry.all_transitions:
            md += f"- **Fact {t.fact_id} ({t.entity_name})**: `{t.target_column}`\n"
            md += f"  - States: `{t.valid_states}`\n"
            md += f"  - Transitions: `{t.transitions}`\n"
            md += f"  - Logic: {t.logic}\n"
        md += "\n"

    # 4. Event Processes
    md += "### 4. Event Processes (Arrival Logic)\n"
    if not registry.all_processes:
        md += "_No event processes identified._\n\n"
    else:
        for p in registry.all_processes:
            md += f"- **Fact {p.fact_id} ({p.event_table})**: `{p.process_type}`\n"
            md += f"  - Intensity: {p.intensity_logic}\n"
            md += f"  - Params: `{p.parameters}`\n"
        md += "\n"

    # 5. Conditional Policies (Backward Modeling)
    md += "### 5. Conditional Policies & Latents\n"
    if not registry.all_policies:
        md += "_No conditional policies identified._\n\n"
    else:
        for po in registry.all_policies:
            md += f"- **Fact {po.fact_id} ({po.condition_source})**: Backward Logic\n"
            md += f"  - Affects: `{po.affected_entities}`\n"
            md += f"  - Policy: {po.policy_logic}\n"
            md += f"  - Latent Hint: `{po.latent_variable_hint}`\n"
        md += "\n"
        
    # 6. Statistical Mappings
    md += "### 6. Schema Mappings\n"
    if not registry.all_mappings:
        md += "_No mappings generated._\n\n"
    else:
        md += "| Table | Column | Goal | Logic |\n"
        md += "| :--- | :--- | :--- | :--- |\n"
        for m in registry.all_mappings:
            # Determine Goal from source lists
            goal = "Custom"
            if any(k.fact_id == m.fact_id for k in registry.all_kernels): goal = "Distribution"
            elif any(t.fact_id == m.fact_id for t in registry.all_transitions): goal = "State Machine"
            elif any(p.fact_id == m.fact_id for p in registry.all_processes): goal = "Event Process"
            elif any(po.fact_id == m.fact_id for po in registry.all_policies): goal = "Conditional Policy"
            elif any(c.fact_id == m.fact_id for c in registry.all_cardinalities): goal = "Cardinality"
            
            md += f"| `{m.target_table}` | `{m.target_column}` | `{goal}` | {m.mapping_logic} |\n"
        md += "\n"
        
    return md
