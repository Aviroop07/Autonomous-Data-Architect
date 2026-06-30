import asyncio
from src.orchestration.stage1.entry import orchestrate as stage1_orchestrate
from src.orchestration.stage2.entry import orchestrate as stage2_orchestrate
from src.util.config.ablation import AblationConfig

COMPLEX_NL = """
Design a database for a global hospital management network. 
We need to track Patients, who have a unique medical record number (MRN), full name, date of birth, and blood type.
Patients can have multiple Appointments with Doctors. Each Appointment has a scheduled date, time, and status (e.g. Completed, Cancelled).
Doctors have a unique employee ID, specialty, and emergency contact number. Doctors can belong to multiple Departments.
Departments have a unique department code, name, location, and an annual budget.
We also need to track Prescriptions. A Doctor prescribes a Medication to a Patient during an Appointment. The Prescription has a dosage, frequency, and start date.
Medications have a generic name, brand name, and manufacturer.
Additionally, track Medical Procedures. A Procedure is performed on a Patient by a Doctor in a specific Department. It has a procedure code, description, and total cost.
A procedure requires multiple pieces of Equipment. Equipment has an asset tag, type, and next maintenance date.
"""


async def run():
    print("==================================================")
    print("--- STARTING STAGE 1 (Fact Extraction) ---")
    print("==================================================")

    # We use ablation_config to enable everything normally
    s1_out, s1_tokens = await stage1_orchestrate(nl_description=COMPLEX_NL)

    print(f"\n[Stage 1 Tokens]: {s1_tokens}")
    print(f"[Extracted Facts]: {len(s1_out.final_facts)}")
    for f in s1_out.final_facts:
        seg = (
            f' | seg=[{f.start_char}:{f.end_char}] "{f.segment_text[:40]}"'
            if f.segment_text
            else " | (standalone)"
        )
        print(f"  [{f.id}] {f.fact} (Tags: {[t.value for t in f.tags]}){seg}")

    print("\n==================================================")
    print(f"--- FACT CLUSTERS (graph chunker): {len(s1_out.plan.chunks)} chunks ---")
    print("==================================================")
    for i, chunk in enumerate(s1_out.plan.chunks, 1):
        print(f"  CHUNK {i} ({len(chunk)} facts): ids={sorted(cf.id for cf in chunk)}")

    print("\n==================================================")
    print("--- STARTING STAGE 2 (Schema Generation) ---")
    print("==================================================")

    s2_out, s2_tokens, registry = await stage2_orchestrate(
        plan=s1_out.plan,
        facts=s1_out.final_facts,
        domain=s1_out.domain,
        analytical_goal=s1_out.analytical_goal,
        nl_query=COMPLEX_NL,
        ablation_config=AblationConfig.full(),
    )

    print("\n==================================================")
    print(f"--- ER SHARDS (per-chunk schemas): {len(s2_out.segments)} shards ---")
    print("==================================================")
    for i, shard in enumerate(s2_out.segments, 1):
        tbls = ", ".join(t.name for t in shard.tables)
        print(f"  SHARD {i}: {len(shard.tables)} tables -> {tbls}")

    print(f"\n[Stage 2 Tokens]: {s2_tokens}")

    print("\n==================================================")
    print("--- FINAL RELATIONAL SCHEMA ---")
    print("==================================================")
    final_schema = s2_out.final_global_schema
    print(
        final_schema.model_dump_json(indent=2)
        if final_schema
        else "(no schema produced)"
    )

    print("\n==================================================")
    print("--- TABLE FACT REGISTRY (Provenance) ---")
    print("==================================================")
    for table_name, fact_ids in registry.table_to_facts.items():
        print(f"  {table_name}: {fact_ids}")

    print("\n==================================================")
    print("--- UNCOVERED FACTS ---")
    print("==================================================")
    print(s2_out.uncovered_fact_ids)

    print("\n==================================================")
    print("--- COMPLIANCE CERTIFIER REPORT ---")
    print("==================================================")
    cert_patches = s2_out.cert_report.patches if s2_out.cert_report else []
    print(f"Patches applied: {len(cert_patches)}")
    for p in cert_patches:
        reason = getattr(p, "reason", None) or getattr(p, "action", type(p).__name__)
        print(f"  - {reason}")


if __name__ == "__main__":
    asyncio.run(run())
