"""
Large-scale end-to-end test: stages 1 + 2 only.
Hospital network management system -- rich domain, many entities.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

NL = """
A global hospital network management system for a chain of 200 hospitals across 40 countries.
The system tracks patients from registration through discharge, billing, and post-discharge follow-up.

Each patient has a unique medical record number, full name, date of birth, biological sex, blood type,
nationality, preferred language, marital status, emergency contact name and phone, primary insurance
provider, secondary insurance provider, insurance policy numbers, and a home address with street, city,
state, country, and postal code. Patients are enrolled at a specific hospital but can receive care at
any hospital in the network. The system supports both inpatient and outpatient visits.

Each visit has a visit type, admission date and time, discharge date and time, admitting diagnosis,
discharge diagnosis, discharge disposition, attending physician, admitting physician, and a list of
consulting physicians. Inpatient visits are associated with a ward, a room, and a bed number.
Bed occupancy history must be preserved, including the dates a patient occupied each bed.

Wards have a name, floor number, wing, capacity, ward type (ICU, general, maternity, oncology,
pediatrics, cardiac), and a head nurse. Each hospital has a hospital code, full legal name,
registration number, street address, city, state, country, postal code, phone number, fax, email,
total bed count, ICU bed count, operating theater count, accreditation status, accreditation body,
accreditation expiry date, founding year, hospital type (general, specialty, teaching, trauma center),
and a parent organization if it belongs to a group.

The network includes departments per hospital such as cardiology, neurology, orthopedics, emergency
medicine, radiology, pathology, pharmacy, and administration. Each department has a department code,
name, type, floor, extension number, head of department, and budget allocation for the fiscal year.

The system tracks all medical staff: physicians, nurses, pharmacists, lab technicians, radiologists,
and administrative staff. Each staff member has an employee ID, full name, date of birth, national ID,
contact details, professional license number, license expiry date, specialization, department,
employment type (full-time, part-time, contract), employment start and end date, salary grade, bank
account number, and tax identification number. Physicians have a primary specialization and up to three
secondary specializations, board certification status, certification body, and certification expiry.
Nurses have a nursing level (RN, LPN, NP) and ward assignment.

The appointment scheduling module tracks appointment ID, patient, physician, department, hospital,
scheduled date and time, appointment type (routine, follow-up, urgent, specialist referral), status
(scheduled, confirmed, completed, cancelled, no-show), reason for visit, duration in minutes, and
whether in-person or telemedicine. If telemedicine, the platform and session URL are recorded.

Referrals between physicians and hospitals are tracked with source physician, target physician,
referring hospital, receiving hospital, reason, priority (routine or urgent), date issued, and date
accepted.

The clinical module tracks diagnoses, procedures, medications, lab tests, and radiology studies.
Each diagnosis is linked to an ICD-10 code, description, diagnosis date, diagnosing physician,
certainty level (confirmed or suspected), and primary or secondary flag. Procedures use CPT codes and
include procedure name, performing physician, assistant physicians, anesthesiologist, procedure date
and time, duration, operating theater, outcome, and complications.

Prescriptions have a prescription ID, prescribing physician, patient, visit, medication name, generic
name, brand name, drug class, form (tablet, injection, syrup), strength, dosage instructions,
frequency, route of administration, start and end date, refill count, and substitution permitted flag.
The pharmacy records each dispensing event with pharmacist, date, quantity dispensed, lot number, and
expiry date. Medication administrations in the ward record the nurse, time, actual dose, and adverse
reactions.

The drug formulary tracks NDC code, generic name, brand names, drug class, controlled substance
schedule, storage requirements, and drug interactions with severity levels (contraindicated, major,
moderate, minor).

Lab orders include ordering physician, patient, visit, panel name, individual tests, urgency (routine
or STAT), and specimen type. Lab results record the technician, result date and time, each test value
with units, reference range, and abnormal flag, plus an overall interpretation. Radiology orders cover
modality (X-ray, CT, MRI, PET, ultrasound), body region, clinical indication, ordering physician,
assigned radiologist, appointment, and contrast requirement. Radiology reports store radiologist,
date, findings, impression, and recommendation.

The billing module records charge items linked to a visit, service type (procedure, lab, medication,
room charge), charge code, quantity, unit price, total amount, insurance claim status, approved
amount, patient responsibility, and payment status. Insurance claims have a claim ID, insurance
provider, policy number, submission date, status, adjudication date, approved amount, denial reason,
and appeal status. Patient payments record payment method, amount, date, and receipt number.

Medical records include clinical notes with note type (admission note, progress note, operative note,
discharge summary), author, co-signing physician, date, full text, original language, and optional
English translation. Vital signs are recorded per nursing assessment: blood pressure systolic and
diastolic, heart rate, respiratory rate, temperature, oxygen saturation, height, weight, BMI, pain
scale, glucose, timestamp, and recording nurse.

Allergy records track allergen name, type (drug, food, environmental), reaction description, severity
(mild, moderate, severe), onset date, and verification status. Immunization records store vaccine
name, vaccine code, lot number, manufacturer, administration date, administering provider, body site,
dose number in series, and next due date.

Quality and compliance tracking includes incident reports for adverse events, near-misses, and
medication errors. Each incident has date, location, staff involved, patient involved if applicable,
type, description, severity grade, root cause analysis status, and resolution. Hospital accreditation
audit findings are stored per audit cycle, per standard, with finding type, description, corrective
action plan, and due date.

The inventory and procurement module tracks items with item code, name, category (medical supplies,
pharmaceuticals, equipment), unit of measure, reorder point, reorder quantity, current stock level,
and storage location. Purchase orders track vendor, items, quantity, unit price, order date, expected
and actual delivery date, and invoice number. Vendors have a vendor code, legal name, contact
details, category, certifications, and contract terms. Equipment assets are tracked with asset tag,
name, manufacturer, model, serial number, purchase date, warranty expiry, maintenance dates,
assigned department, and operational status.

The staff scheduling module manages shifts with shift code, department, ward, start and end time,
type (day, evening, night), minimum staff count, and actual assigned staff. Staff-shift assignments
record employee, shift, role, and attendance status. Leave requests track employee, leave type
(annual, sick, maternity), start and end date, approval status, and approving manager.

The training module has courses with course code, name, category (clinical, safety, compliance),
delivery method (in-person, e-learning, simulation), duration, mandatory flag, and renewal frequency.
Staff training records store employee, course, completion date, score, pass status, certificate number,
and expiry date.

Patient satisfaction surveys have a visit, survey date, channel (in-person, email, SMS), overall
score (1-10), scores for dimensions (communication, cleanliness, pain management, discharge process),
and open-ended comments.

The system links patients to clinical trial enrollments. Each enrollment has a trial ID, sponsor,
phase, protocol number, enrollment date, arm assignment, consent date, and withdrawal date if
applicable.
"""


async def main() -> None:
    import warnings

    warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

    from src.orchestration.stage1.entry import orchestrate as stage1
    from src.orchestration.stage2.entry import orchestrate as stage2
    from src.util.config.ablation import AblationConfig

    ablation = AblationConfig(enable_enrichment=True, enable_sharding=True)
    out_dir = PROJECT_ROOT / "output" / "runs" / "large_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Test] Starting Stage 1: Fact Extraction")
    print(f"[Test] NL length: {len(NL)} chars\n")
    t0 = time.time()

    s1_output, s1_tokens = await stage1(
        nl_description=NL,
        model="gpt-4o",
        ablation_config=ablation,
    )

    s1_elapsed = time.time() - t0
    facts = s1_output.final_facts
    print(
        f"\n[Stage 1] Done in {s1_elapsed:.1f}s | {s1_tokens} tokens | {len(facts)} facts"
    )

    (out_dir / "stage1_output.json").write_text(
        json.dumps(s1_output.model_dump(), indent=2, default=str), encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print(f"  Stage 1 Summary")
    print("=" * 60)
    print(f"  Facts extracted : {len(facts)}")
    print(f"  Domain          : {s1_output.domain}")
    print(f"  Analytical goal : {s1_output.analytical_goal}")
    print(f"  Tokens          : {s1_tokens}")
    print(f"  Time            : {s1_elapsed:.1f}s")
    tag_counts: dict[str, int] = {}
    for f in facts:
        for tag in f.tags or []:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    print(f"  Tag distribution:")
    for tag, cnt in sorted(tag_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {tag:<30} {cnt}")
    print("=" * 60)

    print("\n[Test] Starting Stage 2: Schema Generation")
    t0 = time.time()

    s2_output, s2_tokens, registry = await stage2(
        facts=facts,
        domain=s1_output.domain,
        analytical_goal=s1_output.analytical_goal,
        model="gpt-4o",
        ablation_config=ablation,
    )

    s2_elapsed = time.time() - t0
    schema = s2_output.final_global_schema
    print(f"\n[Stage 2] Done in {s2_elapsed:.1f}s | {s2_tokens} tokens")

    (out_dir / "stage2_output.json").write_text(
        json.dumps(s2_output.model_dump(), indent=2, default=str), encoding="utf-8"
    )

    validation_errors = schema._validate()

    print("\n" + "=" * 60)
    print(f"  Stage 2 Summary")
    print("=" * 60)
    print(f"  Tables          : {len(schema.tables)}")
    print(f"  Relationships   : {len(schema.relationships or [])}")
    print(f"  Validation errs : {len(validation_errors)}")
    print(f"  Tokens          : {s2_tokens}")
    print(f"  Time            : {s2_elapsed:.1f}s")
    print(f"\n  Tables:")
    for t in sorted(schema.tables, key=lambda x: x.name):
        col_count = len(t.columns)
        fk_count = sum(
            1 for r in (schema.relationships or []) if r.referencing_table == t.name
        )
        print(f"    {t.name:<40} cols={col_count:<4} fks={fk_count}")
    if validation_errors:
        print(f"\n  Validation errors (first 20):")
        for err in validation_errors[:20]:
            print(f"    - {err}")
    if s2_output.cycles:
        print(f"\n  Cycles detected: {s2_output.cycles}")
    print("=" * 60)
    print(f"\n[Test] Output saved to: {out_dir}")
    print(f"[Test] Total tokens: {s1_tokens + s2_tokens}")
    print(f"[Test] Total time  : {s1_elapsed + s2_elapsed:.1f}s\n")


if __name__ == "__main__":
    asyncio.run(main())
