from typing import List
from src.pipeline.stage2.mapper.conceptual_model import (
    CMAttribute,
    ConceptualModel,
    Entity,
    Participant,
    Relationship,
    FunctionalDependency,
)
from src.pipeline.stage2.models.data_types import DataType
from src.pipeline.stage2.mapper.relational_mapper import map_conceptual_to_relational


def test_entity_to_table_with_surrogate_pk():
    cm = ConceptualModel(
        entities=[
            Entity(
                name="Customer",
                attributes=[
                    CMAttribute(name="Name", type=DataType.VARCHAR),
                    CMAttribute(name="Email", type=DataType.VARCHAR),
                ],
            )
        ]
    )
    schema = map_conceptual_to_relational(cm)
    assert len(schema.tables) == 1
    t = schema.tables[0]
    assert t.name == "CUSTOMER"
    assert t.primary_key == ["customer_id"]
    assert len(t.columns) == 3
    col_names = [c.name for c in t.columns]
    assert "customer_id" in col_names
    assert "name" in col_names
    assert "email" in col_names


def test_entity_to_table_with_natural_pk():
    cm = ConceptualModel(
        entities=[
            Entity(
                name="Product",
                attributes=[
                    CMAttribute(name="UPC_Code", type=DataType.VARCHAR),
                    CMAttribute(name="Description", type=DataType.VARCHAR),
                ],
                identifier_attributes=["UPC_Code"],
            )
        ]
    )
    schema = map_conceptual_to_relational(cm)
    t = schema.tables[0]
    assert t.primary_key == ["upc_code"]
    assert len(t.columns) == 2
    assert "upc_code" in [c.name for c in t.columns]


def test_entity_to_table_with_fd_pk():
    cm = ConceptualModel(
        entities=[
            Entity(
                name="Employee",
                attributes=[
                    CMAttribute(name="SSN", type=DataType.VARCHAR),
                    CMAttribute(name="Name", type=DataType.VARCHAR),
                ],
            )
        ],
        functional_dependencies=[
            FunctionalDependency(
                determinant=["Employee.SSN"], dependent=["Employee.Name"]
            )
        ],
    )
    schema = map_conceptual_to_relational(cm)
    t = schema.tables[0]
    assert t.primary_key == ["ssn"]
    assert len(t.columns) == 2


def test_binary_1_n_relationship():
    cm = ConceptualModel(
        entities=[
            Entity(name="Department", identifier_attributes=["Dept_ID"]),
            Entity(name="Employee", identifier_attributes=["Emp_ID"]),
        ],
        relationships=[
            Relationship(
                name="Works_In",
                degree="binary",
                kind="1:N",
                participants=[
                    Participant(entity="Department", cardinality_max=1),
                    Participant(entity="Employee", cardinality_max=None),
                ],
            )
        ],
    )
    schema = map_conceptual_to_relational(cm)
    assert len(schema.tables) == 2
    dept = next(t for t in schema.tables if t.name == "DEPARTMENT")
    emp = next(t for t in schema.tables if t.name == "EMPLOYEE")

    assert "dept_id" in [c.name for c in emp.columns]

    assert len(schema.relationships) == 1
    fk = schema.relationships[0]
    assert fk.referencing_table == "EMPLOYEE"
    assert fk.referencing_column == "dept_id"
    assert fk.referred_table == "DEPARTMENT"


def test_binary_1_1_relationship():
    cm = ConceptualModel(
        entities=[
            Entity(name="Manager", identifier_attributes=["Manager_ID"]),
            Entity(name="Department", identifier_attributes=["Dept_ID"]),
        ],
        relationships=[
            Relationship(
                name="Manages",
                degree="binary",
                kind="1:1",
                participants=[
                    Participant(entity="Manager", cardinality_min=0, cardinality_max=1),
                    Participant(
                        entity="Department", cardinality_min=1, cardinality_max=1
                    ),
                ],
            )
        ],
    )
    schema = map_conceptual_to_relational(cm)
    dept = next(t for t in schema.tables if t.name == "DEPARTMENT")
    assert "manager_id" in [c.name for c in dept.columns]

    fk = schema.relationships[0]
    assert fk.referencing_table == "DEPARTMENT"
    assert fk.referencing_column == "manager_id"

    # Check UNIQUE constraint
    assert dept.unique is not None
    assert any(u.columns == ["manager_id"] for u in dept.unique)


def test_m_n_relationship():
    cm = ConceptualModel(
        entities=[
            Entity(name="Student", identifier_attributes=["Student_ID"]),
            Entity(name="Course", identifier_attributes=["Course_Code"]),
        ],
        relationships=[
            Relationship(
                name="Enrolled_In",
                degree="binary",
                kind="M:N",
                participants=[
                    Participant(entity="Student"),
                    Participant(entity="Course"),
                ],
                attributes=[CMAttribute(name="Grade", type=DataType.VARCHAR)],
            )
        ],
    )
    schema = map_conceptual_to_relational(cm)
    assert len(schema.tables) == 3
    enrolled = next(t for t in schema.tables if t.name == "ENROLLED_IN")

    assert set(enrolled.primary_key) == {"student_id", "course_code"}
    assert "grade" in [c.name for c in enrolled.columns]

    fks = [fk for fk in schema.relationships if fk.referencing_table == "ENROLLED_IN"]
    assert len(fks) == 2


def test_role_pairs_same_entity():
    cm = ConceptualModel(
        entities=[Entity(name="Employee", identifier_attributes=["Emp_ID"])],
        relationships=[
            Relationship(
                name="Supervises",
                degree="binary",
                kind="1:N",
                participants=[
                    Participant(
                        entity="Employee", role="Supervisor", cardinality_max=1
                    ),
                    Participant(
                        entity="Employee", role="Subordinate", cardinality_max=None
                    ),
                ],
            )
        ],
    )
    schema = map_conceptual_to_relational(cm)
    emp = next(t for t in schema.tables if t.name == "EMPLOYEE")
    # Subordinate is the N side
    assert "supervisor_emp_id" in [c.name for c in emp.columns]


def test_multivalued_attribute():
    cm = ConceptualModel(
        entities=[
            Entity(
                name="Person",
                identifier_attributes=["Person_ID"],
                attributes=[CMAttribute(name="Phone", type=DataType.VARCHAR, is_multivalued=True)],
            )
        ]
    )
    schema = map_conceptual_to_relational(cm)
    assert len(schema.tables) == 2
    phone_t = next(t for t in schema.tables if t.name == "PERSON_PHONE")

    assert set(phone_t.primary_key) == {"person_id", "phone"}
    assert len(schema.relationships) == 1
    assert schema.relationships[0].referencing_table == "PERSON_PHONE"


def test_weak_entity():
    cm = ConceptualModel(
        entities=[
            Entity(name="Building", identifier_attributes=["Building_Name"]),
            Entity(
                name="Room",
                identifier_attributes=["Room_Number"],
                is_weak=True,
                owner="Building",
            ),
        ]
    )
    schema = map_conceptual_to_relational(cm)
    assert len(schema.tables) == 2
    room = next(t for t in schema.tables if t.name == "ROOM")

    assert set(room.primary_key) == {"building_name", "room_number"}
    assert "building_name" in [c.name for c in room.columns]

    fk = next(fk for fk in schema.relationships if fk.referencing_table == "ROOM")
    assert fk.referred_table == "BUILDING"


def test_m_n_verb_name_composed_from_participants():
    # A verb/plural relationship name ("Operates") is NOT a clean noun -> the junction is
    # named deterministically from its participants (noun, golden-aligned), not the verb.
    cm = ConceptualModel(
        entities=[
            Entity(name="Flight", identifier_attributes=["Flight_Number"]),
            Entity(name="Crew", identifier_attributes=["Crew_ID"]),
        ],
        relationships=[
            Relationship(
                name="Operates",
                degree="binary",
                kind="M:N",
                participants=[Participant(entity="Flight"), Participant(entity="Crew")],
            )
        ],
    )
    schema = map_conceptual_to_relational(cm)
    junction = next(t for t in schema.tables if t.name not in ("FLIGHT", "CREW"))
    assert junction.name == "CREW_FLIGHT"  # sorted participant entity names
    assert set(junction.primary_key) == {"crew_id", "flight_number"}
    assert "OPERATES" not in {t.name for t in schema.tables}


def test_m_n_clean_noun_name_preserved():
    # A clean singular-noun relationship name is kept as the junction table name.
    cm = ConceptualModel(
        entities=[
            Entity(name="Student", identifier_attributes=["Student_ID"]),
            Entity(name="Course", identifier_attributes=["Course_Code"]),
        ],
        relationships=[
            Relationship(
                name="Enrollment",
                degree="binary",
                kind="M:N",
                participants=[
                    Participant(entity="Student"),
                    Participant(entity="Course"),
                ],
            )
        ],
    )
    schema = map_conceptual_to_relational(cm)
    assert "ENROLLMENT" in {t.name for t in schema.tables}


def test_m_n_collision_disambiguated():
    # Two distinct M:N relationships over the same entity pair, both verb-named, must
    # produce two DISTINCT junction tables (collision disambiguation).
    cm = ConceptualModel(
        entities=[
            Entity(name="User", identifier_attributes=["User_ID"]),
            Entity(name="Post", identifier_attributes=["Post_ID"]),
        ],
        relationships=[
            Relationship(
                name="Likes",
                degree="binary",
                kind="M:N",
                participants=[Participant(entity="User"), Participant(entity="Post")],
            ),
            Relationship(
                name="Reports",
                degree="binary",
                kind="M:N",
                participants=[Participant(entity="User"), Participant(entity="Post")],
            ),
        ],
    )
    schema = map_conceptual_to_relational(cm)
    junctions = [t for t in schema.tables if t.name not in ("USER", "POST")]
    assert len(junctions) == 2
    assert len({t.name for t in junctions}) == 2  # distinct names, no clobber
