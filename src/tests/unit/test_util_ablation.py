"""Unit tests for src.util.ablation.AblationConfig.

Deterministic, offline. Verifies each factory classmethod toggles exactly
the intended flag(s) and that the model has sane defaults.
"""

from src.util.ablation import AblationConfig


def test_default_construction_all_enabled():
    cfg = AblationConfig()  # pyright: ignore[reportCallIssue]  # pydantic Field defaults
    assert cfg.enable_enrichment is True
    assert cfg.enable_sharding is True
    assert cfg.enable_logical_constraints is True


def test_full_enables_everything():
    cfg = AblationConfig.full()
    assert (cfg.enable_enrichment, cfg.enable_sharding, cfg.enable_logical_constraints) == (
        True,
        True,
        True,
    )


def test_no_enrichment_disables_only_enrichment():
    cfg = AblationConfig.no_enrichment()
    assert cfg.enable_enrichment is False
    assert cfg.enable_sharding is True
    assert cfg.enable_logical_constraints is True


def test_no_sharding_disables_only_sharding():
    cfg = AblationConfig.no_sharding()
    assert cfg.enable_enrichment is True
    assert cfg.enable_sharding is False
    assert cfg.enable_logical_constraints is True


def test_no_logical_constraints_disables_only_logical_constraints():
    cfg = AblationConfig.no_logical_constraints()
    assert cfg.enable_enrichment is True
    assert cfg.enable_sharding is True
    assert cfg.enable_logical_constraints is False


def test_each_factory_disables_exactly_one_flag():
    factories = {
        "no_enrichment": "enable_enrichment",
        "no_sharding": "enable_sharding",
        "no_logical_constraints": "enable_logical_constraints",
    }
    for factory_name, disabled_field in factories.items():
        cfg = getattr(AblationConfig, factory_name)()
        flags = {
            "enable_enrichment": cfg.enable_enrichment,
            "enable_sharding": cfg.enable_sharding,
            "enable_logical_constraints": cfg.enable_logical_constraints,
        }
        # Exactly one flag is False, and it is the expected one.
        false_flags = [k for k, v in flags.items() if v is False]
        assert false_flags == [disabled_field]


def test_factories_return_ablationconfig_instances():
    for factory in (
        AblationConfig.full,
        AblationConfig.no_enrichment,
        AblationConfig.no_sharding,
        AblationConfig.no_logical_constraints,
    ):
        assert isinstance(factory(), AblationConfig)


def test_explicit_override_respected():
    cfg = AblationConfig(enable_enrichment=False, enable_sharding=False, enable_logical_constraints=False)
    assert not any(
        (cfg.enable_enrichment, cfg.enable_sharding, cfg.enable_logical_constraints)
    )
