"""Which structured-output method each provider gets, and why.

Every entry here is a MEASURED capability of a real provider rather than a
preference, so a change to one should be deliberate. Measured 2026-07-30 by
running the real ER-extraction schema against each provider
(experiments/provider_capacity_probe.py):

  cerebras  function_calling  38 entities from 121 facts, 82.5s   -- works
  deepseek  json_mode         37 entities from 121 facts, 144.2s  -- works
  groq      function_calling  400 "Failed to call a function"     -- WALL
  groq      json_mode         works at 30 facts, output-limited at 121

The groq row is the reason this file exists. Under function_calling that provider
could not run the pipeline AT ALL -- an unconditional rejection of the schema, not
a quality problem. Under json_mode it runs, and its remaining limit is output
SIZE, which chunking already addresses.
"""

from __future__ import annotations

from src.util.core.providers import PROVIDERS


class TestGroqUsesJsonMode:
    def test_groq_is_json_mode_not_function_calling(self):
        """Regression: function_calling here means groq cannot run at all."""
        assert PROVIDERS["groq"].method == "json_mode"

    def test_the_reason_is_recorded_next_to_the_choice(self):
        """This is a measured capability, so the evidence must travel with it --
        a future reader must not 'tidy' groq back to function_calling."""
        import inspect

        from src.util.core import providers

        source = inspect.getsource(providers)
        assert "Failed to call a function" in source


class TestOtherProvidersUnchanged:
    def test_cerebras_keeps_function_calling(self):
        """cerebras succeeds on the same schemas groq rejects, so the groq change
        must not be generalised to every function_calling provider."""
        assert PROVIDERS["cerebras"].method == "function_calling"

    def test_gemini_and_deepseek_keep_json_mode(self):
        assert PROVIDERS["gemini"].method == "json_mode"
        assert PROVIDERS["deepseek"].method == "json_mode"

    def test_vllm_keeps_json_schema(self):
        """The only provider that can ENFORCE a schema during sampling; see
        agent.py for why that is not interchangeable with json_mode."""
        assert PROVIDERS["vllm"].method == "json_schema"

    def test_every_provider_declares_a_known_method(self):
        allowed = {"function_calling", "json_mode", "json_schema"}
        for name, spec in PROVIDERS.items():
            assert spec.method in allowed, f"{name} has method {spec.method!r}"
