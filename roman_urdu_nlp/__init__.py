"""
roman_urdu_nlp — A pipeline for normalizing, transliterating,
and translating Roman Urdu text.

Public API:
    from roman_urdu_nlp import Pipeline
    nlp = Pipeline()
    result = nlp("Muje boht bhok lgi h")
    result.normalized  # -> str
    result.urdu        # -> str (Urdu script)
    result.english     # -> str (English translation)
"""

__version__ = "0.1.0"
__all__ = ["Pipeline", "PipelineResult", "env_flag"]


def __getattr__(name: str):
    if name in ("Pipeline", "PipelineResult"):
        from roman_urdu_nlp.pipeline import Pipeline, PipelineResult  # noqa: F401
        import roman_urdu_nlp as _self
        _self.Pipeline = Pipeline
        _self.PipelineResult = PipelineResult
        return {"Pipeline": Pipeline, "PipelineResult": PipelineResult}[name]
    if name == "env_flag":
        from roman_urdu_nlp.utils import env_flag
        import roman_urdu_nlp as _self
        _self.env_flag = env_flag
        return env_flag
    raise AttributeError(f"module 'roman_urdu_nlp' has no attribute {name!r}")
