import ast
from pathlib import Path


def test_bot_reuses_a_single_ocr_pipeline() -> None:
    """Protect against recreating all Paddle models once per Telegram photo."""
    source = Path("app/bot/router.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    pipeline_assignments = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "id", None) == "RecognitionPipeline"
    ]
    assert len(pipeline_assignments) == 1
    assert "recognition_pipeline.recognize" in source
