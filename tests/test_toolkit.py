import pytest

from app.tools.toolkit import ToolError, analyze_csv, safe_calculate


def test_safe_calculate_basic():
    assert safe_calculate("(2 + 3) * 4")["result"] == 20.0


def test_safe_calculate_rejects_code():
    with pytest.raises(ToolError):
        safe_calculate("__import__('os').system('echo bad')")


def test_analyze_csv_returns_prediction_metrics():
    result = analyze_csv("sample_predictions.csv")
    assert result["row_count"] == 8
    assert set(result["prediction_metrics"]) >= {"MAE", "RMSE", "R2"}


def test_path_traversal_is_rejected():
    with pytest.raises(ToolError):
        analyze_csv("../.env")
