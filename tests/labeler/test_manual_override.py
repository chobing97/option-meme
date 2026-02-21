"""Tests for apply_manual_overrides()."""

from unittest.mock import patch

import pandas as pd
import pytest

from src.labeler.label_generator import apply_manual_overrides


@pytest.fixture
def auto_labeled_df() -> pd.DataFrame:
    """자동 레이블링 결과 샘플."""
    return pd.DataFrame(
        {
            "symbol": ["AAPL"] * 5,
            "datetime": pd.date_range("2025-01-02 09:30", periods=5, freq="min"),
            "close": [150.0, 151.0, 152.0, 151.5, 150.5],
            "label": [0, 1, 0, 0, 2],
        }
    )


@pytest.fixture
def manual_override_df() -> pd.DataFrame:
    """수작업 오버라이드 데이터 (label 변경 2건)."""
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "datetime": pd.to_datetime(["2025-01-02 09:31", "2025-01-02 09:33"]),
            "label": [0, 2],  # 09:31: peak→neither, 09:33: neither→trough
        }
    )


def test_override_applied(auto_labeled_df, manual_override_df, tmp_path):
    """수작업 파일이 있으면 해당 행의 label이 오버라이드된다."""
    manual_path = tmp_path / "us_manual.parquet"
    manual_override_df.to_parquet(manual_path, index=False)

    with patch("src.labeler.label_generator.LABELED_MANUAL_DIR", tmp_path):
        result = apply_manual_overrides(auto_labeled_df, "us")

    # 09:31 (idx 1): 1 → 0
    assert result.loc[1, "label"] == 0
    # 09:33 (idx 3): 0 → 2
    assert result.loc[3, "label"] == 2
    # 나머지는 변경 없음
    assert result.loc[0, "label"] == 0
    assert result.loc[2, "label"] == 0
    assert result.loc[4, "label"] == 2


def test_no_manual_file(auto_labeled_df, tmp_path):
    """수작업 파일이 없으면 원본 그대로 반환."""
    with patch("src.labeler.label_generator.LABELED_MANUAL_DIR", tmp_path):
        result = apply_manual_overrides(auto_labeled_df, "us")

    pd.testing.assert_frame_equal(result, auto_labeled_df)


def test_empty_manual_file(auto_labeled_df, tmp_path):
    """수작업 파일이 비어있으면 원본 그대로 반환."""
    empty = pd.DataFrame(columns=["symbol", "datetime", "label"])
    manual_path = tmp_path / "us_manual.parquet"
    empty.to_parquet(manual_path, index=False)

    with patch("src.labeler.label_generator.LABELED_MANUAL_DIR", tmp_path):
        result = apply_manual_overrides(auto_labeled_df, "us")

    pd.testing.assert_frame_equal(result, auto_labeled_df)


def test_no_label_manual_column_remains(auto_labeled_df, manual_override_df, tmp_path):
    """머지 후 label_manual 임시 컬럼이 남아있지 않아야 한다."""
    manual_path = tmp_path / "us_manual.parquet"
    manual_override_df.to_parquet(manual_path, index=False)

    with patch("src.labeler.label_generator.LABELED_MANUAL_DIR", tmp_path):
        result = apply_manual_overrides(auto_labeled_df, "us")

    assert "label_manual" not in result.columns
