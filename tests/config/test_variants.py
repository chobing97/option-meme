"""Tests for config/variants.py — Label/Model 변형 설정 검증."""

from config.variants import (
    LABEL_CONFIGS,
    LABEL_CONFIGS_5M,
    MODEL_CONFIGS,
    MODEL_CONFIGS_5M,
    get_label_configs,
    get_model_configs,
)


class TestLabelConfigs:
    """1분봉 Label Config 테스트."""

    def test_has_l1_l2_l3(self):
        assert "L1" in LABEL_CONFIGS
        assert "L2" in LABEL_CONFIGS
        assert "L3" in LABEL_CONFIGS

    def test_all_have_shift_key(self):
        for label_id, cfg in LABEL_CONFIGS.items():
            assert "shift" in cfg, f"{label_id}에 shift 키가 없음"

    def test_l1_l2_shift_1(self):
        assert LABEL_CONFIGS["L1"]["shift"] == 1
        assert LABEL_CONFIGS["L2"]["shift"] == 1

    def test_l3_shift_0(self):
        assert LABEL_CONFIGS["L3"]["shift"] == 0

    def test_all_have_required_keys(self):
        required = {"prominence_pct", "width", "distance", "shift"}
        for label_id, cfg in LABEL_CONFIGS.items():
            assert required.issubset(cfg.keys()), f"{label_id}에 필수 키 누락"


class TestLabelConfigs5M:
    """5분봉 Label Config 테스트."""

    def test_exists_with_l1_l2_l3(self):
        assert "L1" in LABEL_CONFIGS_5M
        assert "L2" in LABEL_CONFIGS_5M
        assert "L3" in LABEL_CONFIGS_5M

    def test_all_have_shift_key(self):
        for label_id, cfg in LABEL_CONFIGS_5M.items():
            assert "shift" in cfg, f"5M {label_id}에 shift 키가 없음"

    def test_all_have_required_keys(self):
        required = {"prominence_pct", "width", "distance", "shift"}
        for label_id, cfg in LABEL_CONFIGS_5M.items():
            assert required.issubset(cfg.keys()), f"5M {label_id}에 필수 키 누락"


class TestModelConfigs5M:
    """5분봉 Model Config 테스트."""

    def test_exists_with_m1_to_m4(self):
        for m in ["M1", "M2", "M3", "M4"]:
            assert m in MODEL_CONFIGS_5M, f"MODEL_CONFIGS_5M에 {m} 없음"

    def test_all_have_required_keys(self):
        required = {"gbm_lookback", "lstm_lookback", "fill_method"}
        for model_id, cfg in MODEL_CONFIGS_5M.items():
            assert required.issubset(cfg.keys()), f"5M {model_id}에 필수 키 누락"


class TestGetLabelConfigs:
    """get_label_configs 헬퍼 테스트."""

    def test_1m_returns_label_configs(self):
        assert get_label_configs("1m") is LABEL_CONFIGS

    def test_5m_returns_label_configs_5m(self):
        assert get_label_configs("5m") is LABEL_CONFIGS_5M

    def test_default_returns_1m(self):
        assert get_label_configs() is LABEL_CONFIGS


class TestGetModelConfigs:
    """get_model_configs 헬퍼 테스트."""

    def test_1m_returns_model_configs(self):
        assert get_model_configs("1m") is MODEL_CONFIGS

    def test_5m_returns_model_configs_5m(self):
        assert get_model_configs("5m") is MODEL_CONFIGS_5M

    def test_default_returns_1m(self):
        assert get_model_configs() is MODEL_CONFIGS


class TestInvalidTimeframe:
    """잘못된 timeframe 입력 시 에러 검증."""

    def test_get_label_configs_invalid(self):
        import pytest
        with pytest.raises(ValueError, match="지원하지 않는 timeframe"):
            get_label_configs("15m")

    def test_get_model_configs_invalid(self):
        import pytest
        with pytest.raises(ValueError, match="지원하지 않는 timeframe"):
            get_model_configs("1M")
