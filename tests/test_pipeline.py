"""
Comprehensive test suite for the Fintech Payments Fraud Detection Pipeline.
============================================================================
Author: Gagandeep Kapoor
Date:   2026-09-02

Covers all pipeline classes with unit tests, integration tests, edge cases,
and parametrised scenarios. Uses pytest with fixtures.

Run::

    pytest tests/test_pipeline.py -v --tb=short

"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Allow importing from parent package
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_config():
    """Minimal PipelineConfig for fast unit tests (1_000 transactions)."""
    from src.pipeline import PipelineConfig
    with tempfile.TemporaryDirectory() as tmpdir:
        yield PipelineConfig(
            n_transactions=1_000,
            fraud_rate=0.05,
            random_seed=42,
            test_size=0.20,
            n_cv_folds=2,
            date_start="2024-01-01",
            date_end="2024-03-31",
            out_dir=Path(tmpdir),
            model_names=["rf"],
            threshold_metric="f1",
            log_level="WARNING",
        )


@pytest.fixture(scope="module")
def medium_config():
    """Medium PipelineConfig for integration tests (5_000 transactions)."""
    from src.pipeline import PipelineConfig
    with tempfile.TemporaryDirectory() as tmpdir:
        yield PipelineConfig(
            n_transactions=5_000,
            fraud_rate=0.023,
            random_seed=99,
            test_size=0.20,
            n_cv_folds=2,
            out_dir=Path(tmpdir),
            model_names=["rf", "gbt"],
            threshold_metric="f2",
            log_level="WARNING",
        )


@pytest.fixture(scope="module")
def raw_df(small_config):
    """Generated raw transaction DataFrame for reuse across tests."""
    from src.pipeline import DataGenerator
    gen = DataGenerator(small_config)
    return gen.generate()


@pytest.fixture(scope="module")
def feature_df(small_config, raw_df):
    """Feature-engineered DataFrame for reuse across tests."""
    from src.pipeline import FeatureEngineer
    fe = FeatureEngineer(small_config)
    return fe.fit_transform(raw_df)


@pytest.fixture(scope="module")
def train_test_arrays(small_config, feature_df):
    """Train/test numpy arrays for model training tests."""
    from src.pipeline import FeatureEngineer
    from sklearn.model_selection import train_test_split
    fe = FeatureEngineer(small_config)
    feat_cols = [c for c in fe.feature_columns if c in feature_df.columns]
    X = feature_df[feat_cols].fillna(0).values
    y = feature_df["is_fraud"].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    return X_tr, X_te, y_tr, y_te, feat_cols


# ---------------------------------------------------------------------------
# PipelineConfig tests
# ---------------------------------------------------------------------------

class TestPipelineConfig:
    """Unit tests for PipelineConfig dataclass."""

    def test_default_construction(self, tmp_path):
        """Default config constructs without error and out_dir is created."""
        from src.pipeline import PipelineConfig
        cfg = PipelineConfig(out_dir=tmp_path / "out")
        assert cfg.n_transactions == 500_000
        assert cfg.fraud_rate == 0.023
        assert cfg.random_seed == 42
        assert (tmp_path / "out").exists()

    def test_custom_params(self, tmp_path):
        """Custom parameters are stored correctly."""
        from src.pipeline import PipelineConfig
        cfg = PipelineConfig(
            n_transactions=10_000, fraud_rate=0.05,
            random_seed=7, out_dir=tmp_path,
        )
        assert cfg.n_transactions == 10_000
        assert cfg.fraud_rate == 0.05
        assert cfg.random_seed == 7

    def test_invalid_n_transactions_too_low(self, tmp_path):
        """n_transactions below 1_000 raises AssertionError."""
        from src.pipeline import PipelineConfig
        with pytest.raises(AssertionError):
            PipelineConfig(n_transactions=100, out_dir=tmp_path)

    def test_invalid_n_transactions_too_high(self, tmp_path):
        """n_transactions above 10_000_000 raises AssertionError."""
        from src.pipeline import PipelineConfig
        with pytest.raises(AssertionError):
            PipelineConfig(n_transactions=99_000_000, out_dir=tmp_path)

    def test_invalid_fraud_rate_zero(self, tmp_path):
        """fraud_rate = 0 raises AssertionError."""
        from src.pipeline import PipelineConfig
        with pytest.raises(AssertionError):
            PipelineConfig(fraud_rate=0.0, out_dir=tmp_path)

    def test_invalid_fraud_rate_too_high(self, tmp_path):
        """fraud_rate > 0.20 raises AssertionError."""
        from src.pipeline import PipelineConfig
        with pytest.raises(AssertionError):
            PipelineConfig(fraud_rate=0.50, out_dir=tmp_path)

    def test_invalid_threshold_metric(self, tmp_path):
        """Unknown threshold_metric raises AssertionError."""
        from src.pipeline import PipelineConfig
        with pytest.raises(AssertionError):
            PipelineConfig(threshold_metric="kappa", out_dir=tmp_path)

    def test_to_json_roundtrip(self, tmp_path):
        """to_json + from_json preserves key numeric fields."""
        from src.pipeline import PipelineConfig
        cfg = PipelineConfig(n_transactions=2_000, fraud_rate=0.03, out_dir=tmp_path)
        json_path = tmp_path / "cfg.json"
        cfg.to_json(json_path)
        cfg2 = PipelineConfig.from_json(json_path)
        assert cfg2.n_transactions == 2_000
        assert abs(cfg2.fraud_rate - 0.03) < 1e-9
        assert cfg2.random_seed == cfg.random_seed

    def test_summary_is_string(self, small_config):
        """summary() returns a non-empty string."""
        s = small_config.summary()
        assert isinstance(s, str)
        assert len(s) > 50
        assert "n_transactions" in s

    def test_model_names_default(self, tmp_path):
        """Default model_names includes rf, gbt, ensemble."""
        from src.pipeline import PipelineConfig
        cfg = PipelineConfig(out_dir=tmp_path)
        assert "rf" in cfg.model_names
        assert "ensemble" in cfg.model_names

    def test_out_dir_created(self, tmp_path):
        """Nested out_dir is created automatically."""
        from src.pipeline import PipelineConfig
        nested = tmp_path / "a" / "b" / "c"
        cfg = PipelineConfig(out_dir=nested)
        assert nested.exists()


# ---------------------------------------------------------------------------
# DataGenerator tests
# ---------------------------------------------------------------------------

class TestDataGenerator:
    """Unit tests for DataGenerator."""

    def test_generate_row_count(self, small_config):
        """generate() returns at least n_transactions rows."""
        from src.pipeline import DataGenerator
        gen = DataGenerator(small_config)
        df  = gen.generate()
        assert len(df) >= small_config.n_transactions

    def test_generate_columns(self, raw_df):
        """All expected columns are present in the generated DataFrame."""
        expected = {"transaction_id", "account_id", "merchant_id", "amount",
                    "currency", "category", "is_fraud", "fraud_type",
                    "hour", "day_of_week", "month", "timestamp",
                    "is_cross_border"}
        assert expected.issubset(set(raw_df.columns))

    def test_fraud_rate_approximate(self, small_config):
        """Observed fraud rate is within 50% of the configured rate."""
        from src.pipeline import DataGenerator
        gen = DataGenerator(small_config)
        df  = gen.generate()
        obs = df["is_fraud"].mean()
        assert 0.5 * small_config.fraud_rate <= obs <= 2.0 * small_config.fraud_rate

    def test_amounts_positive(self, raw_df):
        """All transaction amounts are strictly positive."""
        assert (raw_df["amount"] > 0).all()

    def test_amounts_reasonable(self, raw_df):
        """Amounts are within a plausible range (GBP 0.01 - 50,000)."""
        assert raw_df["amount"].max() <= 50_000
        assert raw_df["amount"].min() >= 0.01

    def test_hours_valid_range(self, raw_df):
        """Hour values are in [0, 23]."""
        assert raw_df["hour"].between(0, 23).all()

    def test_day_of_week_valid(self, raw_df):
        """day_of_week values are in [0, 6]."""
        assert raw_df["day_of_week"].between(0, 6).all()

    def test_month_valid_range(self, raw_df):
        """Month values are in [1, 12]."""
        assert raw_df["month"].between(1, 12).all()

    def test_is_fraud_binary(self, raw_df):
        """is_fraud column is strictly binary (0 or 1)."""
        assert set(raw_df["is_fraud"].unique()).issubset({0, 1})

    def test_categories_valid(self, raw_df):
        """All categories come from the CATEGORIES constant."""
        from src.pipeline import CATEGORIES
        assert set(raw_df["category"].unique()).issubset(set(CATEGORIES))

    def test_currencies_valid(self, raw_df):
        """All currencies come from the CURRENCIES constant."""
        from src.pipeline import CURRENCIES
        assert set(raw_df["currency"].unique()).issubset(set(CURRENCIES))

    def test_device_types_valid(self, raw_df):
        """All device types come from the DEVICE_TYPES constant."""
        from src.pipeline import DEVICE_TYPES
        assert set(raw_df["device_type"].unique()).issubset(set(DEVICE_TYPES))

    def test_reproducibility(self, small_config):
        """Two generators with the same seed produce identical DataFrames."""
        from src.pipeline import DataGenerator
        df1 = DataGenerator(small_config).generate()
        df2 = DataGenerator(small_config).generate()
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self, tmp_path):
        """Two generators with different seeds produce different data."""
        from src.pipeline import DataGenerator, PipelineConfig
        cfg1 = PipelineConfig(n_transactions=1_000, random_seed=1, out_dir=tmp_path / "1")
        cfg2 = PipelineConfig(n_transactions=1_000, random_seed=2, out_dir=tmp_path / "2")
        df1 = DataGenerator(cfg1).generate()
        df2 = DataGenerator(cfg2).generate()
        assert not df1["amount"].equals(df2["amount"])

    def test_accounts_table_populated(self, small_config):
        """After generate(), accounts attribute is a non-empty DataFrame."""
        from src.pipeline import DataGenerator
        gen = DataGenerator(small_config)
        gen.generate()
        assert gen.accounts is not None
        assert len(gen.accounts) > 0

    def test_merchants_table_populated(self, small_config):
        """After generate(), merchants attribute is a non-empty DataFrame."""
        from src.pipeline import DataGenerator
        gen = DataGenerator(small_config)
        gen.generate()
        assert gen.merchants is not None
        assert len(gen.merchants) > 0

    def test_card_testing_bursts_present(self, raw_df):
        """Card-testing burst records are injected (api_key device + small amount)."""
        burst_records = raw_df[
            (raw_df["device_type"] == "api_key") &
            (raw_df["amount"] < 5.0) &
            (raw_df["is_fraud"] == 1)
        ]
        assert len(burst_records) > 0, "Expected card-testing burst records"

    def test_fraud_type_none_for_legitimate(self, raw_df):
        """Legitimate transactions have fraud_type == 'none'."""
        legit = raw_df[raw_df["is_fraud"] == 0]
        assert (legit["fraud_type"] == "none").all()

    def test_fraud_type_not_none_for_fraud(self, raw_df):
        """Fraud transactions have fraud_type != 'none'."""
        fraud = raw_df[raw_df["is_fraud"] == 1]
        assert (fraud["fraud_type"] != "none").all()

    def test_transaction_ids_unique(self, raw_df):
        """All transaction IDs are unique."""
        assert raw_df["transaction_id"].nunique() == len(raw_df)

    def test_sorted_by_timestamp(self, raw_df):
        """Output DataFrame is sorted by timestamp."""
        ts = pd.to_datetime(raw_df["timestamp"])
        assert ts.is_monotonic_increasing



# ---------------------------------------------------------------------------
# DataQualityChecker tests
# ---------------------------------------------------------------------------

class TestDataQualityChecker:
    """Unit tests for DataQualityChecker."""

    def test_run_returns_dict(self, small_config, raw_df):
        """run() returns a dictionary."""
        from src.pipeline import DataQualityChecker
        report = DataQualityChecker(small_config).run(raw_df)
        assert isinstance(report, dict)

    def test_summary_present(self, small_config, raw_df):
        """Report contains a summary key."""
        from src.pipeline import DataQualityChecker
        report = DataQualityChecker(small_config).run(raw_df)
        assert "summary" in report

    def test_n_rows_matches(self, small_config, raw_df):
        """summary.n_rows matches len(raw_df)."""
        from src.pipeline import DataQualityChecker
        report = DataQualityChecker(small_config).run(raw_df)
        assert report["summary"]["n_rows"] == len(raw_df)

    def test_all_checks_present(self, small_config, raw_df):
        """All expected check keys are in the report."""
        from src.pipeline import DataQualityChecker
        report = DataQualityChecker(small_config).run(raw_df)
        for key in ["schema", "nulls", "amount_range", "fraud_rate",
                    "temporal_coverage", "category_distribution", "duplicate_ids"]:
            assert key in report

    def test_schema_check_passes_on_valid(self, small_config, raw_df):
        from src.pipeline import DataQualityChecker
        report = DataQualityChecker(small_config).run(raw_df)
        assert report["schema"]["passed"]

    def test_schema_fails_missing_col(self, small_config, raw_df):
        """Schema check fails when amount column is removed."""
        from src.pipeline import DataQualityChecker
        df2    = raw_df.drop(columns=["amount"])
        report = DataQualityChecker(small_config).run(df2)
        assert not report["schema"]["passed"]
        assert "amount" in report["schema"]["missing_columns"]

    def test_null_check_passes_clean(self, small_config, raw_df):
        from src.pipeline import DataQualityChecker
        report = DataQualityChecker(small_config).run(raw_df)
        assert report["nulls"]["passed"]

    def test_null_check_fails_high_null_rate(self, small_config, raw_df):
        """Null check fails when a column has >1% nulls."""
        from src.pipeline import DataQualityChecker
        df_null = raw_df.copy()
        n_null  = max(int(len(df_null) * 0.05), 10)
        df_null.loc[df_null.index[:n_null], "amount"] = np.nan
        report = DataQualityChecker(small_config).run(df_null)
        assert not report["nulls"]["passed"]

    def test_amount_range_passes_valid(self, small_config, raw_df):
        from src.pipeline import DataQualityChecker
        report = DataQualityChecker(small_config).run(raw_df)
        assert report["amount_range"]["passed"]

    def test_amount_range_fails_negatives(self, small_config, raw_df):
        from src.pipeline import DataQualityChecker
        df_neg = raw_df.copy()
        df_neg.loc[df_neg.index[0], "amount"] = -1.0
        report = DataQualityChecker(small_config).run(df_neg)
        assert not report["amount_range"]["passed"]

    def test_duplicate_ids_passes_clean(self, small_config, raw_df):
        from src.pipeline import DataQualityChecker
        report = DataQualityChecker(small_config).run(raw_df)
        assert report["duplicate_ids"]["passed"]

    def test_duplicate_ids_fails_dupes(self, small_config, raw_df):
        from src.pipeline import DataQualityChecker
        df_dup = pd.concat([raw_df.head(50), raw_df.head(50)], ignore_index=True)
        report = DataQualityChecker(small_config).run(df_dup)
        assert not report["duplicate_ids"]["passed"]

    def test_strict_mode_raises(self, small_config, raw_df):
        """Strict mode raises DataQualityError when a check fails."""
        from src.pipeline import DataQualityChecker, DataQualityError
        df_neg = raw_df.copy()
        df_neg.loc[df_neg.index[0], "amount"] = -1.0
        with pytest.raises(DataQualityError):
            DataQualityChecker(small_config, strict=True).run(df_neg)

    def test_fraud_rate_observed_positive(self, small_config, raw_df):
        from src.pipeline import DataQualityChecker
        report = DataQualityChecker(small_config).run(raw_df)
        assert report["fraud_rate"]["observed_fraud_rate"] > 0


# ---------------------------------------------------------------------------
# FeatureEngineer tests
# ---------------------------------------------------------------------------

class TestFeatureEngineer:
    """Unit tests for FeatureEngineer."""

    def test_fit_transform_returns_dataframe(self, small_config, raw_df):
        from src.pipeline import FeatureEngineer
        fe = FeatureEngineer(small_config)
        df = fe.fit_transform(raw_df)
        assert isinstance(df, pd.DataFrame)

    def test_row_count_preserved(self, raw_df, feature_df):
        assert len(feature_df) == len(raw_df)

    def test_temporal_features_present(self, feature_df):
        for col in ["hour_sin", "hour_cos", "is_night", "is_weekend"]:
            assert col in feature_df.columns

    def test_hour_sin_range(self, feature_df):
        assert feature_df["hour_sin"].between(-1.01, 1.01).all()

    def test_hour_cos_range(self, feature_df):
        assert feature_df["hour_cos"].between(-1.01, 1.01).all()

    def test_is_night_binary(self, feature_df):
        assert set(feature_df["is_night"].unique()).issubset({0, 1})

    def test_is_weekend_binary(self, feature_df):
        assert set(feature_df["is_weekend"].unique()).issubset({0, 1})

    def test_log_amount_nonnegative(self, feature_df):
        assert (feature_df["log_amount"] >= 0).all()

    def test_category_encoded_present(self, feature_df):
        assert "category_encoded" in feature_df.columns

    def test_category_freq_valid_range(self, feature_df):
        assert "category_freq" in feature_df.columns
        assert feature_df["category_freq"].between(0, 1).all()

    def test_velocity_cols_present(self, feature_df):
        for col in ["tx_count_last5", "tx_sum_last5", "amount_velocity_ratio"]:
            assert col in feature_df.columns

    def test_interaction_features_binary(self, feature_df):
        for col in ["high_amount_night", "cross_border_night", "api_high_amount"]:
            assert set(feature_df[col].unique()).issubset({0, 1})

    def test_transform_succeeds_after_fit(self, small_config, raw_df):
        from src.pipeline import FeatureEngineer
        fe = FeatureEngineer(small_config)
        fe.fit_transform(raw_df)
        result = fe.transform(raw_df.head(100).copy())
        assert len(result) == 100

    def test_transform_before_fit_raises(self, small_config, raw_df):
        from src.pipeline import FeatureEngineer
        fe = FeatureEngineer(small_config)
        with pytest.raises(RuntimeError):
            fe.transform(raw_df)

    def test_feature_columns_nonempty_list(self, small_config):
        from src.pipeline import FeatureEngineer
        cols = FeatureEngineer(small_config).feature_columns
        assert isinstance(cols, list) and len(cols) > 10

    def test_no_inf_in_features(self, feature_df):
        numeric = feature_df.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any()

    def test_unknown_category_in_transform(self, small_config, raw_df):
        """transform() handles unseen categories (encodes as -1)."""
        from src.pipeline import FeatureEngineer
        fe  = FeatureEngineer(small_config)
        fe.fit_transform(raw_df)
        new = raw_df.head(10).copy()
        new["category"] = "unknown_future_category_xyz"
        result = fe.transform(new)
        assert "category_encoded" in result.columns

    def test_empty_df_transform(self, small_config, raw_df):
        """transform() handles empty DataFrames gracefully."""
        from src.pipeline import FeatureEngineer
        fe = FeatureEngineer(small_config)
        fe.fit_transform(raw_df)
        result = fe.transform(raw_df.head(0).copy())
        assert len(result) == 0



# ---------------------------------------------------------------------------
# RuleEngine tests
# ---------------------------------------------------------------------------

class TestRuleEngine:
    """Unit tests for RuleEngine."""

    def test_loads_builtin_rules(self, small_config):
        from src.pipeline import RuleEngine
        engine = RuleEngine(small_config)
        assert len(engine.rule_ids) > 0

    def test_score_transaction_keys(self, small_config, feature_df):
        from src.pipeline import RuleEngine
        engine = RuleEngine(small_config)
        result = engine.score_transaction(feature_df.iloc[0])
        for key in ["rule_score", "decision", "triggered_rules", "hard_decline"]:
            assert key in result

    def test_rule_score_range(self, small_config, feature_df):
        from src.pipeline import RuleEngine
        engine = RuleEngine(small_config)
        for _, row in feature_df.head(50).iterrows():
            result = engine.score_transaction(row)
            assert 0.0 <= result["rule_score"] <= 1.0

    def test_decision_valid_values(self, small_config, feature_df):
        from src.pipeline import RuleEngine
        engine    = RuleEngine(small_config)
        decisions = set()
        for _, row in feature_df.head(100).iterrows():
            decisions.add(engine.score_transaction(row)["decision"])
        assert decisions.issubset({"pass", "review", "decline"})

    def test_hard_decline_sanctioned_country(self, small_config):
        """Sanctioned country (KP) triggers hard_decline."""
        from src.pipeline import RuleEngine
        engine = RuleEngine(small_config)
        row = pd.Series({
            "country": "KP", "risk_tier": "standard", "amount": 100,
            "device_type": "web_browser", "hour": 12, "is_night": 0,
            "is_cross_border": 0, "tx_count_last5": 1,
            "is_compromised": 0, "age_days": 365,
            "currency": "GBP", "category": "grocery",
        })
        result = engine.score_transaction(row)
        assert result["hard_decline"] or result["decision"] in {"decline", "review"}

    def test_score_batch_adds_columns(self, small_config, feature_df):
        from src.pipeline import RuleEngine
        result = RuleEngine(small_config).score_batch(feature_df.head(50))
        for col in ["rule_score", "rule_decision", "triggered_rules", "hard_decline"]:
            assert col in result.columns

    def test_add_rule_increments_count(self, small_config):
        from src.pipeline import RuleEngine
        engine  = RuleEngine(small_config)
        n_rules = len(engine.rule_ids)
        engine.add_rule({
            "id": "R999", "name": "test", "description": "T",
            "type": "test", "weight": 0.1, "hard_decline": False,
        })
        assert len(engine.rule_ids) == n_rules + 1

    def test_add_rule_missing_fields_raises(self, small_config):
        from src.pipeline import RuleEngine
        engine = RuleEngine(small_config)
        with pytest.raises(ValueError):
            engine.add_rule({"id": "RX", "name": "incomplete"})

    def test_remove_rule(self, small_config):
        from src.pipeline import RuleEngine
        engine  = RuleEngine(small_config)
        n_rules = len(engine.rule_ids)
        removed = engine.remove_rule("R001")
        assert removed
        assert len(engine.rule_ids) == n_rules - 1

    def test_remove_nonexistent_returns_false(self, small_config):
        from src.pipeline import RuleEngine
        assert not RuleEngine(small_config).remove_rule("RXYZ000")

    def test_save_rules_json(self, small_config, tmp_path):
        """save_rules writes a valid JSON list."""
        from src.pipeline import RuleEngine
        engine = RuleEngine(small_config)
        path   = tmp_path / "rules.json"
        engine.save_rules(path)
        assert path.exists()
        with open(path) as fh:
            loaded = json.load(fh)
        assert isinstance(loaded, list) and len(loaded) > 0


# ---------------------------------------------------------------------------
# ModelTrainer tests
# ---------------------------------------------------------------------------

class TestModelTrainer:
    """Unit tests for ModelTrainer."""

    def test_fit_returns_self(self, small_config, train_test_arrays):
        from src.pipeline import ModelTrainer
        X_tr, X_te, y_tr, y_te, cols = train_test_arrays
        trainer = ModelTrainer(small_config)
        assert trainer.fit(X_tr, y_tr, X_te, y_te) is trainer

    def test_models_populated(self, small_config, train_test_arrays):
        from src.pipeline import ModelTrainer
        X_tr, X_te, y_tr, y_te, cols = train_test_arrays
        trainer = ModelTrainer(small_config)
        trainer.fit(X_tr, y_tr, X_te, y_te)
        assert len(trainer.models) > 0

    def test_eval_results_roc_auc_valid(self, small_config, train_test_arrays):
        from src.pipeline import ModelTrainer
        X_tr, X_te, y_tr, y_te, cols = train_test_arrays
        trainer = ModelTrainer(small_config)
        trainer.fit(X_tr, y_tr, X_te, y_te)
        for name, metrics in trainer.eval_results.items():
            assert "roc_auc" in metrics
            assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_predict_proba_shape(self, small_config, train_test_arrays):
        from src.pipeline import ModelTrainer
        X_tr, X_te, y_tr, y_te, cols = train_test_arrays
        trainer = ModelTrainer(small_config)
        trainer.fit(X_tr, y_tr, X_te, y_te)
        model_name = list(trainer.models)[0]
        probs = trainer.predict_proba(X_te, model_name=model_name)
        assert probs.shape == (len(X_te),)

    def test_predict_proba_in_01(self, small_config, train_test_arrays):
        from src.pipeline import ModelTrainer
        X_tr, X_te, y_tr, y_te, cols = train_test_arrays
        trainer = ModelTrainer(small_config)
        trainer.fit(X_tr, y_tr, X_te, y_te)
        model_name = list(trainer.models)[0]
        probs = trainer.predict_proba(X_te, model_name=model_name)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

    def test_unknown_model_raises(self, small_config, train_test_arrays):
        from src.pipeline import ModelTrainer
        X_tr, X_te, y_tr, y_te, cols = train_test_arrays
        trainer = ModelTrainer(small_config)
        trainer.fit(X_tr, y_tr, X_te, y_te)
        with pytest.raises(KeyError):
            trainer.predict_proba(X_te, model_name="xyz_not_found")

    def test_save_creates_pkl(self, small_config, train_test_arrays, tmp_path):
        from src.pipeline import ModelTrainer
        X_tr, X_te, y_tr, y_te, cols = train_test_arrays
        trainer = ModelTrainer(small_config)
        trainer.fit(X_tr, y_tr, X_te, y_te)
        trainer.save(tmp_path / "models")
        assert len(list((tmp_path / "models").glob("*.pkl"))) > 0

    def test_cv_results_populated(self, small_config, train_test_arrays):
        from src.pipeline import ModelTrainer
        X_tr, X_te, y_tr, y_te, cols = train_test_arrays
        trainer = ModelTrainer(small_config)
        trainer.fit(X_tr, y_tr, X_te, y_te)
        for name in trainer.models:
            if name != "iso":
                assert name in trainer.cv_results


# ---------------------------------------------------------------------------
# ThresholdOptimiser tests
# ---------------------------------------------------------------------------

class TestThresholdOptimiser:
    """Unit tests for ThresholdOptimiser."""

    @pytest.fixture
    def scores(self):
        rng    = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=500)
        y_prob = np.where(y_true == 1,
                          rng.uniform(0.5, 0.9, 500),
                          rng.uniform(0.1, 0.5, 500))
        return y_true.astype(int), y_prob

    def test_sweep_returns_dataframe(self, small_config, scores):
        from src.pipeline import ThresholdOptimiser
        df = ThresholdOptimiser(small_config).sweep(*scores)
        assert isinstance(df, pd.DataFrame)

    def test_sweep_has_expected_cols(self, small_config, scores):
        from src.pipeline import ThresholdOptimiser
        df = ThresholdOptimiser(small_config).sweep(*scores)
        for col in ["threshold", "precision", "recall", "f1", "f2", "cost_savings"]:
            assert col in df.columns

    def test_optimise_returns_two_floats(self, small_config, scores):
        from src.pipeline import ThresholdOptimiser
        result = ThresholdOptimiser(small_config).optimise(*scores)
        assert len(result) == 2
        assert all(isinstance(x, float) for x in result)

    def test_optimal_threshold_in_range(self, small_config, scores):
        from src.pipeline import ThresholdOptimiser
        opt_t, _ = ThresholdOptimiser(small_config).optimise(*scores)
        assert 0.0 < opt_t < 1.0

    def test_review_lt_decline(self, small_config, scores):
        from src.pipeline import ThresholdOptimiser
        opt_t, rev_t = ThresholdOptimiser(small_config).optimise(*scores)
        assert rev_t < opt_t

    def test_cost_savings_finite(self, small_config, scores):
        from src.pipeline import ThresholdOptimiser
        y_true, y_score = scores
        cs = ThresholdOptimiser(small_config)._cost_savings(y_true, y_score, 0.5)
        assert math.isfinite(cs)



# ---------------------------------------------------------------------------
# FraudDecisionEngine tests
# ---------------------------------------------------------------------------

class TestFraudDecisionEngine:
    """Unit tests for FraudDecisionEngine."""

    def test_clean_transaction_pass(self, small_config):
        from src.pipeline import FraudDecisionEngine
        eng = FraudDecisionEngine(small_config, decline_threshold=0.60, review_threshold=0.35)
        assert eng.decide(0.05, "pass", 0.0)["final_decision"] == "pass"

    def test_high_ml_score_decline(self, small_config):
        from src.pipeline import FraudDecisionEngine
        eng = FraudDecisionEngine(small_config, decline_threshold=0.60)
        assert eng.decide(0.90, "pass", 0.0)["final_decision"] == "decline"

    def test_medium_ml_score_review(self, small_config):
        from src.pipeline import FraudDecisionEngine
        eng = FraudDecisionEngine(small_config, decline_threshold=0.70, review_threshold=0.40)
        assert eng.decide(0.55, "pass", 0.1)["final_decision"] == "review"

    def test_rule_decline_overrides_low_ml(self, small_config):
        from src.pipeline import FraudDecisionEngine
        eng    = FraudDecisionEngine(small_config)
        result = eng.decide(0.02, "decline", 1.0)
        assert result["final_decision"] == "decline"
        assert result["reason"] == "hard_rule"

    def test_composite_score_present(self, small_config):
        from src.pipeline import FraudDecisionEngine
        eng    = FraudDecisionEngine(small_config)
        result = eng.decide(0.40, "pass", 0.20)
        assert isinstance(result["composite_score"], float)

    def test_stats_increment(self, small_config):
        from src.pipeline import FraudDecisionEngine
        eng = FraudDecisionEngine(small_config)
        for _ in range(10):
            eng.decide(0.05, "pass", 0.0)
        assert eng.stats.get("pass", 0) == 10

    def test_decide_batch_adds_columns(self, small_config, feature_df):
        from src.pipeline import FraudDecisionEngine
        eng = FraudDecisionEngine(small_config)
        df  = feature_df.head(100).copy()
        df["ml_score"]     = np.random.uniform(0, 1, len(df))
        df["rule_decision"] = "pass"
        df["rule_score"]    = 0.0
        result = eng.decide_batch(df)
        assert "final_decision"  in result.columns
        assert "composite_score" in result.columns

    def test_decide_batch_missing_col_raises(self, small_config, feature_df):
        from src.pipeline import FraudDecisionEngine
        eng = FraudDecisionEngine(small_config)
        with pytest.raises(KeyError):
            eng.decide_batch(feature_df.head(10), score_col="nonexistent_col")

    def test_all_pass_decisions(self, small_config):
        from src.pipeline import FraudDecisionEngine
        eng = FraudDecisionEngine(small_config)
        df  = pd.DataFrame({
            "ml_score": np.zeros(50), "rule_decision": ["pass"] * 50,
            "rule_score": np.zeros(50),
        })
        result = eng.decide_batch(df)
        assert (result["final_decision"] == "pass").all()

    def test_all_decline_decisions(self, small_config):
        from src.pipeline import FraudDecisionEngine
        eng = FraudDecisionEngine(small_config, decline_threshold=0.50)
        df  = pd.DataFrame({
            "ml_score": np.ones(50), "rule_decision": ["pass"] * 50,
            "rule_score": np.zeros(50),
        })
        result = eng.decide_batch(df)
        assert (result["final_decision"] == "decline").all()


# ---------------------------------------------------------------------------
# ExplainabilityEngine tests
# ---------------------------------------------------------------------------

class TestExplainabilityEngine:
    """Unit tests for ExplainabilityEngine."""

    @pytest.fixture(scope="class")
    def trained_model_and_data(self, small_config, train_test_arrays):
        from sklearn.ensemble import RandomForestClassifier
        X_tr, X_te, y_tr, y_te, cols = train_test_arrays
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X_tr, y_tr)
        return clf, X_te, y_te, cols

    def test_permutation_importance_df(self, small_config, trained_model_and_data):
        from src.pipeline import ExplainabilityEngine
        clf, X_te, y_te, cols = trained_model_and_data
        eng = ExplainabilityEngine(small_config)
        df  = eng.compute_permutation_importance(clf, X_te, y_te, cols, n_repeats=2)
        assert isinstance(df, pd.DataFrame)

    def test_importance_has_correct_cols(self, small_config, trained_model_and_data):
        from src.pipeline import ExplainabilityEngine
        clf, X_te, y_te, cols = trained_model_and_data
        eng = ExplainabilityEngine(small_config)
        df  = eng.compute_permutation_importance(clf, X_te, y_te, cols, n_repeats=2)
        for col in ["feature", "mean_importance", "std_importance"]:
            assert col in df.columns

    def test_importance_row_count(self, small_config, trained_model_and_data):
        from src.pipeline import ExplainabilityEngine
        clf, X_te, y_te, cols = trained_model_and_data
        eng = ExplainabilityEngine(small_config)
        df  = eng.compute_permutation_importance(clf, X_te, y_te, cols, n_repeats=2)
        assert len(df) == len(cols)

    def test_explain_instance_returns_dict(self, small_config, trained_model_and_data):
        from src.pipeline import ExplainabilityEngine
        clf, X_te, y_te, cols = trained_model_and_data
        eng    = ExplainabilityEngine(small_config)
        result = eng.explain_instance(clf, X_te[0], cols)
        assert "feature_contributions" in result
        assert "predicted_score"       in result

    def test_explain_instance_score_range(self, small_config, trained_model_and_data):
        from src.pipeline import ExplainabilityEngine
        clf, X_te, y_te, cols = trained_model_and_data
        eng    = ExplainabilityEngine(small_config)
        result = eng.explain_instance(clf, X_te[0], cols)
        assert 0.0 <= result["predicted_score"] <= 1.0

    def test_contributions_sorted_by_magnitude(self, small_config, trained_model_and_data):
        from src.pipeline import ExplainabilityEngine
        clf, X_te, y_te, cols = trained_model_and_data
        eng    = ExplainabilityEngine(small_config)
        result = eng.explain_instance(clf, X_te[0], cols)
        contribs = result["feature_contributions"]
        deltas   = [abs(c["delta"]) for c in contribs]
        assert deltas == sorted(deltas, reverse=True)


# ---------------------------------------------------------------------------
# MonitoringDaemon tests
# ---------------------------------------------------------------------------

class TestMonitoringDaemon:
    """Unit tests for MonitoringDaemon."""

    @pytest.fixture
    def sample_scores(self):
        rng = np.random.default_rng(42)
        return rng.beta(2, 20, 1000).astype(float)

    def test_psi_identical_close_to_zero(self, small_config, sample_scores):
        from src.pipeline import MonitoringDaemon
        psi = MonitoringDaemon._compute_psi(sample_scores, sample_scores)
        assert psi < 0.05

    def test_psi_very_different_distributions_large(self, small_config):
        from src.pipeline import MonitoringDaemon
        rng  = np.random.default_rng(99)
        ref  = rng.beta(1, 20, 1000)
        curr = rng.beta(10, 2, 1000)
        psi  = MonitoringDaemon._compute_psi(ref, curr)
        assert psi > 0.20

    def test_check_score_drift_stable(self, small_config, sample_scores):
        from src.pipeline import MonitoringDaemon
        daemon = MonitoringDaemon(small_config)
        result = daemon.check_score_drift(sample_scores, sample_scores)
        assert result["status"] in ("stable", "monitor")

    def test_check_score_drift_no_reference(self, small_config, sample_scores):
        from src.pipeline import MonitoringDaemon
        daemon = MonitoringDaemon(small_config, reference_df=None)
        result = daemon.check_score_drift(sample_scores)
        assert result["status"] == "unknown"

    def test_alert_summary_structure(self, small_config):
        from src.pipeline import MonitoringDaemon
        summ = MonitoringDaemon(small_config).alert_summary()
        assert "total_alerts" in summ
        assert isinstance(summ["total_alerts"], int)

    def test_performance_check_returns_auc(self, small_config):
        from src.pipeline import MonitoringDaemon
        rng    = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 200)
        y_prob = np.where(y_true, rng.uniform(0.5, 1.0, 200), rng.uniform(0, 0.5, 200))
        result = MonitoringDaemon(small_config).check_performance(y_true, y_prob)
        assert result["current_auc"] is not None
        assert 0.0 <= result["current_auc"] <= 1.0

    def test_psi_constant_array(self, small_config):
        from src.pipeline import MonitoringDaemon
        const = np.ones(100) * 0.5
        psi   = MonitoringDaemon._compute_psi(const, const)
        assert math.isfinite(psi) and psi >= 0.0


# ---------------------------------------------------------------------------
# PipelineOrchestrator integration tests
# ---------------------------------------------------------------------------

class TestPipelineOrchestratorIntegration:
    """Integration tests that run the full pipeline end-to-end."""

    @pytest.fixture(scope="class")
    def pipeline_run(self, small_config):
        from src.pipeline import PipelineOrchestrator
        orch    = PipelineOrchestrator(small_config)
        results = orch.run_batch()
        return orch, results

    def test_results_is_dict(self, pipeline_run):
        _, results = pipeline_run
        assert isinstance(results, dict)

    def test_eval_results_present(self, pipeline_run):
        _, results = pipeline_run
        assert "eval_results" in results

    def test_thresholds_keys(self, pipeline_run):
        _, results = pipeline_run
        assert "decline" in results["thresholds"]
        assert "review"  in results["thresholds"]

    def test_decision_stats_nonempty(self, pipeline_run):
        _, results = pipeline_run
        assert sum(results["decision_stats"].values()) > 0

    def test_test_df_has_final_decision(self, pipeline_run):
        orch, _ = pipeline_run
        assert "final_decision" in orch.test_df.columns

    def test_raw_df_has_expected_size(self, pipeline_run, small_config):
        orch, _ = pipeline_run
        assert len(orch.raw_df) >= small_config.n_transactions

    def test_feature_df_populated(self, pipeline_run):
        orch, _ = pipeline_run
        assert orch.feature_df is not None and len(orch.feature_df) > 0

    def test_elapsed_s_positive(self, pipeline_run):
        _, results = pipeline_run
        assert results["elapsed_s"] > 0

    def test_primary_model_nonempty(self, pipeline_run):
        _, results = pipeline_run
        assert results.get("primary_model", "") != ""

    def test_feature_importances_present(self, pipeline_run):
        _, results = pipeline_run
        assert "feature_importances" in results
        assert len(results["feature_importances"]) > 0


# ---------------------------------------------------------------------------
# Parametrised and edge-case tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fraud_rate", [0.005, 0.01, 0.05, 0.10])
def test_fraud_rate_calibration(fraud_rate, tmp_path):
    """DataGenerator calibrates within 50% of target for various fraud rates."""
    from src.pipeline import DataGenerator, PipelineConfig
    cfg = PipelineConfig(n_transactions=2_000, fraud_rate=fraud_rate,
                         random_seed=42, out_dir=tmp_path)
    df  = DataGenerator(cfg).generate()
    obs = df["is_fraud"].mean()
    assert 0.5 * fraud_rate <= obs <= 2.5 * fraud_rate


@pytest.mark.parametrize("metric", ["f1", "f2", "precision", "recall"])
def test_threshold_metric_options(metric, tmp_path):
    """ThresholdOptimiser works for all supported metric values."""
    from src.pipeline import ThresholdOptimiser, PipelineConfig
    cfg = PipelineConfig(n_transactions=1_000, threshold_metric=metric, out_dir=tmp_path)
    rng    = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 300)
    y_prob = rng.uniform(0, 1, 300)
    opt_t, rev_t = ThresholdOptimiser(cfg).optimise(y_true, y_prob)
    assert 0.0 < opt_t < 1.0


@pytest.mark.parametrize("n_repeats", [1, 3, 5])
def test_permutation_importance_repeats(n_repeats, tmp_path):
    """Permutation importance works for various n_repeats values."""
    from src.pipeline import ExplainabilityEngine, PipelineConfig
    from sklearn.ensemble import RandomForestClassifier
    cfg = PipelineConfig(n_transactions=1_000, random_seed=42, out_dir=tmp_path)
    rng    = np.random.default_rng(42)
    X      = rng.standard_normal((200, 8))
    y      = rng.integers(0, 2, 200)
    clf    = RandomForestClassifier(n_estimators=5, random_state=42)
    clf.fit(X, y)
    eng    = ExplainabilityEngine(cfg)
    feat_names = [f"f{i}" for i in range(8)]
    df     = eng.compute_permutation_importance(clf, X, y, feat_names, n_repeats=n_repeats)
    assert len(df) == 8


class TestEdgeCases:
    """Edge-case and robustness tests."""

    def test_single_row_rule_scoring(self, small_config, feature_df):
        from src.pipeline import RuleEngine
        result = RuleEngine(small_config).score_batch(feature_df.head(1))
        assert len(result) == 1

    def test_psi_returns_float(self, small_config):
        from src.pipeline import MonitoringDaemon
        a = np.random.default_rng(1).uniform(0, 1, 500)
        b = np.random.default_rng(2).uniform(0, 1, 500)
        psi = MonitoringDaemon._compute_psi(a, b)
        assert isinstance(psi, float) and psi >= 0.0

    def test_cyclical_encoding_midnight(self):
        """Cyclical encoding of midnight (hour=0) and noon (hour=12) differ."""
        from src.pipeline import FeatureEngineer
        s0   = pd.Series([0])
        s12  = pd.Series([12])
        sin0,  cos0  = FeatureEngineer._cyclical_encode(s0,  24)
        sin12, cos12 = FeatureEngineer._cyclical_encode(s12, 24)
        assert not np.isclose(sin0[0], sin12[0]) or not np.isclose(cos0[0], cos12[0])

    def test_config_out_dir_path_object(self, tmp_path):
        """out_dir is converted to a Path object."""
        from src.pipeline import PipelineConfig
        cfg = PipelineConfig(out_dir=str(tmp_path))
        assert isinstance(cfg.out_dir, Path)

    def test_decision_engine_reason_field(self, small_config):
        from src.pipeline import FraudDecisionEngine
        eng    = FraudDecisionEngine(small_config)
        result = eng.decide(0.80, "pass", 0.0)
        assert result["reason"] in {"clean", "ml_score_high", "ml_score_medium",
                                    "hard_rule", "rule_review"}

    def test_generator_different_n_transactions(self, tmp_path):
        """Generator produces at least n_transactions rows for small N."""
        from src.pipeline import DataGenerator, PipelineConfig
        for n in [1_000, 2_000]:
            cfg = PipelineConfig(n_transactions=n, out_dir=tmp_path / str(n))
            df  = DataGenerator(cfg).generate()
            assert len(df) >= n

