"""
Tests for analytics.py and simulation.py modules.
Author: Gagandeep Kapoor
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from analytics import (
    FraudPatternAnalyser, SegmentAnalyser, TimeSeriesAnalyser,
    AnomalyDetector, MetricTracker, CohortAnalyser, ReportBuilder,
)
from simulation import (
    SimulationConfig, TransactionVolumeModel, FraudAttackSimulator,
    EconomicModel, AnalystQueueModel, ScenarioLibrary, BootstrapAnalyser,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_transactions():
    rng = np.random.default_rng(42)
    n = 500
    return pd.DataFrame({
        "transaction_id": [f"TXN{i:05d}" for i in range(n)],
        "amount": rng.lognormal(4, 1.5, n),
        "is_fraud": rng.binomial(1, 0.05, n),
        "merchant_category": rng.choice(["retail","travel","food","online","atm"], n),
        "channel": rng.choice(["card_present","card_not_present","contactless"], n),
        "customer_age": rng.integers(18, 80, n),
        "hour": rng.integers(0, 24, n),
        "day_of_week": rng.integers(0, 7, n),
        "country_risk": rng.choice(["low","medium","high"], n),
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
        "customer_id": rng.integers(1, 100, n),
        "fraud_score": rng.uniform(0, 1, n),
        "velocity_1h": rng.integers(0, 10, n),
        "velocity_24h": rng.integers(0, 30, n),
    })


@pytest.fixture
def fraud_only(sample_transactions):
    return sample_transactions[sample_transactions["is_fraud"] == 1].copy()


@pytest.fixture
def metric_tracker():
    return MetricTracker(window_size=50)


# ---------------------------------------------------------------------------
# TestFraudPatternAnalyser
# ---------------------------------------------------------------------------

class TestFraudPatternAnalyser:
    def setup_method(self):
        self.analyser = FraudPatternAnalyser()

    def test_init_defaults(self):
        assert hasattr(self.analyser, "min_support")
        assert hasattr(self.analyser, "patterns")

    def test_analyse_by_category(self, sample_transactions):
        result = self.analyser.analyse_by_category(sample_transactions, "merchant_category")
        assert isinstance(result, pd.DataFrame)
        assert "fraud_rate" in result.columns
        assert len(result) > 0

    def test_analyse_by_category_returns_sorted(self, sample_transactions):
        result = self.analyser.analyse_by_category(sample_transactions, "merchant_category")
        rates = result["fraud_rate"].values
        assert all(rates[i] >= rates[i+1] for i in range(len(rates)-1))

    def test_analyse_by_channel(self, sample_transactions):
        result = self.analyser.analyse_by_channel(sample_transactions)
        assert isinstance(result, pd.DataFrame)
        assert "channel" in result.columns or "fraud_rate" in result.columns

    def test_temporal_pattern(self, sample_transactions):
        result = self.analyser.temporal_pattern(sample_transactions, "hour")
        assert isinstance(result, pd.Series) or isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_amount_distribution(self, sample_transactions):
        result = self.analyser.amount_distribution(sample_transactions)
        assert isinstance(result, dict)
        assert "fraud" in result
        assert "legit" in result

    def test_correlation_matrix(self, sample_transactions):
        numeric_cols = ["amount","fraud_score","velocity_1h","velocity_24h","is_fraud"]
        result = self.analyser.correlation_matrix(sample_transactions[numeric_cols])
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == result.shape[1]

    def test_top_risk_segments(self, sample_transactions):
        result = self.analyser.top_risk_segments(sample_transactions, top_n=3)
        assert isinstance(result, list)
        assert len(result) <= 3

    def test_velocity_analysis(self, sample_transactions):
        result = self.analyser.velocity_analysis(sample_transactions)
        assert isinstance(result, dict)

    def test_geographic_risk(self, sample_transactions):
        result = self.analyser.geographic_risk(sample_transactions)
        assert isinstance(result, pd.DataFrame)

    def test_empty_dataframe_raises(self):
        empty = pd.DataFrame(columns=["amount","is_fraud","merchant_category"])
        with pytest.raises((ValueError, KeyError)):
            self.analyser.analyse_by_category(empty, "merchant_category")

    def test_pattern_summary(self, sample_transactions):
        result = self.analyser.pattern_summary(sample_transactions)
        assert isinstance(result, dict)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# TestSegmentAnalyser
# ---------------------------------------------------------------------------

class TestSegmentAnalyser:
    def setup_method(self):
        self.analyser = SegmentAnalyser()

    def test_init(self):
        assert hasattr(self.analyser, "segments")

    def test_define_amount_segments(self, sample_transactions):
        result = self.analyser.define_amount_segments(sample_transactions)
        assert "amount_segment" in result.columns

    def test_amount_segment_coverage(self, sample_transactions):
        result = self.analyser.define_amount_segments(sample_transactions)
        assert result["amount_segment"].isna().sum() == 0

    def test_segment_fraud_rates(self, sample_transactions):
        df = self.analyser.define_amount_segments(sample_transactions)
        rates = self.analyser.segment_fraud_rates(df, "amount_segment")
        assert isinstance(rates, pd.DataFrame)
        assert "fraud_rate" in rates.columns

    def test_cross_segment(self, sample_transactions):
        result = self.analyser.cross_segment(
            sample_transactions, "merchant_category", "channel"
        )
        assert isinstance(result, pd.DataFrame)

    def test_high_risk_segments(self, sample_transactions):
        result = self.analyser.high_risk_segments(sample_transactions, threshold=0.0)
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 0

    def test_segment_volume(self, sample_transactions):
        df = self.analyser.define_amount_segments(sample_transactions)
        vol = self.analyser.segment_volume(df, "amount_segment")
        assert isinstance(vol, pd.Series)

    def test_compare_segments(self, sample_transactions):
        df = self.analyser.define_amount_segments(sample_transactions)
        result = self.analyser.compare_segments(df, "amount_segment")
        assert isinstance(result, pd.DataFrame)

    def test_customer_age_segments(self, sample_transactions):
        result = self.analyser.define_age_segments(sample_transactions)
        assert "age_segment" in result.columns

    def test_segment_lift(self, sample_transactions):
        df = self.analyser.define_amount_segments(sample_transactions)
        lift = self.analyser.segment_lift(df, "amount_segment")
        assert isinstance(lift, pd.DataFrame)
        assert "lift" in lift.columns

    def test_segment_report(self, sample_transactions):
        result = self.analyser.segment_report(sample_transactions)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# TestTimeSeriesAnalyser
# ---------------------------------------------------------------------------

class TestTimeSeriesAnalyser:
    def setup_method(self):
        self.analyser = TimeSeriesAnalyser(freq="1h")

    def test_init(self):
        assert self.analyser.freq == "1h"

    def test_resample_fraud_rate(self, sample_transactions):
        result = self.analyser.resample_fraud_rate(
            sample_transactions, timestamp_col="timestamp", freq="6h"
        )
        assert isinstance(result, pd.Series)

    def test_rolling_fraud_rate(self, sample_transactions):
        result = self.analyser.rolling_fraud_rate(
            sample_transactions, timestamp_col="timestamp", window=24
        )
        assert isinstance(result, pd.Series)

    def test_detect_spikes(self, sample_transactions):
        ts = self.analyser.resample_fraud_rate(
            sample_transactions, timestamp_col="timestamp", freq="6h"
        )
        spikes = self.analyser.detect_spikes(ts, z_threshold=2.0)
        assert isinstance(spikes, pd.Series)
        assert spikes.dtype == bool

    def test_trend_decomposition(self, sample_transactions):
        ts = self.analyser.resample_fraud_rate(
            sample_transactions, timestamp_col="timestamp", freq="6h"
        )
        result = self.analyser.trend_decomposition(ts)
        assert isinstance(result, dict)
        assert "trend" in result

    def test_hourly_heatmap_data(self, sample_transactions):
        result = self.analyser.hourly_heatmap_data(sample_transactions)
        assert isinstance(result, pd.DataFrame)

    def test_day_of_week_pattern(self, sample_transactions):
        result = self.analyser.day_of_week_pattern(sample_transactions)
        assert isinstance(result, pd.Series)
        assert len(result) == 7

    def test_volume_vs_fraud_correlation(self, sample_transactions):
        corr = self.analyser.volume_vs_fraud_correlation(
            sample_transactions, timestamp_col="timestamp", freq="6h"
        )
        assert isinstance(corr, float)
        assert -1.0 <= corr <= 1.0

    def test_forecast_naive(self, sample_transactions):
        ts = self.analyser.resample_fraud_rate(
            sample_transactions, timestamp_col="timestamp", freq="6h"
        )
        forecast = self.analyser.forecast_naive(ts, steps=5)
        assert isinstance(forecast, pd.Series)
        assert len(forecast) == 5

    def test_empty_series_spike_detection(self):
        empty = pd.Series(dtype=float)
        spikes = self.analyser.detect_spikes(empty, z_threshold=2.0)
        assert len(spikes) == 0


# ---------------------------------------------------------------------------
# TestAnomalyDetector
# ---------------------------------------------------------------------------

class TestAnomalyDetector:
    def setup_method(self):
        self.detector = AnomalyDetector(contamination=0.05)

    def test_init(self):
        assert self.detector.contamination == 0.05

    def test_fit_predict(self, sample_transactions):
        numeric = sample_transactions[["amount","fraud_score","velocity_1h"]].values
        labels = self.detector.fit_predict(numeric)
        assert len(labels) == len(numeric)
        assert set(labels).issubset({-1, 1})

    def test_anomaly_fraction(self, sample_transactions):
        numeric = sample_transactions[["amount","fraud_score","velocity_1h"]].values
        labels = self.detector.fit_predict(numeric)
        frac = (labels == -1).mean()
        assert frac < 0.2

    def test_score_samples(self, sample_transactions):
        numeric = sample_transactions[["amount","fraud_score","velocity_1h"]].values
        self.detector.fit(numeric)
        scores = self.detector.score_samples(numeric)
        assert len(scores) == len(numeric)

    def test_threshold_at_percentile(self, sample_transactions):
        numeric = sample_transactions[["amount","fraud_score","velocity_1h"]].values
        self.detector.fit(numeric)
        thresh = self.detector.threshold_at_percentile(5)
        assert isinstance(thresh, float)

    def test_detect_in_dataframe(self, sample_transactions):
        feature_cols = ["amount","fraud_score","velocity_1h"]
        result = self.detector.detect_in_dataframe(sample_transactions, feature_cols)
        assert "anomaly_label" in result.columns

    def test_explain_anomaly(self, sample_transactions):
        numeric = sample_transactions[["amount","fraud_score","velocity_1h"]].values
        self.detector.fit(numeric)
        explanation = self.detector.explain_anomaly(numeric[0])
        assert isinstance(explanation, dict)

    def test_high_contamination(self, sample_transactions):
        detector = AnomalyDetector(contamination=0.5)
        numeric = sample_transactions[["amount","fraud_score"]].values
        labels = detector.fit_predict(numeric)
        frac = (labels == -1).mean()
        assert abs(frac - 0.5) < 0.1

    def test_single_feature(self, sample_transactions):
        numeric = sample_transactions[["amount"]].values
        labels = self.detector.fit_predict(numeric)
        assert len(labels) == len(numeric)

    def test_reproducibility(self, sample_transactions):
        numeric = sample_transactions[["amount","fraud_score"]].values
        l1 = AnomalyDetector(contamination=0.05).fit_predict(numeric)
        l2 = AnomalyDetector(contamination=0.05).fit_predict(numeric)
        assert np.array_equal(l1, l2)

    def test_empty_input_raises(self):
        with pytest.raises((ValueError, Exception)):
            self.detector.fit_predict(np.array([]).reshape(0, 2))

    def test_anomaly_overlap_with_fraud(self, sample_transactions):
        feature_cols = ["amount","fraud_score","velocity_1h"]
        result = self.detector.detect_in_dataframe(sample_transactions, feature_cols)
        overlap = (result["anomaly_label"] == -1) & (result["is_fraud"] == 1)
        assert overlap.sum() >= 0


# ---------------------------------------------------------------------------
# TestMetricTracker
# ---------------------------------------------------------------------------

class TestMetricTracker:
    def test_record_and_latest(self, metric_tracker):
        metric_tracker.record("precision", 0.85)
        assert metric_tracker.latest("precision") == 0.85

    def test_multiple_records(self, metric_tracker):
        for v in [0.80, 0.82, 0.85, 0.87]:
            metric_tracker.record("recall", v)
        assert metric_tracker.latest("recall") == 0.87

    def test_moving_average(self, metric_tracker):
        for v in [0.80, 0.82, 0.84, 0.86, 0.88]:
            metric_tracker.record("f1", v)
        ma = metric_tracker.moving_average("f1", window=3)
        assert isinstance(ma, float)
        assert abs(ma - (0.84 + 0.86 + 0.88) / 3) < 1e-6

    def test_trend(self, metric_tracker):
        for v in [0.80, 0.82, 0.84, 0.86]:
            metric_tracker.record("auc", v)
        trend = metric_tracker.trend("auc")
        assert trend > 0

    def test_unknown_metric_returns_none(self, metric_tracker):
        assert metric_tracker.latest("nonexistent") is None

    def test_history_length(self, metric_tracker):
        for i in range(60):
            metric_tracker.record("vol", float(i))
        hist = metric_tracker.history("vol")
        assert len(hist) <= 50

    def test_alert_on_drop(self, metric_tracker):
        metric_tracker.record("precision", 0.90)
        metric_tracker.record("precision", 0.70)
        alerts = metric_tracker.check_alerts("precision", drop_threshold=0.10)
        assert alerts is True

    def test_no_alert_stable(self, metric_tracker):
        for v in [0.88, 0.89, 0.88, 0.90]:
            metric_tracker.record("precision", v)
        alerts = metric_tracker.check_alerts("precision", drop_threshold=0.10)
        assert alerts is False


# ---------------------------------------------------------------------------
# TestCohortAnalyser
# ---------------------------------------------------------------------------

class TestCohortAnalyser:
    def setup_method(self):
        self.analyser = CohortAnalyser()

    def test_init(self):
        assert hasattr(self.analyser, "cohorts")

    def test_define_cohorts_by_column(self, sample_transactions):
        cohorts = self.analyser.define_cohorts(sample_transactions, "merchant_category")
        assert isinstance(cohorts, dict)
        assert len(cohorts) > 0

    def test_cohort_fraud_rates(self, sample_transactions):
        cohorts = self.analyser.define_cohorts(sample_transactions, "merchant_category")
        rates = self.analyser.cohort_fraud_rates(cohorts)
        assert isinstance(rates, pd.Series)

    def test_cohort_comparison(self, sample_transactions):
        result = self.analyser.cohort_comparison(
            sample_transactions, "merchant_category", "channel"
        )
        assert isinstance(result, pd.DataFrame)

    def test_retention_proxy(self, sample_transactions):
        result = self.analyser.retention_proxy(
            sample_transactions, customer_col="customer_id", time_col="timestamp"
        )
        assert isinstance(result, pd.DataFrame)

    def test_cohort_size_distribution(self, sample_transactions):
        cohorts = self.analyser.define_cohorts(sample_transactions, "channel")
        sizes = self.analyser.cohort_sizes(cohorts)
        assert isinstance(sizes, pd.Series)
        assert sizes.sum() == len(sample_transactions)

    def test_export_cohort_summary(self, sample_transactions):
        summary = self.analyser.export_summary(sample_transactions, "channel")
        assert isinstance(summary, dict)


# ---------------------------------------------------------------------------
# TestReportBuilder
# ---------------------------------------------------------------------------

class TestReportBuilder:
    def setup_method(self):
        self.builder = ReportBuilder()

    def test_init(self):
        assert hasattr(self.builder, "sections")

    def test_add_section(self):
        self.builder.add_section("Executive Summary", {"total_txns": 10000})
        assert "Executive Summary" in self.builder.sections

    def test_build_returns_dict(self):
        self.builder.add_section("Overview", {"fraud_rate": 0.05})
        report = self.builder.build()
        assert isinstance(report, dict)

    def test_build_includes_metadata(self):
        report = self.builder.build()
        assert "generated_at" in report or "metadata" in report

    def test_multiple_sections(self):
        for s in ["Summary", "Segments", "Trends", "Recommendations"]:
            self.builder.add_section(s, {"key": s})
        report = self.builder.build()
        assert len(report) >= 4

    def test_clear_sections(self):
        self.builder.add_section("Test", {})
        self.builder.clear()
        assert len(self.builder.sections) == 0

    def test_export_json(self, tmp_path):
        self.builder.add_section("Data", {"n": 42})
        path = str(tmp_path / "report.json")
        self.builder.export_json(path)
        import json
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# TestSimulationConfig
# ---------------------------------------------------------------------------

class TestSimulationConfig:
    def test_defaults(self):
        cfg = SimulationConfig()
        assert cfg.n_simulations > 0
        assert cfg.time_horizon_days > 0
        assert cfg.base_fraud_rate > 0

    def test_custom_values(self):
        cfg = SimulationConfig(n_simulations=500, time_horizon_days=90, base_fraud_rate=0.03)
        assert cfg.n_simulations == 500
        assert cfg.time_horizon_days == 90
        assert cfg.base_fraud_rate == 0.03

    def test_invalid_fraud_rate_raises(self):
        with pytest.raises((ValueError, AssertionError)):
            SimulationConfig(base_fraud_rate=1.5)

    def test_invalid_n_sims_raises(self):
        with pytest.raises((ValueError, AssertionError)):
            SimulationConfig(n_simulations=0)

    def test_serialise(self):
        cfg = SimulationConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert "n_simulations" in d

    def test_from_dict(self):
        d = {"n_simulations": 200, "time_horizon_days": 30, "base_fraud_rate": 0.04}
        cfg = SimulationConfig.from_dict(d)
        assert cfg.n_simulations == 200

    def test_repr(self):
        cfg = SimulationConfig()
        assert "SimulationConfig" in repr(cfg)


# ---------------------------------------------------------------------------
# TestTransactionVolumeModel
# ---------------------------------------------------------------------------

class TestTransactionVolumeModel:
    def setup_method(self):
        cfg = SimulationConfig(n_simulations=20, time_horizon_days=30)
        self.model = TransactionVolumeModel(cfg)

    def test_generate_returns_array(self):
        result = self.model.generate()
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2

    def test_shape(self):
        result = self.model.generate()
        assert result.shape[0] == 20
        assert result.shape[1] == 30

    def test_all_positive(self):
        result = self.model.generate()
        assert (result >= 0).all()

    def test_mean_volume(self):
        result = self.model.generate()
        assert result.mean() > 0

    def test_seasonal_effect(self):
        cfg = SimulationConfig(n_simulations=50, time_horizon_days=30)
        m1 = TransactionVolumeModel(cfg, seasonal=True).generate()
        m2 = TransactionVolumeModel(cfg, seasonal=False).generate()
        assert not np.allclose(m1.mean(0), m2.mean(0))

    def test_seed_reproducibility(self):
        cfg = SimulationConfig(n_simulations=10, time_horizon_days=7)
        a = TransactionVolumeModel(cfg, seed=0).generate()
        b = TransactionVolumeModel(cfg, seed=0).generate()
        assert np.allclose(a, b)


# ---------------------------------------------------------------------------
# TestFraudAttackSimulator
# ---------------------------------------------------------------------------

class TestFraudAttackSimulator:
    def setup_method(self):
        cfg = SimulationConfig(n_simulations=20, time_horizon_days=30)
        self.sim = FraudAttackSimulator(cfg)

    def test_simulate_returns_array(self):
        result = self.sim.simulate()
        assert isinstance(result, np.ndarray)

    def test_simulate_shape(self):
        result = self.sim.simulate()
        assert result.shape[0] == 20

    def test_attack_probability_range(self):
        result = self.sim.simulate()
        assert (result >= 0).all() and (result <= 1).all()

    def test_high_intensity_increases_fraud(self):
        cfg = SimulationConfig(n_simulations=100, time_horizon_days=30)
        low = FraudAttackSimulator(cfg, attack_intensity=0.1).simulate().mean()
        high = FraudAttackSimulator(cfg, attack_intensity=0.9).simulate().mean()
        assert high > low


# ---------------------------------------------------------------------------
# TestEconomicModel
# ---------------------------------------------------------------------------

class TestEconomicModel:
    def setup_method(self):
        self.model = EconomicModel(
            avg_transaction_value=150.0,
            chargeback_cost=25.0,
            investigation_cost=15.0,
            false_positive_cost=5.0,
        )

    def test_fraud_loss(self):
        loss = self.model.fraud_loss(n_frauds=100)
        assert loss > 0

    def test_operational_cost(self):
        cost = self.model.operational_cost(n_alerts=200, n_investigations=50)
        assert cost > 0

    def test_net_benefit(self):
        benefit = self.model.net_benefit(
            n_frauds_caught=80, n_frauds_total=100,
            n_false_positives=20, n_alerts=200
        )
        assert isinstance(benefit, float)

    def test_roi(self):
        roi = self.model.roi(
            fraud_prevented=50000.0, total_cost=10000.0
        )
        assert roi == pytest.approx(4.0)

    def test_break_even_fraud_rate(self):
        rate = self.model.break_even_fraud_rate(daily_volume=10000)
        assert 0 < rate < 1


# ---------------------------------------------------------------------------
# TestAnalystQueueModel
# ---------------------------------------------------------------------------

class TestAnalystQueueModel:
    def setup_method(self):
        self.model = AnalystQueueModel(
            n_analysts=5,
            cases_per_analyst_per_day=20,
        )

    def test_queue_depth(self):
        depth = self.model.queue_depth(daily_alerts=200)
        assert isinstance(depth, (int, float))
        assert depth >= 0

    def test_zero_alerts(self):
        depth = self.model.queue_depth(daily_alerts=0)
        assert depth == 0

    def test_overload_detection(self):
        overloaded = self.model.is_overloaded(daily_alerts=500)
        assert overloaded is True

    def test_no_overload(self):
        overloaded = self.model.is_overloaded(daily_alerts=50)
        assert overloaded is False

    def test_time_to_clear(self):
        ttc = self.model.time_to_clear(backlog=100)
        assert ttc >= 0

    def test_optimal_analyst_count(self):
        optimal = self.model.optimal_analyst_count(daily_alerts=200, target_sla_days=1)
        assert isinstance(optimal, int)
        assert optimal > 0


# ---------------------------------------------------------------------------
# TestScenarioLibrary
# ---------------------------------------------------------------------------

class TestScenarioLibrary:
    def setup_method(self):
        self.lib = ScenarioLibrary()

    def test_list_scenarios(self):
        scenarios = self.lib.list_scenarios()
        assert isinstance(scenarios, list)
        assert len(scenarios) > 0

    def test_get_known_scenario(self):
        scenarios = self.lib.list_scenarios()
        s = self.lib.get(scenarios[0])
        assert isinstance(s, SimulationConfig)

    def test_get_unknown_raises(self):
        with pytest.raises((KeyError, ValueError)):
            self.lib.get("nonexistent_scenario_xyz")

    def test_baseline_scenario_exists(self):
        assert "baseline" in self.lib.list_scenarios()

    def test_stress_scenario_exists(self):
        scenarios = self.lib.list_scenarios()
        assert any("stress" in s or "high" in s for s in scenarios)


# ---------------------------------------------------------------------------
# TestBootstrapAnalyser
# ---------------------------------------------------------------------------

class TestBootstrapAnalyser:
    def setup_method(self):
        self.analyser = BootstrapAnalyser(n_bootstrap=200, confidence=0.95)

    def test_init(self):
        assert self.analyser.n_bootstrap == 200
        assert self.analyser.confidence == 0.95

    def test_ci_mean(self):
        data = np.random.default_rng(0).normal(0, 1, 500)
        lo, hi = self.analyser.ci_mean(data)
        assert lo < 0 < hi

    def test_ci_proportion(self):
        data = np.array([0]*950 + [1]*50)
        lo, hi = self.analyser.ci_proportion(data)
        assert 0.03 < lo < hi < 0.08

    def test_ci_width_decreases_with_n(self):
        rng = np.random.default_rng(1)
        small = rng.normal(0, 1, 50)
        large = rng.normal(0, 1, 500)
        lo_s, hi_s = self.analyser.ci_mean(small)
        lo_l, hi_l = self.analyser.ci_mean(large)
        assert (hi_s - lo_s) > (hi_l - lo_l)

    def test_bootstrap_statistic(self):
        data = np.random.default_rng(2).exponential(1, 300)
        result = self.analyser.bootstrap_statistic(data, np.median)
        assert isinstance(result, dict)
        assert "estimate" in result
        assert "ci_low" in result
        assert "ci_high" in result

    def test_compare_groups(self):
        rng = np.random.default_rng(3)
        g1 = rng.normal(0.05, 0.01, 200)
        g2 = rng.normal(0.07, 0.01, 200)
        result = self.analyser.compare_groups(g1, g2)
        assert isinstance(result, dict)
        assert "significant" in result

    def test_significant_difference_detected(self):
        rng = np.random.default_rng(4)
        g1 = rng.normal(0.03, 0.005, 300)
        g2 = rng.normal(0.10, 0.005, 300)
        result = self.analyser.compare_groups(g1, g2)
        assert result["significant"] is True

    def test_no_difference_not_significant(self):
        rng = np.random.default_rng(5)
        g1 = rng.normal(0.05, 0.01, 300)
        g2 = rng.normal(0.05, 0.01, 300)
        result = self.analyser.compare_groups(g1, g2)
        assert result["significant"] is False


# ---------------------------------------------------------------------------
# Parametrised cross-module smoke tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('col', ['merchant_category', 'channel', 'country_risk'])
def test_fraud_pattern_by_column(col, sample_transactions):
    analyser = FraudPatternAnalyser()
    result = analyser.analyse_by_category(sample_transactions, col)
    assert isinstance(result, pd.DataFrame)
    assert 'fraud_rate' in result.columns
    assert len(result) >= 1


@pytest.mark.parametrize('contamination', [0.01, 0.05, 0.10, 0.20])
def test_anomaly_detector_contamination(contamination, sample_transactions):
    detector = AnomalyDetector(contamination=contamination)
    numeric = sample_transactions[['amount', 'fraud_score', 'velocity_1h']].values
    labels = detector.fit_predict(numeric)
    actual_frac = (labels == -1).mean()
    assert abs(actual_frac - contamination) < 0.05


@pytest.mark.parametrize('freq', ['3h', '6h', '12h'])
def test_resample_various_frequencies(freq, sample_transactions):
    analyser = TimeSeriesAnalyser(freq=freq)
    result = analyser.resample_fraud_rate(
        sample_transactions, timestamp_col='timestamp', freq=freq
    )
    assert isinstance(result, pd.Series)
    assert len(result) > 0


@pytest.mark.parametrize('n_analysts,alerts,expected_overload', [
    (2, 500, True),
    (10, 50, False),
    (5, 100, False),
    (1, 300, True),
])
def test_analyst_queue_overload_parametrised(n_analysts, alerts, expected_overload):
    model = AnalystQueueModel(n_analysts=n_analysts, cases_per_analyst_per_day=20)
    assert model.is_overloaded(daily_alerts=alerts) is expected_overload


@pytest.mark.parametrize('window', [5, 10, 20])
def test_metric_tracker_moving_average_windows(window):
    tracker = MetricTracker(window_size=100)
    for i in range(30):
        tracker.record('metric', float(i))
    ma = tracker.moving_average('metric', window=window)
    assert isinstance(ma, float)
    expected = sum(range(30 - window, 30)) / window
    assert abs(ma - expected) < 1e-6
