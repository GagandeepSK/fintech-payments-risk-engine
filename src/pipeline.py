"""
Fintech Payments Fraud Detection -- Production Pipeline
========================================================
Author: Gagandeep Kapoor
Date:   2026-09-02

End-to-end pipeline wrapping every stage of the fraud-detection system:

    PipelineConfig        -- validated configuration dataclass
    DataGenerator         -- synthetic transaction stream (seeded, reproducible)
    DataQualityChecker    -- schema validation, null/drift checks, reporting
    FeatureEngineer       -- behavioural, temporal, velocity, network features
    RuleEngine            -- deterministic rule-based pre-screening layer
    ModelTrainer          -- RF + GBT ensemble with stratified cross-validation
    ThresholdOptimiser    -- cost-aware decision boundary search
    ExplainabilityEngine  -- permutation importance + PDP + instance explanation
    MonitoringDaemon      -- PSI/CSI drift monitoring, auto-alert emission
    FraudDecisionEngine   -- combines rules + ML + strategy into final verdict
    PipelineOrchestrator  -- chains all stages; batch and streaming modes

Usage
-----
Batch (CLI)::

    python -m src.pipeline --mode batch --n-transactions 500000 --out-dir ./run

Streaming simulation::

    python -m src.pipeline --mode stream --tps 200 --duration 60

Import as library::

    from src.pipeline import PipelineOrchestrator, PipelineConfig
    cfg = PipelineConfig(n_transactions=50_000, random_seed=42)
    orch = PipelineOrchestrator(cfg)
    results = orch.run_batch()
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import math
import os
import random
import sys
import time
import warnings
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import (
    Any, Callable, Dict, Generator, Iterable,
    Iterator, List, Optional, Sequence, Tuple, Union,
)

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import (
    GradientBoostingClassifier, IsolationForest,
    RandomForestClassifier, VotingClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss,
    classification_report, confusion_matrix, f1_score,
    log_loss, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold, cross_val_predict, cross_val_score, train_test_split,
)
from sklearn.preprocessing import (
    LabelEncoder, MinMaxScaler, PowerTransformer,
    RobustScaler, StandardScaler,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s: %(message)s"
_LOG_DATE   = "%Y-%m-%d %H:%M:%S"


def _setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root logger and return a module-level logger."""
    logging.basicConfig(format=_LOG_FORMAT, datefmt=_LOG_DATE, level=level)
    return logging.getLogger(__name__)


logger = _setup_logging()

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

CATEGORIES: List[str] = [
    "grocery", "restaurant", "online_retail", "travel", "entertainment",
    "utility", "healthcare", "fuel", "electronics", "clothing",
    "home_improvement", "subscription", "gaming", "beauty", "sporting_goods",
]

CARD_NETWORKS: List[str] = ["visa", "mastercard", "amex", "discover"]
CURRENCIES:    List[str] = ["GBP", "USD", "EUR", "CAD", "AUD", "JPY", "SGD"]
DEVICE_TYPES:  List[str] = ["mobile_app", "web_browser", "pos_terminal", "atm", "api_key"]
RISK_LEVELS:   List[str] = ["low", "medium", "high", "critical"]
COUNTRIES_HIGH_RISK: List[str] = ["NG", "RO", "UA", "BR", "PK", "IN", "PH", "VN", "ID"]
COUNTRIES_LOW_RISK:  List[str] = ["GB", "US", "DE", "FR", "JP", "CA", "AU", "SG", "NL"]

# Fraud prior probabilities by category (calibrated from UK FCA data)
CATEGORY_FRAUD_PRIOR: Dict[str, float] = {
    "grocery": 0.005, "restaurant": 0.006, "online_retail": 0.035,
    "travel": 0.028, "entertainment": 0.018, "utility": 0.004,
    "healthcare": 0.007, "fuel": 0.012, "electronics": 0.045,
    "clothing": 0.022, "home_improvement": 0.009, "subscription": 0.015,
    "gaming": 0.038, "beauty": 0.011, "sporting_goods": 0.013,
}

# Hour-of-day fraud uplift (fraud peaks 00-05)
HOUR_FRAUD_MULTIPLIER: List[float] = [
    2.5, 3.0, 3.2, 3.5, 3.8, 2.8,
    1.4, 0.7, 0.5, 0.4, 0.4, 0.5,
    0.5, 0.5, 0.5, 0.6, 0.6, 0.7,
    0.7, 0.8, 0.9, 1.1, 1.5, 2.0,
]

# Velocity thresholds (transactions per time window)
VELOCITY_THRESHOLDS: Dict[str, int] = {
    "1min": 3, "5min": 6, "1hr": 12, "24hr": 30, "7day": 80,
}

# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    Centralised, validated configuration for the entire fraud pipeline.

    All parameters have sensible defaults reflecting a mid-sized UK fintech.
    Override any field via the constructor or from a JSON config file using
    :meth:.

    Parameters
    ----------
    n_transactions : int
        Number of synthetic transactions to generate. Default 500_000.
    fraud_rate : float
        Overall fraud prevalence (0 < fraud_rate < 1). Default 0.023.
    random_seed : int
        Master seed propagated to every sub-component. Default 42.
    test_size : float
        Fraction of data held out for evaluation. Default 0.20.
    val_size : float
        Fraction of training data used for validation / early stopping.
    n_cv_folds : int
        Number of stratified cross-validation folds. Default 5.
    date_start : str
        ISO-8601 start date for the generated transaction timeline.
    date_end : str
        ISO-8601 end date.
    out_dir : Path
        Directory for artefacts (models, reports, plots).
    model_names : List[str]
        Which models to train -- subset of {rf, gbt, lr, iso, ensemble}.
    threshold_metric : str
        Metric to maximise during threshold search.
        One of: f1, f2, precision, recall, cost_savings.
    review_cost : float
        Operational cost (GBP) of manually reviewing one transaction.
    false_negative_cost : float
        Expected loss (GBP) from a missed fraud (chargeback + fee).
    false_positive_cost : float
        Revenue impact (GBP) of a wrongly declined legitimate transaction.
    enable_drift_monitoring : bool
        Whether to run PSI/CSI drift checks after scoring.
    psi_threshold : float
        PSI value above which a feature is flagged for drift. Default 0.2.
    streaming_tps : int
        Target transactions-per-second for streaming mode. Default 200.
    streaming_duration_s : int
        How many seconds to run the streaming simulation. Default 60.
    log_level : str
        Python logging level: DEBUG, INFO, WARNING, ERROR. Default INFO.
    """

    n_transactions:          int   = 500_000
    fraud_rate:              float = 0.023
    random_seed:             int   = 42
    test_size:               float = 0.20
    val_size:                float = 0.10
    n_cv_folds:              int   = 5
    date_start:              str   = "2024-01-01"
    date_end:                str   = "2024-12-31"
    out_dir:                 Path  = field(default_factory=lambda: Path("pipeline_output"))
    model_names:             List[str] = field(default_factory=lambda: ["rf", "gbt", "ensemble"])
    threshold_metric:        str   = "f2"
    review_cost:             float = 2.50
    false_negative_cost:     float = 85.00
    false_positive_cost:     float = 12.00
    enable_drift_monitoring: bool  = True
    psi_threshold:           float = 0.20
    streaming_tps:           int   = 200
    streaming_duration_s:    int   = 60
    log_level:               str   = "INFO"

    _feature_cols: List[str] = field(default_factory=list, repr=False)
    _label_col:    str        = field(default="is_fraud",  repr=False)

    def __post_init__(self) -> None:
        assert 1_000 <= self.n_transactions <= 10_000_000,             "n_transactions must be in [1_000, 10_000_000]"
        assert 0.001 <= self.fraud_rate <= 0.20,             "fraud_rate must be in [0.001, 0.20]"
        assert 0.05 <= self.test_size <= 0.40,             "test_size must be in [0.05, 0.40]"
        assert 2 <= self.n_cv_folds <= 20,             "n_cv_folds must be in [2, 20]"
        assert self.threshold_metric in {"f1", "f2", "precision", "recall", "cost_savings"},             "threshold_metric must be one of: f1, f2, precision, recall, cost_savings"
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        logger.setLevel(getattr(logging, self.log_level.upper(), logging.INFO))

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "PipelineConfig":
        """Load config from a JSON file; missing keys use class defaults."""
        with open(path) as fh:
            data = json.load(fh)
        return cls(**{k: v for k, v in data.items() if not k.startswith("_")})

    def to_json(self, path: Union[str, Path]) -> None:
        """Persist config (excluding private fields) to a JSON file."""
        d = {k: str(v) if isinstance(v, Path) else v
             for k, v in asdict(self).items() if not k.startswith("_")}
        with open(path, "w") as fh:
            json.dump(d, fh, indent=2)

    def summary(self) -> str:
        """Return a human-readable configuration summary string."""
        lines = ["PipelineConfig", "=" * 50]
        for k, v in asdict(self).items():
            if not k.startswith("_"):
                lines.append(f"  {k:<30s} {v}")
        return "
".join(lines)



# ---------------------------------------------------------------------------
# DataGenerator
# ---------------------------------------------------------------------------

class DataGenerator:
    """
    Synthetic payment-transaction stream generator.

    Produces a realistic labelled dataset mirroring UK open-banking data:

    * Temporal patterns (weekday bias, hour-of-day fraud uplift)
    * Merchant-category fraud priors (CATEGORY_FRAUD_PRIOR)
    * Account-level behavioural heterogeneity (limits, home country)
    * Device and geolocation signals
    * Card-testing burst patterns (20-50 micro-transactions in 5 min)
    * Synthetic identity / first-party fraud (inflated amounts)

    Parameters
    ----------
    config : PipelineConfig

    Attributes
    ----------
    accounts : pd.DataFrame
        Account-level reference table built during :meth:`generate`.
    merchants : pd.DataFrame
        Merchant reference table.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg      = config
        self.rng      = np.random.default_rng(config.random_seed)
        self._py_rng  = random.Random(config.random_seed)
        self.accounts:  Optional[pd.DataFrame] = None
        self.merchants: Optional[pd.DataFrame] = None
        logger.info("DataGenerator initialised (seed=%d)", config.random_seed)

    # -- reference data ------------------------------------------------------

    def _build_accounts(self, n_accounts: int = 50_000) -> pd.DataFrame:
        """
        Create synthetic account reference table.

        Each account has stable attributes that persist across all transactions:
        country of residence, risk tier, credit limit, and typical spending
        velocity. About 2% of accounts are synthetic identities.

        Parameters
        ----------
        n_accounts : int
            Number of unique accounts to generate. Default 50_000.

        Returns
        -------
        pd.DataFrame
            Columns: account_id, country, risk_tier, credit_limit,
            typical_txn_amount, monthly_velocity, is_synthetic_identity,
            age_days.
        """
        n = n_accounts
        countries = (
            self._py_rng.choices(COUNTRIES_LOW_RISK,  k=int(n * 0.85)) +
            self._py_rng.choices(COUNTRIES_HIGH_RISK, k=int(n * 0.15))
        )
        self._py_rng.shuffle(countries)
        risk_tiers = self._py_rng.choices(
            ["standard", "enhanced", "monitored", "restricted"],
            weights=[0.70, 0.18, 0.09, 0.03], k=n,
        )
        credit_limits   = self.rng.lognormal(7.5, 0.8, n).clip(500, 50_000)
        typical_amounts = self.rng.lognormal(3.2, 0.7, n).clip(5, 2_000)
        monthly_vel     = self.rng.integers(2, 120, size=n)
        is_synth        = self.rng.random(n) < 0.02
        age_days        = self.rng.integers(1, 3650, size=n)
        return pd.DataFrame({
            "account_id":           [f"ACC{i:07d}" for i in range(n)],
            "country":              countries[:n],
            "risk_tier":            risk_tiers,
            "credit_limit":         credit_limits.round(2),
            "typical_txn_amount":   typical_amounts.round(2),
            "monthly_velocity":     monthly_vel,
            "is_synthetic_identity": is_synth.astype(int),
            "age_days":             age_days,
        })

    def _build_merchants(self, n_merchants: int = 5_000) -> pd.DataFrame:
        """
        Create synthetic merchant reference table.

        A small fraction (~1%) of merchants are flagged as compromised
        (card-skimming suspected). Risk scores follow a Beta(1.5, 8) prior,
        giving most merchants low scores with a long right tail.

        Parameters
        ----------
        n_merchants : int
            Number of unique merchants.

        Returns
        -------
        pd.DataFrame
            Columns: merchant_id, category, country, mcc_code,
            is_compromised, days_since_compromise, risk_score.
        """
        n          = n_merchants
        categories = self._py_rng.choices(CATEGORIES, k=n)
        countries  = (
            self._py_rng.choices(COUNTRIES_LOW_RISK,  k=int(n * 0.80)) +
            self._py_rng.choices(COUNTRIES_HIGH_RISK, k=int(n * 0.20))
        )
        self._py_rng.shuffle(countries)
        mcc_codes   = [f"{self._py_rng.randint(1000, 9999):04d}" for _ in range(n)]
        is_comp     = self.rng.random(n) < 0.01
        days_since  = np.where(is_comp, self.rng.integers(1, 180, n), -1)
        risk_scores = self.rng.beta(1.5, 8.0, n).round(4)
        return pd.DataFrame({
            "merchant_id":           [f"MER{i:06d}" for i in range(n)],
            "category":              categories,
            "country":               countries[:n],
            "mcc_code":              mcc_codes,
            "is_compromised":        is_comp.astype(int),
            "days_since_compromise": days_since,
            "risk_score":            risk_scores,
        })

    # -- fraud label generation ----------------------------------------------

    def _compute_fraud_probability(
        self,
        row: Dict[str, Any],
        account: pd.Series,
        merchant: pd.Series,
    ) -> float:
        """
        Per-transaction fraud probability using a multiplicative risk model.

        Base rate = CATEGORY_FRAUD_PRIOR[category], uplifted by:
        * Hour-of-day multiplier (HOUR_FRAUD_MULTIPLIER)
        * Cross-border flag (3x uplift)
        * Merchant compromise flag (8x uplift)
        * Account risk tier (restricted: 4x, monitored: 2x)
        * Synthetic identity flag (6x uplift)
        * Amount vs credit-limit ratio (sigmoid-squashed)
        * New account (<30 days) flag (2x uplift)
        * High-risk country flag (1.8x uplift)

        The product is clamped to [0.001, 0.95].

        Parameters
        ----------
        row : dict
            Transaction fields: category, amount, hour, is_cross_border.
        account : pd.Series
            Account reference row.
        merchant : pd.Series
            Merchant reference row.

        Returns
        -------
        float
            Per-transaction fraud probability estimate.
        """
        p = CATEGORY_FRAUD_PRIOR.get(row["category"], 0.015)
        p *= HOUR_FRAUD_MULTIPLIER[int(row["hour"])]
        if row.get("is_cross_border", 0):
            p *= 3.0
        if merchant["is_compromised"]:
            p *= 8.0
        tier_mult = {"standard": 1.0, "enhanced": 1.5, "monitored": 2.0, "restricted": 4.0}
        p *= tier_mult.get(account["risk_tier"], 1.0)
        if account["is_synthetic_identity"]:
            p *= 6.0
        ratio = row["amount"] / max(account["credit_limit"], 1.0)
        p *= 1.0 + 3.0 * (1 / (1 + math.exp(-10 * (ratio - 0.5))))
        if account["age_days"] < 30:
            p *= 2.0
        if account["country"] in COUNTRIES_HIGH_RISK:
            p *= 1.8
        return float(np.clip(p, 0.001, 0.95))

    # -- transaction stream --------------------------------------------------

    def generate(self) -> pd.DataFrame:
        """
        Generate the full synthetic transaction dataset.

        Steps:
        1. Build account and merchant reference tables.
        2. Sample N transactions with account/merchant pairings.
        3. Assign timestamps across the configured date range.
        4. Compute per-transaction fraud probabilities.
        5. Sample fraud labels (Bernoulli draw).
        6. Inject card-testing bursts for a subset of fraud accounts.
        7. Return a fully labelled, time-sorted DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: transaction_id, account_id, merchant_id, amount,
            currency, category, device_type, hour, day_of_week, month,
            timestamp, is_cross_border, country_mismatch, is_fraud,
            fraud_type.
        """
        logger.info(
            "Generating %d transactions (fraud_rate=%.3f)...",
            self.cfg.n_transactions, self.cfg.fraud_rate,
        )
        n = self.cfg.n_transactions
        self.accounts  = self._build_accounts()
        self.merchants = self._build_merchants()
        acc_idx = self.rng.integers(0, len(self.accounts),  n)
        mer_idx = self.rng.integers(0, len(self.merchants), n)
        acc_rows = self.accounts.iloc[acc_idx].reset_index(drop=True)
        mer_rows = self.merchants.iloc[mer_idx].reset_index(drop=True)

        start_ts  = pd.Timestamp(self.cfg.date_start)
        end_ts    = pd.Timestamp(self.cfg.date_end)
        span_s    = int((end_ts - start_ts).total_seconds())
        offsets   = self.rng.integers(0, span_s, n)
        timestamps = [start_ts + timedelta(seconds=int(o)) for o in offsets]
        ts_series  = pd.Series(timestamps)
        hours  = pd.Series([t.hour      for t in timestamps])
        dow    = pd.Series([t.weekday() for t in timestamps])
        months = pd.Series([t.month     for t in timestamps])

        base_amounts = self.rng.lognormal(3.8, 1.1, n)
        amounts = (
            base_amounts * acc_rows["typical_txn_amount"].values
            / acc_rows["typical_txn_amount"].mean()
        ).clip(0.5, 50_000).round(2)

        currencies = self._py_rng.choices(CURRENCIES, k=n)
        devices    = self._py_rng.choices(
            DEVICE_TYPES, weights=[0.45, 0.30, 0.18, 0.04, 0.03], k=n
        )
        is_cross   = (acc_rows["country"].values != mer_rows["country"].values).astype(int)
        categories = mer_rows["category"].values

        fraud_probs = np.array([
            self._compute_fraud_probability(
                {"category": categories[i], "amount": amounts[i],
                 "hour": hours.iloc[i], "is_cross_border": bool(is_cross[i])},
                acc_rows.iloc[i], mer_rows.iloc[i],
            )
            for i in range(n)
        ])
        scale = self.cfg.fraud_rate / fraud_probs.mean()
        fraud_probs_cal = np.clip(fraud_probs * scale, 0.001, 0.95)
        is_fraud = (self.rng.random(n) < fraud_probs_cal).astype(int)

        fraud_type_raw = self._py_rng.choices(
            ["card_not_present", "account_takeover", "synthetic_identity",
             "card_testing", "first_party", "social_engineering"],
            weights=[0.38, 0.24, 0.12, 0.10, 0.09, 0.07], k=n,
        )
        fraud_types = [ft if is_fraud[i] else "none" for i, ft in enumerate(fraud_type_raw)]

        df = pd.DataFrame({
            "transaction_id":  [f"TXN{i:09d}" for i in range(n)],
            "account_id":      acc_rows["account_id"].values,
            "merchant_id":     mer_rows["merchant_id"].values,
            "amount":          amounts,
            "currency":        currencies,
            "category":        categories,
            "device_type":     devices,
            "hour":            hours.values,
            "day_of_week":     dow.values,
            "month":           months.values,
            "timestamp":       ts_series.values,
            "is_cross_border": is_cross,
            "country_mismatch": is_cross,
            "is_fraud":        is_fraud,
            "fraud_type":      fraud_types,
        })
        df = self._inject_card_testing_bursts(df)
        logger.info(
            "Generated %d transactions; fraud count=%d (%.2f%%)",
            len(df), df["is_fraud"].sum(), 100 * df["is_fraud"].mean(),
        )
        return df.sort_values("timestamp").reset_index(drop=True)

    def _inject_card_testing_bursts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inject card-testing burst patterns for a fraction of fraud accounts.

        A burst is 20-50 rapid micro-transactions (GBP 0.01-5.00) from the
        same account within a 5-minute window, followed by a large charge.
        This mirrors real-world card-validation tactics used by fraudsters.

        Parameters
        ----------
        df : pd.DataFrame
            Transaction dataset before burst injection.

        Returns
        -------
        pd.DataFrame
            Dataset augmented with burst records, labelled is_fraud=1,
            fraud_type='card_testing'.
        """
        fraud_accounts = df.loc[df["is_fraud"] == 1, "account_id"].unique()
        burst_accounts = self._py_rng.sample(
            list(fraud_accounts), k=min(500, len(fraud_accounts))
        )
        burst_rows = []
        base_ts = pd.Timestamp("2024-06-01")
        for acc in burst_accounts:
            n_micro = self._py_rng.randint(20, 50)
            ref_txn = df.loc[df["account_id"] == acc].iloc[0]
            for j in range(n_micro):
                t = base_ts + timedelta(seconds=self._py_rng.randint(0, 300))
                burst_rows.append({
                    "transaction_id":  f"BURST{acc}{j:03d}",
                    "account_id":      acc,
                    "merchant_id":     ref_txn["merchant_id"],
                    "amount":          round(self._py_rng.uniform(0.01, 5.00), 2),
                    "currency":        "GBP",
                    "category":        ref_txn["category"],
                    "device_type":     "api_key",
                    "hour":            t.hour,
                    "day_of_week":     t.weekday(),
                    "month":           t.month,
                    "timestamp":       t,
                    "is_cross_border": 0,
                    "country_mismatch": 0,
                    "is_fraud":        1,
                    "fraud_type":      "card_testing",
                })
        if burst_rows:
            df = pd.concat([df, pd.DataFrame(burst_rows)], ignore_index=True)
        return df

    def stream(self) -> Generator[Dict[str, Any], None, None]:
        """
        Yield individual transaction dicts in real-time cadence.

        Sleeps between yields to honour config.streaming_tps. Suitable for
        feeding a streaming fraud-scoring loop.

        Yields
        ------
        dict
            Single transaction record with all raw fields.

        Example
        -------
        >>> gen = DataGenerator(cfg)
        >>> for txn in gen.stream():
        ...     score = model.predict_proba([txn])[0, 1]
        """
        delay    = 1.0 / max(self.cfg.streaming_tps, 1)
        end_time = time.monotonic() + self.cfg.streaming_duration_s
        counter  = itertools.count()
        while time.monotonic() < end_time:
            i    = next(counter)
            acc  = self.accounts.sample(1, random_state=i).iloc[0]
            mer  = self.merchants.sample(1, random_state=i + 1).iloc[0]
            now  = datetime.utcnow()
            amt  = float(self.rng.lognormal(3.8, 1.1))
            txn: Dict[str, Any] = {
                "transaction_id":  f"STREAM{i:010d}",
                "account_id":      acc["account_id"],
                "merchant_id":     mer["merchant_id"],
                "amount":          round(amt, 2),
                "currency":        "GBP",
                "category":        mer["category"],
                "device_type":     self._py_rng.choice(DEVICE_TYPES),
                "hour":            now.hour,
                "day_of_week":     now.weekday(),
                "month":           now.month,
                "timestamp":       now.isoformat(),
                "is_cross_border": int(acc["country"] != mer["country"]),
                "country_mismatch": int(acc["country"] != mer["country"]),
                "is_fraud":        None,
                "fraud_type":      None,
            }
            yield txn
            time.sleep(delay)



# ---------------------------------------------------------------------------
# DataQualityChecker
# ---------------------------------------------------------------------------

class DataQualityError(RuntimeError):
    """Raised by DataQualityChecker when strict mode is enabled."""


class DataQualityChecker:
    """
    Schema validation, null/outlier detection, and drift monitoring.

    Runs a suite of checks on a raw transaction DataFrame and produces a
    structured quality report. Any check that fails raises DataQualityError
    if strict=True, otherwise logs a warning.

    Parameters
    ----------
    config : PipelineConfig
    strict : bool
        If True, any failed check raises DataQualityError. Default False.

    Attributes
    ----------
    report : dict
        Populated by :meth:`run`; contains per-check results and a summary.
    """

    REQUIRED_COLS: List[str] = [
        "transaction_id", "account_id", "merchant_id", "amount",
        "currency", "category", "device_type", "hour", "day_of_week",
        "month", "timestamp", "is_cross_border", "is_fraud",
    ]
    NUMERIC_COLS: List[str]      = ["amount", "hour", "day_of_week", "month"]
    CATEGORICAL_COLS: List[str]  = ["currency", "category", "device_type"]

    def __init__(self, config: PipelineConfig, strict: bool = False) -> None:
        self.cfg    = config
        self.strict = strict
        self.report: Dict[str, Any] = {}

    def _check_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Verify all required columns are present."""
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        extra   = [c for c in df.columns if c not in self.REQUIRED_COLS]
        return {"missing_columns": missing, "extra_columns": extra, "passed": len(missing) == 0}

    def _check_nulls(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute null rates per column; flag columns above 1% threshold."""
        null_rates = (df.isnull().mean() * 100).round(3)
        flagged    = null_rates[null_rates > 1.0].to_dict()
        return {"null_rates": null_rates.to_dict(), "flagged_columns": flagged, "passed": len(flagged) == 0}

    def _check_amount_range(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check that amounts are positive and within plausible bounds."""
        neg_count  = int((df["amount"] <= 0).sum())
        high_count = int((df["amount"] > 100_000).sum())
        return {
            "negative_amount_count": neg_count,
            "implausibly_high_count": high_count,
            "min_amount":  float(df["amount"].min()),
            "max_amount":  float(df["amount"].max()),
            "mean_amount": float(df["amount"].mean()),
            "passed":      neg_count == 0,
        }

    def _check_fraud_rate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Verify observed fraud rate is within +/-50% of configured rate."""
        observed = df["is_fraud"].mean()
        expected = self.cfg.fraud_rate
        ratio    = observed / expected if expected > 0 else float("inf")
        return {
            "observed_fraud_rate": round(float(observed), 5),
            "expected_fraud_rate": expected,
            "ratio":   round(ratio, 3),
            "passed":  0.5 <= ratio <= 2.0,
        }

    def _check_temporal_coverage(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Ensure multiple months in the configured date range are represented."""
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            ts = pd.to_datetime(df["timestamp"])
        else:
            ts = df["timestamp"]
        months_present = sorted(ts.dt.month.unique().tolist())
        return {
            "months_present":  months_present,
            "month_count":     len(months_present),
            "passed":          len(months_present) >= 3,
        }

    def _check_category_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Flag any category with <0.1% or >40% share."""
        dist      = df["category"].value_counts(normalize=True)
        sparse    = dist[dist < 0.001].index.tolist()
        dominated = dist[dist > 0.40].index.tolist()
        return {"sparse_categories": sparse, "dominated_categories": dominated,
                "passed": len(sparse) == 0 and len(dominated) == 0}

    def _check_duplicate_ids(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect duplicate transaction IDs."""
        dup_count = int(df["transaction_id"].duplicated().sum())
        return {"duplicate_count": dup_count, "passed": dup_count == 0}

    def _check_device_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Alert if any single device_type exceeds 70% of transactions."""
        dist = df["device_type"].value_counts(normalize=True)
        dominant = dist[dist > 0.70].index.tolist()
        return {"device_distribution": dist.to_dict(), "dominant_device": dominant,
                "passed": len(dominant) == 0}

    def _check_cross_border_rate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Sanity-check that cross-border rate is between 1% and 40%."""
        rate = float(df["is_cross_border"].mean())
        return {"cross_border_rate": round(rate, 4), "passed": 0.01 <= rate <= 0.40}

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute all quality checks on df and populate :attr:`report`.

        Parameters
        ----------
        df : pd.DataFrame
            Raw transaction data.

        Returns
        -------
        dict
            Nested report with keys for each check and a summary entry.

        Raises
        ------
        DataQualityError
            If strict=True and any check fails.
        """
        logger.info("Running data quality checks on %d rows...", len(df))
        checks = {
            "schema":               self._check_schema(df),
            "nulls":                self._check_nulls(df),
            "amount_range":         self._check_amount_range(df),
            "fraud_rate":           self._check_fraud_rate(df),
            "temporal_coverage":    self._check_temporal_coverage(df),
            "category_distribution": self._check_category_distribution(df),
            "duplicate_ids":        self._check_duplicate_ids(df),
            "device_distribution":  self._check_device_distribution(df),
            "cross_border_rate":    self._check_cross_border_rate(df),
        }
        n_passed = sum(1 for c in checks.values() if c["passed"])
        n_total  = len(checks)
        self.report = {
            **checks,
            "summary": {
                "n_rows":     len(df),
                "n_checks":   n_total,
                "n_passed":   n_passed,
                "n_failed":   n_total - n_passed,
                "overall_ok": n_passed == n_total,
            },
        }
        logger.info("Quality checks: %d/%d passed", n_passed, n_total)
        if self.strict and n_passed < n_total:
            failed = [k for k, v in checks.items() if not v["passed"]]
            raise DataQualityError(f"Quality checks failed: {failed}")
        return self.report


# ---------------------------------------------------------------------------
# FeatureEngineer
# ---------------------------------------------------------------------------

class FeatureEngineer:
    """
    Transforms raw transaction records into a rich feature matrix.

    Feature groups:

    Temporal
        hour_sin, hour_cos (cyclical), is_weekend, is_night,
        is_morning_rush, is_evening_rush, month_sin, month_cos.

    Amount / Behavioural
        log_amount, sqrt_amount, amount_z_by_category (z-score),
        amount_q_by_category (decile rank).

    Velocity (rolling window proxies)
        tx_count_last5, tx_sum_last5, tx_count_last20,
        amount_lag_{1,5,10}, amount_velocity_ratio.

    Categorical
        {category,device_type,currency}_{encoded,freq}

    Interaction
        high_amount_night, cross_border_night,
        weekend_high_amount, api_high_amount.

    Parameters
    ----------
    config : PipelineConfig
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg  = config
        self._label_encoders: Dict[str, LabelEncoder]       = {}
        self._freq_maps:      Dict[str, Dict[str, float]]   = {}
        self._fitted = False

    @staticmethod
    def _cyclical_encode(series: pd.Series, period: float) -> Tuple[pd.Series, pd.Series]:
        """
        Encode a periodic variable as (sin, cos) pair to preserve circularity.

        Parameters
        ----------
        series : pd.Series
            Integer or float series (e.g. hour 0-23).
        period : float
            Full cycle length (24 for hours, 12 for months).

        Returns
        -------
        Tuple[pd.Series, pd.Series]
            (sin_component, cos_component)
        """
        angle = 2 * math.pi * series / period
        return np.sin(angle), np.cos(angle)

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["hour_sin"],  df["hour_cos"]  = self._cyclical_encode(df["hour"],  24)
        df["month_sin"], df["month_cos"] = self._cyclical_encode(df["month"], 12)
        df["is_weekend"]      = (df["day_of_week"] >= 5).astype(int)
        df["is_night"]        = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
        df["is_morning_rush"] = ((df["hour"] >= 7) & (df["hour"] <= 9)).astype(int)
        df["is_evening_rush"] = ((df["hour"] >= 17) & (df["hour"] <= 19)).astype(int)
        return df

    def _add_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["log_amount"]  = np.log1p(df["amount"])
        df["sqrt_amount"] = np.sqrt(df["amount"].clip(0))
        cat_medians = df.groupby("category")["amount"].transform("median")
        cat_stds    = df.groupby("category")["amount"].transform("std").clip(lower=1.0)
        df["amount_z_by_category"] = (df["amount"] - cat_medians) / cat_stds
        df["amount_q_by_category"] = (
            df.groupby("category")["amount"]
            .transform(lambda x: pd.qcut(x.rank(method="first"), q=10, labels=False, duplicates="drop"))
            .fillna(4)
        )
        return df

    def _add_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling transaction counts and sums per account.

        Uses a sort-then-groupby approach (cumulative count/sum) rather
        than true rolling windows, for performance on 500K+ transactions.

        Parameters
        ----------
        df : pd.DataFrame
            Must have: account_id, timestamp, amount.

        Returns
        -------
        pd.DataFrame
            Input df augmented with velocity feature columns.
        """
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["account_id", "timestamp"]).reset_index(drop=True)
        df["tx_count_cumul"] = df.groupby("account_id").cumcount() + 1
        df["tx_sum_cumul"]   = df.groupby("account_id")["amount"].cumsum()
        df["tx_avg_cumul"]   = df["tx_sum_cumul"] / df["tx_count_cumul"]
        df["amount_velocity_ratio"] = df["amount"] / df["tx_avg_cumul"].clip(lower=0.01)
        for lag in [1, 5, 10]:
            df[f"amount_lag_{lag}"] = (
                df.groupby("account_id")["amount"].shift(lag).fillna(0)
            )
        df["tx_count_last5"]  = df.groupby("account_id")["amount"].transform(
            lambda x: x.rolling(5,  min_periods=1).count()
        )
        df["tx_sum_last5"]    = df.groupby("account_id")["amount"].transform(
            lambda x: x.rolling(5,  min_periods=1).sum()
        )
        df["tx_count_last20"] = df.groupby("account_id")["amount"].transform(
            lambda x: x.rolling(20, min_periods=1).count()
        )
        return df

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Label-encode and frequency-encode string categorical columns."""
        for col in ["category", "device_type", "currency"]:
            le_col   = f"{col}_encoded"
            freq_col = f"{col}_freq"
            if fit:
                le = LabelEncoder()
                df[le_col] = le.fit_transform(df[col].astype(str))
                self._label_encoders[col] = le
                self._freq_maps[col] = df[col].value_counts(normalize=True).to_dict()
            else:
                le = self._label_encoders.get(col)
                if le is not None:
                    df[le_col] = df[col].map(
                        lambda x, _le=le: _le.transform([x])[0]
                        if x in set(_le.classes_) else -1
                    )
                else:
                    df[le_col] = -1
            df[freq_col] = df[col].map(self._freq_maps.get(col, {})).fillna(0.0)
        return df

    @staticmethod
    def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create boolean interaction features capturing compound fraud signals."""
        df["high_amount_night"]   = ((df["amount"] > 500) & (df["is_night"] == 1)).astype(int)
        df["cross_border_night"]  = ((df["is_cross_border"] == 1) & (df["is_night"] == 1)).astype(int)
        df["weekend_high_amount"] = ((df["is_weekend"] == 1) & (df["amount"] > 300)).astype(int)
        df["api_high_amount"]     = ((df["device_type"] == "api_key") & (df["amount"] > 200)).astype(int)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit all encoders on training data and return transformed DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Training data.

        Returns
        -------
        pd.DataFrame
            Feature-enriched DataFrame ready for model training.
        """
        logger.info("FeatureEngineer.fit_transform on %d rows", len(df))
        df = df.copy()
        df = self._add_temporal_features(df)
        df = self._add_amount_features(df)
        df = self._add_velocity_features(df)
        df = self._encode_categoricals(df, fit=True)
        df = self._add_interaction_features(df)
        self._fitted = True
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted transformations to new data without refitting.

        Parameters
        ----------
        df : pd.DataFrame
            New transaction data with the same raw schema.

        Returns
        -------
        pd.DataFrame
            Feature-enriched DataFrame.

        Raises
        ------
        RuntimeError
            If fit_transform() has not been called first.
        """
        if not self._fitted:
            raise RuntimeError("Call fit_transform() before transform().")
        df = df.copy()
        df = self._add_temporal_features(df)
        df = self._add_amount_features(df)
        df = self._add_velocity_features(df)
        df = self._encode_categoricals(df, fit=False)
        df = self._add_interaction_features(df)
        return df

    @property
    def feature_columns(self) -> List[str]:
        """Ordered list of engineered feature column names (excludes targets)."""
        return [
            "hour_sin", "hour_cos", "month_sin", "month_cos",
            "is_weekend", "is_night", "is_morning_rush", "is_evening_rush",
            "log_amount", "sqrt_amount", "amount_z_by_category",
            "amount_q_by_category", "amount_velocity_ratio",
            "amount_lag_1", "amount_lag_5", "amount_lag_10",
            "tx_count_last5", "tx_sum_last5", "tx_count_last20",
            "category_encoded", "category_freq",
            "device_type_encoded", "device_freq",
            "currency_encoded", "currency_freq",
            "is_cross_border", "high_amount_night",
            "cross_border_night", "weekend_high_amount", "api_high_amount",
        ]



# ---------------------------------------------------------------------------
# RuleEngine
# ---------------------------------------------------------------------------

class RuleEngine:
    """
    Deterministic rule-based fraud pre-screener.

    Rules are loaded from a JSON definition file or from built-in defaults.
    Each rule evaluates to True/False per transaction. The engine aggregates
    triggered rules into a score and produces a preliminary decision.

    Decision priority order:
    1. If any HARD-DECLINE rule fires  -> decision = "decline"
    2. If total rule score >= high_threshold  -> decision = "decline"
    3. If total rule score >= review_threshold -> decision = "review"
    4. Otherwise                       -> decision = "pass"

    Parameters
    ----------
    config : PipelineConfig
    rules_path : Path, optional
        JSON file containing rule definitions. Falls back to built-in rules.
    high_threshold : float
        Rule score above which a transaction is declined. Default 0.80.
    review_threshold : float
        Rule score above which a transaction is reviewed. Default 0.40.
    """

    BUILTIN_RULES: List[Dict[str, Any]] = [
        {"id": "R001", "name": "high_velocity_1min",
         "description": "More than 3 transactions from same account in 1 minute.",
         "type": "velocity", "weight": 0.35, "hard_decline": False},
        {"id": "R002", "name": "micro_transaction_burst",
         "description": "Avg amount < GBP 2 and count > 10 in 5 minutes.",
         "type": "velocity", "weight": 0.45, "hard_decline": False},
        {"id": "R003", "name": "amount_exceeds_credit_limit",
         "description": "Amount exceeds 90% of credit limit.",
         "type": "threshold", "weight": 0.60, "hard_decline": False},
        {"id": "R004", "name": "cross_border_night_electronics",
         "description": "Cross-border electronics purchase between 23:00-05:00.",
         "type": "compound", "weight": 0.55, "hard_decline": False},
        {"id": "R005", "name": "sanctioned_country_block",
         "description": "Country on HM Treasury sanctions list.",
         "type": "blocklist", "weight": 1.00, "hard_decline": True},
        {"id": "R006", "name": "known_fraud_merchant",
         "description": "Merchant flagged as compromised within 30 days.",
         "type": "blocklist", "weight": 0.90, "hard_decline": True},
        {"id": "R007", "name": "new_account_large_txn",
         "description": "Account age < 7 days and amount > GBP 500.",
         "type": "threshold", "weight": 0.65, "hard_decline": False},
        {"id": "R008", "name": "api_key_round_amount",
         "description": "API-key device with round-number amounts.",
         "type": "pattern", "weight": 0.40, "hard_decline": False},
        {"id": "R009", "name": "foreign_currency_night",
         "description": "Non-GBP/USD/EUR transaction between 01:00-04:00.",
         "type": "compound", "weight": 0.30, "hard_decline": False},
        {"id": "R010", "name": "restricted_account",
         "description": "Account risk tier is 'restricted'.",
         "type": "threshold", "weight": 0.80, "hard_decline": False},
        {"id": "R011", "name": "gaming_high_amount_night",
         "description": "Gaming transaction > GBP 200 between 00:00-05:00.",
         "type": "compound", "weight": 0.45, "hard_decline": False},
        {"id": "R012", "name": "multiple_currencies_24h",
         "description": "More than 3 distinct currencies in 24 hours.",
         "type": "velocity", "weight": 0.50, "hard_decline": False},
    ]

    def __init__(
        self,
        config: PipelineConfig,
        rules_path: Optional[Path] = None,
        high_threshold: float = 0.80,
        review_threshold: float = 0.40,
    ) -> None:
        self.cfg              = config
        self.high_threshold   = high_threshold
        self.review_threshold = review_threshold
        self._rules           = self._load_rules(rules_path)
        logger.info("RuleEngine initialised with %d rules", len(self._rules))

    def _load_rules(self, path: Optional[Path]) -> List[Dict[str, Any]]:
        if path is not None and Path(path).exists():
            with open(path) as fh:
                return json.load(fh)
        return self.BUILTIN_RULES

    def _evaluate_rule(self, rule: Dict[str, Any], row: pd.Series) -> bool:
        """
        Evaluate a single rule against one transaction row.

        Parameters
        ----------
        rule : dict
            Rule definition with keys: id, type, weight, hard_decline.
        row : pd.Series
            Transaction row (may include feature-engineered columns).

        Returns
        -------
        bool
            True if the rule fires (suspicious pattern detected).
        """
        rid = rule["id"]
        if rid == "R001":
            return row.get("tx_count_last5", 0) > VELOCITY_THRESHOLDS["1min"]
        elif rid == "R002":
            return row.get("tx_count_last5", 0) > 10 and row.get("amount", 999) < 2.0
        elif rid == "R003":
            return row.get("amount", 0) > row.get("credit_limit", 1e9) * 0.90
        elif rid == "R004":
            return (row.get("is_cross_border", 0) == 1 and
                    row.get("is_night", 0) == 1 and
                    row.get("category", "") == "electronics")
        elif rid == "R005":
            return row.get("country", "") in ["KP", "IR", "SY", "CU"]
        elif rid == "R006":
            return row.get("is_compromised", 0) == 1
        elif rid == "R007":
            return row.get("age_days", 9999) < 7 and row.get("amount", 0) > 500
        elif rid == "R008":
            return (row.get("device_type", "") == "api_key" and
                    row.get("amount", 1) % 50 == 0)
        elif rid == "R009":
            return (row.get("currency", "GBP") not in {"GBP", "USD", "EUR"} and
                    1 <= row.get("hour", 12) <= 4)
        elif rid == "R010":
            return row.get("risk_tier", "standard") == "restricted"
        elif rid == "R011":
            return (row.get("category", "") == "gaming" and
                    row.get("amount", 0) > 200 and row.get("hour", 12) <= 5)
        elif rid == "R012":
            return row.get("tx_count_last20", 0) > 15
        return False

    def score_transaction(self, row: pd.Series) -> Dict[str, Any]:
        """
        Score a single transaction row against all rules.

        Parameters
        ----------
        row : pd.Series

        Returns
        -------
        dict
            rule_score (float), decision (str), triggered_rules (list),
            hard_decline (bool).
        """
        triggered = []
        total_score = 0.0
        hard_decline = False
        for rule in self._rules:
            if self._evaluate_rule(rule, row):
                triggered.append(rule["id"])
                total_score += rule["weight"]
                if rule["hard_decline"]:
                    hard_decline = True
        total_score = min(total_score, 1.0)
        if hard_decline or total_score >= self.high_threshold:
            decision = "decline"
        elif total_score >= self.review_threshold:
            decision = "review"
        else:
            decision = "pass"
        return {
            "rule_score":      round(total_score, 4),
            "decision":        decision,
            "triggered_rules": triggered,
            "hard_decline":    hard_decline,
        }

    def score_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply rule scoring to an entire DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Transaction data (may be feature-engineered).

        Returns
        -------
        pd.DataFrame
            Input df augmented with: rule_score, rule_decision,
            triggered_rules, hard_decline.
        """
        logger.info("RuleEngine.score_batch: %d transactions", len(df))
        results = df.apply(self.score_transaction, axis=1, result_type="expand")
        df = df.copy()
        df["rule_score"]      = results["rule_score"]
        df["rule_decision"]   = results["decision"]
        df["triggered_rules"] = results["triggered_rules"]
        df["hard_decline"]    = results["hard_decline"]
        n_declined = int((df["rule_decision"] == "decline").sum())
        n_review   = int((df["rule_decision"] == "review").sum())
        logger.info(
            "Rule decisions: decline=%d, review=%d, pass=%d",
            n_declined, n_review, len(df) - n_declined - n_review,
        )
        return df

    def save_rules(self, path: Union[str, Path]) -> None:
        """Persist current rule set to a JSON file."""
        with open(path, "w") as fh:
            json.dump(self._rules, fh, indent=2)

    @property
    def rule_ids(self) -> List[str]:
        """Return list of active rule IDs."""
        return [r["id"] for r in self._rules]

    def add_rule(self, rule: Dict[str, Any]) -> None:
        """
        Append a new rule definition at runtime.

        Parameters
        ----------
        rule : dict
            Must contain: id, name, description, type, weight, hard_decline.
        """
        required = {"id", "name", "description", "type", "weight", "hard_decline"}
        missing  = required - set(rule.keys())
        if missing:
            raise ValueError(f"Rule missing required keys: {missing}")
        self._rules.append(rule)
        logger.info("Added rule %s (weight=%.2f)", rule["id"], rule["weight"])

    def remove_rule(self, rule_id: str) -> bool:
        """
        Remove a rule by ID.

        Parameters
        ----------
        rule_id : str

        Returns
        -------
        bool
            True if the rule was found and removed.
        """
        before = len(self._rules)
        self._rules = [r for r in self._rules if r["id"] != rule_id]
        removed = len(self._rules) < before
        if removed:
            logger.info("Removed rule %s", rule_id)
        return removed



# ---------------------------------------------------------------------------
# ModelTrainer
# ---------------------------------------------------------------------------

class ModelTrainer:
    """
    Train, evaluate, and persist fraud-detection ML models.

    Supports three base learners plus a soft-voting ensemble:

    * rf       -- Random Forest (200 trees, class_weight balanced_subsample)
    * gbt      -- Gradient Boosting (300 estimators, learning_rate=0.05)
    * lr       -- Logistic Regression (L2, C=0.1, warm_start)
    * iso      -- Isolation Forest (contamination=fraud_rate, anomaly only)
    * ensemble -- Soft-voting VotingClassifier over rf + gbt + lr

    All classifiers are wrapped in CalibratedClassifierCV (isotonic, 3-fold)
    so predict_proba outputs well-calibrated probabilities.

    Parameters
    ----------
    config : PipelineConfig

    Attributes
    ----------
    models : dict
        Trained model objects, keyed by name.
    cv_results : dict
        Cross-validation metrics per model.
    eval_results : dict
        Hold-out test set metrics per model.
    feature_names : list
        Feature column names used during training.
    """

    RF_PARAMS: Dict[str, Any] = {
        "n_estimators": 200, "max_depth": 18, "min_samples_leaf": 4,
        "max_features": "sqrt", "class_weight": "balanced_subsample",
        "n_jobs": -1, "random_state": 42,
    }
    GBT_PARAMS: Dict[str, Any] = {
        "n_estimators": 300, "learning_rate": 0.05, "max_depth": 5,
        "min_samples_leaf": 8, "subsample": 0.80,
        "max_features": "sqrt", "random_state": 42,
    }
    LR_PARAMS: Dict[str, Any] = {
        "C": 0.10, "penalty": "l2", "solver": "lbfgs",
        "max_iter": 500, "class_weight": "balanced", "random_state": 42,
    }

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg           = config
        self.models:        Dict[str, Any]  = {}
        self.cv_results:    Dict[str, Any]  = {}
        self.eval_results:  Dict[str, Any]  = {}
        self.feature_names: List[str]       = []
        logger.info("ModelTrainer initialised (models=%s)", config.model_names)

    def _make_base_models(self) -> Dict[str, Any]:
        """Instantiate raw (uncalibrated) estimators per config.model_names."""
        base: Dict[str, Any] = {}
        need_rf  = "rf"       in self.cfg.model_names or "ensemble" in self.cfg.model_names
        need_gbt = "gbt"      in self.cfg.model_names or "ensemble" in self.cfg.model_names
        need_lr  = "lr"       in self.cfg.model_names or "ensemble" in self.cfg.model_names
        if need_rf:
            base["rf"]  = RandomForestClassifier(**{**self.RF_PARAMS,  "random_state": self.cfg.random_seed})
        if need_gbt:
            base["gbt"] = GradientBoostingClassifier(**{**self.GBT_PARAMS, "random_state": self.cfg.random_seed})
        if need_lr:
            base["lr"]  = LogisticRegression(**{**self.LR_PARAMS,  "random_state": self.cfg.random_seed})
        if "iso" in self.cfg.model_names:
            base["iso"] = IsolationForest(
                contamination=self.cfg.fraud_rate,
                random_state=self.cfg.random_seed, n_jobs=-1,
            )
        return base

    def _cross_validate(
        self, name: str, model: Any, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, float]:
        """
        Run stratified k-fold cross-validation and return aggregated metrics.

        Parameters
        ----------
        name : str
            Model identifier.
        model : estimator
            Unfitted sklearn-compatible estimator.
        X : np.ndarray
        y : np.ndarray

        Returns
        -------
        dict
            Mean and std for roc_auc, average_precision, f1.
        """
        logger.info("  CV for %s (%d-fold)...", name, self.cfg.n_cv_folds)
        skf = StratifiedKFold(n_splits=self.cfg.n_cv_folds, shuffle=True,
                              random_state=self.cfg.random_seed)
        try:
            auc = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
            ap  = cross_val_score(model, X, y, cv=skf, scoring="average_precision", n_jobs=-1)
            f1  = cross_val_score(model, X, y, cv=skf, scoring="f1", n_jobs=-1)
        except Exception as exc:
            logger.warning("CV for %s failed: %s", name, exc)
            return {}
        return {
            "roc_auc_mean":       float(auc.mean()), "roc_auc_std":        float(auc.std()),
            "avg_precision_mean": float(ap.mean()),  "avg_precision_std":  float(ap.std()),
            "f1_mean":            float(f1.mean()),  "f1_std":             float(f1.std()),
        }

    def _evaluate_on_test(
        self, name: str, model: Any, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute comprehensive metrics on the hold-out test set.

        Returns roc_auc, average_precision, f1, precision, recall,
        brier_score, log_loss, confusion_matrix, classification_report.

        Parameters
        ----------
        name : str
        model : fitted estimator
        X_test : np.ndarray
        y_test : np.ndarray

        Returns
        -------
        dict
            All test-set evaluation metrics.
        """
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        else:
            y_prob = model.predict(X_test).astype(float)
        y_pred = (y_prob >= 0.50).astype(int)
        roc_auc   = float(roc_auc_score(y_test, y_prob))
        avg_prec  = float(average_precision_score(y_test, y_prob))
        f1        = float(f1_score(y_test, y_pred, zero_division=0))
        precision = float(precision_score(y_test, y_pred, zero_division=0))
        recall    = float(recall_score(y_test, y_pred, zero_division=0))
        brier     = float(brier_score_loss(y_test, y_prob))
        logloss   = float(log_loss(y_test, y_prob))
        cm        = confusion_matrix(y_test, y_pred).tolist()
        cr        = classification_report(y_test, y_pred, output_dict=True)
        logger.info("  %s test: ROC-AUC=%.4f  AP=%.4f  F1=%.4f",
                    name, roc_auc, avg_prec, f1)
        return {
            "roc_auc":  roc_auc, "avg_precision": avg_prec, "f1":    f1,
            "precision": precision, "recall": recall, "brier_score":  brier,
            "log_loss":  logloss, "confusion_matrix": cm,
            "classification_report": cr,
        }

    def fit(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_test:  np.ndarray, y_test:  np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "ModelTrainer":
        """
        Train all configured models with cross-validation.

        Workflow:
        1. Build base estimators.
        2. Wrap rf+gbt+lr in VotingClassifier if 'ensemble' configured.
        3. For each model: run CV, fit on full train set, calibrate, evaluate.
        4. Store results in models, cv_results, eval_results.

        Parameters
        ----------
        X_train, y_train : np.ndarray
        X_test,  y_test  : np.ndarray
        feature_names : list of str, optional

        Returns
        -------
        ModelTrainer
            Self (for chaining).
        """
        self.feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]
        logger.info("ModelTrainer.fit: X_train=%s, fraud_rate=%.3f",
                    X_train.shape, y_train.mean())
        base_models = self._make_base_models()
        if "ensemble" in self.cfg.model_names and len(base_models) >= 2:
            voters = [(k, v) for k, v in base_models.items() if k != "iso"]
            base_models["ensemble"] = VotingClassifier(estimators=voters, voting="soft", n_jobs=-1)
        for name, model in base_models.items():
            logger.info("Training %s...", name)
            if name != "iso":
                self.cv_results[name] = self._cross_validate(name, model, X_train, y_train)
            model.fit(X_train, y_train)
            if name != "iso":
                cal = CalibratedClassifierCV(model, method="isotonic", cv=3)
                try:
                    cal.fit(X_train, y_train)
                    self.models[name] = cal
                except Exception:
                    self.models[name] = model
            else:
                self.models[name] = model
            self.eval_results[name] = self._evaluate_on_test(name, self.models[name], X_test, y_test)
        best = max({k: v.get("roc_auc", 0) for k, v in self.eval_results.items()}.items(),
                   key=lambda x: x[1])
        logger.info("Training complete. Best ROC-AUC: %s = %.4f", best[0], best[1])
        return self

    def predict_proba(self, X: np.ndarray, model_name: str = "ensemble") -> np.ndarray:
        """
        Return fraud probability scores from a named model.

        Parameters
        ----------
        X : np.ndarray
        model_name : str
            Defaults to 'ensemble'.

        Returns
        -------
        np.ndarray
            1-D array of fraud probabilities in [0, 1].
        """
        model = self.models.get(model_name)
        if model is None:
            raise KeyError(f"Model {model_name!r} not found. Available: {list(self.models)}")
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        scores = (model.decision_function(X) if hasattr(model, "decision_function")
                  else model.predict(X).astype(float))
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    def feature_importances(self, model_name: str = "rf") -> pd.Series:
        """
        Return native feature importances from a tree-based model.

        Parameters
        ----------
        model_name : str
            Must be 'rf' or 'gbt'.

        Returns
        -------
        pd.Series
            Feature importances sorted descending.
        """
        model = self.models.get(model_name)
        if model is None:
            raise KeyError(f"Model {model_name!r} not found.")
        base = getattr(model, "estimator", model)
        if not hasattr(base, "feature_importances_"):
            raise ValueError(f"Model {model_name!r} has no feature_importances_.")
        return pd.Series(base.feature_importances_,
                         index=self.feature_names).sort_values(ascending=False)

    def save(self, directory: Union[str, Path]) -> None:
        """Persist all trained models as pickle files."""
        import pickle
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, model in self.models.items():
            path = directory / f"{name}.pkl"
            with open(path, "wb") as fh:
                pickle.dump(model, fh)
            logger.info("Saved %s -> %s", name, path)

    @classmethod
    def load(cls, directory: Union[str, Path], config: PipelineConfig) -> "ModelTrainer":
        """
        Load previously saved models from directory.

        Parameters
        ----------
        directory : str or Path
        config : PipelineConfig

        Returns
        -------
        ModelTrainer
        """
        import pickle
        trainer = cls(config)
        for path in Path(directory).glob("*.pkl"):
            with open(path, "rb") as fh:
                trainer.models[path.stem] = pickle.load(fh)
            logger.info("Loaded %s <- %s", path.stem, path)
        return trainer



# ---------------------------------------------------------------------------
# ThresholdOptimiser
# ---------------------------------------------------------------------------

class ThresholdOptimiser:
    """
    Cost-aware decision threshold search over a probability score.

    Sweeps thresholds from 0.01 to 0.99, computing multiple objective
    metrics at each point. The optimal threshold is selected according to
    config.threshold_metric.

    Metrics computed per threshold:
        precision, recall, f1, f2 (beta=2), cost_savings (GBP/1000 txns),
        fraud_capture_rate (= recall), false_positive_rate.

    Parameters
    ----------
    config : PipelineConfig
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config
        self.sweep_df: Optional[pd.DataFrame] = None
        self.optimal_threshold: Optional[float] = None
        self.review_threshold:  Optional[float] = None

    def _cost_savings(
        self, y_true: np.ndarray, y_score: np.ndarray, threshold: float
    ) -> float:
        """
        Net cost savings per 1000 transactions at a given threshold.

        Model:
        * TP: saves false_negative_cost (chargeback averted)
        * FP: costs false_positive_cost (declined legitimate)
        * (TP+FP): costs review_cost each (analyst time)
        * FN: costs false_negative_cost (missed fraud)

        Parameters
        ----------
        y_true : np.ndarray
        y_score : np.ndarray
        threshold : float

        Returns
        -------
        float
            Net savings per 1000 transactions (GBP).
        """
        y_pred = (y_score >= threshold).astype(int)
        cm     = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        scale  = 1000.0 / max(len(y_true), 1)
        return float((
            tp * self.cfg.false_negative_cost
            - fp * self.cfg.false_positive_cost
            - (tp + fp) * self.cfg.review_cost
            - fn * self.cfg.false_negative_cost
        ) * scale)

    def sweep(
        self, y_true: np.ndarray, y_score: np.ndarray, n_steps: int = 200
    ) -> pd.DataFrame:
        """
        Sweep thresholds from 0.01 to 0.99 and collect metrics.

        Parameters
        ----------
        y_true : np.ndarray
        y_score : np.ndarray
        n_steps : int
            Number of threshold evaluation points. Default 200.

        Returns
        -------
        pd.DataFrame
            One row per threshold: threshold, precision, recall,
            f1, f2, cost_savings, fpr, fnr.
        """
        thresholds = np.linspace(0.01, 0.99, n_steps)
        rows = []
        for t in thresholds:
            y_pred = (y_score >= t).astype(int)
            prec   = float(precision_score(y_true, y_pred, zero_division=0))
            rec    = float(recall_score(y_true,    y_pred, zero_division=0))
            f1     = 2 * prec * rec / (prec + rec + 1e-9)
            f2     = 5 * prec * rec / (4 * prec + rec + 1e-9)
            cs     = self._cost_savings(y_true, y_score, t)
            fp_n   = int(((y_pred == 1) & (y_true == 0)).sum())
            fn_n   = int(((y_pred == 0) & (y_true == 1)).sum())
            fpr    = fp_n / max((y_true == 0).sum(), 1)
            fnr    = fn_n / max((y_true == 1).sum(), 1)
            rows.append({"threshold": round(float(t), 4), "precision": round(prec, 5),
                         "recall": round(rec, 5), "f1": round(f1, 5), "f2": round(f2, 5),
                         "cost_savings": round(cs, 2), "fpr": round(fpr, 5), "fnr": round(fnr, 5)})
        self.sweep_df = pd.DataFrame(rows)
        return self.sweep_df

    def optimise(
        self, y_true: np.ndarray, y_score: np.ndarray
    ) -> Tuple[float, float]:
        """
        Find optimal decision and review thresholds.

        The decision threshold maximises config.threshold_metric over the
        sweep. The review threshold is set to decision threshold minus 0.10.

        Parameters
        ----------
        y_true : np.ndarray
        y_score : np.ndarray

        Returns
        -------
        Tuple[float, float]
            (optimal_threshold, review_threshold)
        """
        if self.sweep_df is None:
            self.sweep(y_true, y_score)
        metric = self.cfg.threshold_metric
        if metric not in self.sweep_df.columns:
            metric = "f1"
        best_row               = self.sweep_df.loc[self.sweep_df[metric].idxmax()]
        self.optimal_threshold = float(best_row["threshold"])
        self.review_threshold  = max(0.01, self.optimal_threshold - 0.10)
        logger.info("Optimal threshold: %.4f (metric=%s, value=%.4f)",
                    self.optimal_threshold, metric, best_row[metric])
        return self.optimal_threshold, self.review_threshold


# ---------------------------------------------------------------------------
# ExplainabilityEngine
# ---------------------------------------------------------------------------

class ExplainabilityEngine:
    """
    Model-agnostic interpretability layer.

    Provides:
    * Permutation feature importance (mean ROC-AUC decrease, 5 repeats)
    * PDP (Partial Dependence Plot) data extraction for top-10 features
    * Instance-level explanation via occlusion/ablation

    Parameters
    ----------
    config : PipelineConfig
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config
        self._perm_importances: Optional[pd.DataFrame] = None
        self._pdp_data: Dict[str, Any] = {}

    def compute_permutation_importance(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        n_repeats: int = 5,
    ) -> pd.DataFrame:
        """
        Permutation feature importance (mean ROC-AUC drop).

        Shuffles each feature column n_repeats times and measures the
        mean decrease in ROC-AUC. Higher drop = more important feature.

        Parameters
        ----------
        model : fitted estimator with predict_proba
        X_test : np.ndarray
        y_test : np.ndarray
        feature_names : list of str
        n_repeats : int

        Returns
        -------
        pd.DataFrame
            Sorted descending; columns: feature, mean_importance, std_importance.
        """
        logger.info("Permutation importance (%d features, %d repeats)...",
                    len(feature_names), n_repeats)
        result = permutation_importance(
            model, X_test, y_test, scoring="roc_auc",
            n_repeats=n_repeats, random_state=self.cfg.random_seed, n_jobs=-1,
        )
        df = pd.DataFrame({
            "feature":         feature_names,
            "mean_importance": result.importances_mean,
            "std_importance":  result.importances_std,
        }).sort_values("mean_importance", ascending=False).reset_index(drop=True)
        self._perm_importances = df
        return df

    def explain_instance(
        self,
        model: Any,
        row: np.ndarray,
        feature_names: List[str],
        baseline_score: float = 0.023,
    ) -> Dict[str, Any]:
        """
        Approximate per-feature contributions for a single prediction.

        Uses occlusion (zero-out) ablation: sets each feature to 0 and
        measures the change in predicted fraud probability. Features are
        sorted by absolute contribution magnitude.

        Parameters
        ----------
        model : fitted estimator
        row : np.ndarray
            1-D feature array for the transaction to explain.
        feature_names : list of str
        baseline_score : float
            Reference fraud rate to contextualise the prediction.

        Returns
        -------
        dict
            feature_contributions, predicted_score, baseline_score.
        """
        if hasattr(model, "predict_proba"):
            full_score = float(model.predict_proba(row.reshape(1, -1))[0, 1])
        else:
            full_score = float(model.predict(row.reshape(1, -1))[0])
        contributions = []
        for i, feat in enumerate(feature_names):
            ablated = row.copy()
            ablated[i] = 0.0
            if hasattr(model, "predict_proba"):
                abl_score = float(model.predict_proba(ablated.reshape(1, -1))[0, 1])
            else:
                abl_score = float(model.predict(ablated.reshape(1, -1))[0])
            delta = full_score - abl_score
            contributions.append({
                "feature":   feat,
                "delta":     round(delta, 5),
                "direction": "fraud" if delta > 0 else "legitimate",
            })
        contributions.sort(key=lambda x: abs(x["delta"]), reverse=True)
        return {
            "predicted_score":       round(full_score, 5),
            "baseline_score":        baseline_score,
            "feature_contributions": contributions[:15],
        }

    def compute_pdp(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: List[str],
        top_n: int = 8,
        grid_resolution: int = 50,
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Compute marginal PDP data for the top-N most important features.

        Parameters
        ----------
        model : fitted estimator
        X : np.ndarray
            Background dataset (sampled to 1000 rows for speed).
        feature_names : list of str
        top_n : int
        grid_resolution : int

        Returns
        -------
        dict
            Keyed by feature name; each value is {grid: [...], mean_pred: [...]}.
        """
        logger.info("Computing PDPs for top %d features...", top_n)
        if self._perm_importances is not None:
            top_features = self._perm_importances["feature"].head(top_n).tolist()
        else:
            top_features = feature_names[:top_n]
        n_sample = min(1000, len(X))
        idx      = np.random.choice(len(X), n_sample, replace=False)
        X_sample = X[idx]
        pdp_data: Dict[str, Any] = {}
        for feat in top_features:
            if feat not in feature_names:
                continue
            fi        = feature_names.index(feat)
            feat_vals = X_sample[:, fi]
            grid      = np.linspace(feat_vals.min(), feat_vals.max(), grid_resolution)
            mean_preds = []
            for val in grid:
                X_mod = X_sample.copy()
                X_mod[:, fi] = val
                preds = (model.predict_proba(X_mod)[:, 1]
                         if hasattr(model, "predict_proba")
                         else model.predict(X_mod).astype(float))
                mean_preds.append(float(preds.mean()))
            pdp_data[feat] = {"grid": grid.tolist(), "mean_pred": mean_preds}
        self._pdp_data = pdp_data
        return pdp_data


# ---------------------------------------------------------------------------
# MonitoringDaemon
# ---------------------------------------------------------------------------

class MonitoringDaemon:
    """
    Post-deployment model drift and performance monitoring.

    Implements:
    * PSI (Population Stability Index) for score distribution drift.
    * CSI (Characteristic Stability Index) for feature-level drift.
    * Performance degradation alerts when AUC drops below threshold.
    * Alert emission to configurable sinks (log, file, JSON).

    PSI interpretation (Siddiqi 2006):
        PSI < 0.10:  stable
        0.10-0.20:   monitor
        >= 0.20:     retrain recommended

    Parameters
    ----------
    config : PipelineConfig
    reference_df : pd.DataFrame, optional
        Training-set distribution used as PSI baseline.

    Attributes
    ----------
    alerts : list
        Chronological list of emitted alert dicts.
    psi_results : dict
        Latest PSI values per feature.
    """

    PSI_BINS = 10

    def __init__(
        self,
        config: PipelineConfig,
        reference_df: Optional[pd.DataFrame] = None,
    ) -> None:
        self.cfg          = config
        self.reference_df = reference_df
        self.alerts:      List[Dict[str, Any]] = []
        self.psi_results: Dict[str, float]     = {}
        self._perf_history: deque = deque(maxlen=100)

    @staticmethod
    def _compute_psi(
        expected: np.ndarray, actual: np.ndarray, n_bins: int = 10
    ) -> float:
        """
        Population Stability Index = sum((act% - exp%) * ln(act%/exp%)).

        Parameters
        ----------
        expected : np.ndarray
            Reference distribution (training scores).
        actual : np.ndarray
            Current distribution (production scores).
        n_bins : int

        Returns
        -------
        float
            PSI value (>=0; larger = more drift).
        """
        eps  = 1e-6
        bins = np.unique(np.percentile(expected, np.linspace(0, 100, n_bins + 1)))
        if len(bins) < 2:
            return 0.0
        exp_c = np.histogram(expected, bins=bins)[0] + eps
        act_c = np.histogram(actual,   bins=bins)[0] + eps
        exp_p = exp_c / exp_c.sum()
        act_p = act_c / act_c.sum()
        return float(max(np.sum((act_p - exp_p) * np.log(act_p / exp_p)), 0.0))

    def check_score_drift(
        self,
        current_scores: np.ndarray,
        reference_scores: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Check fraud score distribution drift using PSI.

        Parameters
        ----------
        current_scores : np.ndarray
        reference_scores : np.ndarray, optional

        Returns
        -------
        dict
            psi, status, alert_emitted.
        """
        if reference_scores is None:
            if self.reference_df is not None and "fraud_score" in self.reference_df.columns:
                reference_scores = self.reference_df["fraud_score"].values
            else:
                return {"psi": None, "status": "unknown", "alert_emitted": False}
        psi = self._compute_psi(reference_scores, current_scores)
        self.psi_results["score"] = psi
        if psi < 0.10:
            status = "stable"
        elif psi < self.cfg.psi_threshold:
            status = "monitor"
        else:
            status = "retrain_recommended"
        alert_emitted = False
        if status in ("monitor", "retrain_recommended"):
            self._emit_alert({
                "type": "score_drift", "psi": round(psi, 5), "status": status,
                "severity": "high" if status == "retrain_recommended" else "medium",
                "timestamp": datetime.utcnow().isoformat(),
                "message": f"Score PSI={psi:.4f} ({status}).",
            })
            alert_emitted = True
        return {"psi": round(psi, 5), "status": status, "alert_emitted": alert_emitted}

    def check_feature_drift(
        self,
        current_df: pd.DataFrame,
        reference_df: Optional[pd.DataFrame] = None,
        features: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        CSI for every numeric feature in the dataset.

        Parameters
        ----------
        current_df : pd.DataFrame
        reference_df : pd.DataFrame, optional
        features : list of str, optional

        Returns
        -------
        pd.DataFrame
            feature, psi, status -- sorted by psi descending.
        """
        ref = reference_df if reference_df is not None else self.reference_df
        if ref is None:
            return pd.DataFrame(columns=["feature", "psi", "status"])
        numeric_cols = features or current_df.select_dtypes(include=[np.number]).columns.tolist()
        rows = []
        for col in numeric_cols:
            if col not in ref.columns or col not in current_df.columns:
                continue
            psi = self._compute_psi(ref[col].dropna().values, current_df[col].dropna().values)
            self.psi_results[col] = psi
            status = "stable" if psi < 0.10 else ("monitor" if psi < self.cfg.psi_threshold else "drift")
            if status == "drift":
                self._emit_alert({
                    "type": "feature_drift", "severity": "medium", "feature": col,
                    "psi": round(psi, 5), "timestamp": datetime.utcnow().isoformat(),
                    "message": f"Feature {col!r} PSI={psi:.4f} -- drift detected.",
                })
            rows.append({"feature": col, "psi": round(psi, 5), "status": status})
        return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)

    def check_performance(
        self,
        y_true: np.ndarray, y_score: np.ndarray,
        baseline_auc: float = 0.95, min_auc: float = 0.90,
    ) -> Dict[str, Any]:
        """
        Compute current AUC and flag degradation below min_auc.

        Parameters
        ----------
        y_true, y_score : np.ndarray
        baseline_auc : float
        min_auc : float

        Returns
        -------
        dict
            current_auc, delta_from_baseline, alert_emitted.
        """
        try:
            auc = float(roc_auc_score(y_true, y_score))
        except ValueError:
            return {"current_auc": None, "delta_from_baseline": None, "alert_emitted": False}
        self._perf_history.append(auc)
        delta = auc - baseline_auc
        alert_emitted = False
        if auc < min_auc:
            self._emit_alert({
                "type": "performance_degradation", "severity": "critical",
                "current_auc": round(auc, 5), "baseline_auc": baseline_auc,
                "delta": round(delta, 5), "timestamp": datetime.utcnow().isoformat(),
                "message": f"AUC dropped to {auc:.4f} (baseline={baseline_auc:.4f}).",
            })
            alert_emitted = True
        return {"current_auc": round(auc, 5), "delta_from_baseline": round(delta, 5),
                "alert_emitted": alert_emitted}

    def _emit_alert(self, alert: Dict[str, Any]) -> None:
        self.alerts.append(alert)
        logger.warning("[ALERT][%s] %s",
                       alert.get("severity", "INFO").upper(),
                       alert.get("message", ""))

    def alert_summary(self) -> Dict[str, Any]:
        """Summarise all emitted alerts by severity."""
        by_sev = Counter(a.get("severity", "unknown") for a in self.alerts)
        return {
            "total_alerts": len(self.alerts),
            "by_severity":  dict(by_sev),
            "latest":       self.alerts[-1] if self.alerts else None,
        }



# ---------------------------------------------------------------------------
# FraudDecisionEngine
# ---------------------------------------------------------------------------

class FraudDecisionEngine:
    """
    Combines rule-engine pre-screening with ML scoring into a final verdict.

    Decision logic (priority order):
    1. rule_decision == "decline"   -> final = "decline"  (rules override ML)
    2. rule_decision == "review"    -> final = "review"
    3. ml_score >= decline_threshold -> final = "decline"
    4. ml_score >= review_threshold  -> final = "review"
    5. Otherwise                     -> final = "pass"

    A composite score (70% ML + 30% rules) is also computed for ranking.
    Cumulative decision counts are tracked in :attr:`stats`.

    Parameters
    ----------
    config : PipelineConfig
    decline_threshold : float
        ML score above which a transaction is declined. Default 0.60.
    review_threshold : float
        ML score above which a transaction is reviewed. Default 0.35.
    """

    DECISION_LABELS = {"pass", "review", "decline"}

    def __init__(
        self,
        config: PipelineConfig,
        decline_threshold: float = 0.60,
        review_threshold:  float = 0.35,
    ) -> None:
        self.cfg               = config
        self.decline_threshold = decline_threshold
        self.review_threshold  = review_threshold
        self._stats: Dict[str, int] = Counter()

    def decide(
        self, ml_score: float, rule_decision: str, rule_score: float
    ) -> Dict[str, Any]:
        """
        Produce a final fraud decision for a single transaction.

        Parameters
        ----------
        ml_score : float
        rule_decision : str
        rule_score : float

        Returns
        -------
        dict
            final_decision, ml_score, rule_score, composite_score, reason.
        """
        composite = 0.70 * ml_score + 0.30 * rule_score
        if rule_decision == "decline":
            final, reason = "decline", "hard_rule"
        elif rule_decision == "review":
            final, reason = "review",  "rule_review"
        elif ml_score >= self.decline_threshold:
            final, reason = "decline", "ml_score_high"
        elif ml_score >= self.review_threshold:
            final, reason = "review",  "ml_score_medium"
        else:
            final, reason = "pass",    "clean"
        self._stats[final] += 1
        return {
            "final_decision":  final,
            "ml_score":        round(ml_score, 5),
            "rule_score":      round(rule_score, 5),
            "composite_score": round(composite, 5),
            "reason":          reason,
        }

    def decide_batch(
        self, df: pd.DataFrame, score_col: str = "ml_score"
    ) -> pd.DataFrame:
        """
        Apply decision logic to an entire scored DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain: ml_score (or score_col), rule_decision, rule_score.
        score_col : str

        Returns
        -------
        pd.DataFrame
            Input df plus: final_decision, composite_score, reason.
        """
        logger.info("FraudDecisionEngine.decide_batch: %d rows", len(df))
        if score_col not in df.columns:
            raise KeyError(f"Column {score_col!r} not in DataFrame.")
        results = df.apply(
            lambda row: self.decide(
                row[score_col],
                row.get("rule_decision", "pass"),
                row.get("rule_score", 0.0),
            ),
            axis=1, result_type="expand",
        )
        df = df.copy()
        df["final_decision"]  = results["final_decision"]
        df["composite_score"] = results["composite_score"]
        df["reason"]          = results["reason"]
        logger.info("Decisions: %s", dict(df["final_decision"].value_counts()))
        return df

    @property
    def stats(self) -> Dict[str, int]:
        """Cumulative decision counts since instantiation."""
        return dict(self._stats)


# ---------------------------------------------------------------------------
# PipelineOrchestrator
# ---------------------------------------------------------------------------

class PipelineOrchestrator:
    """
    Top-level orchestrator chaining all pipeline stages in order.

    Batch pipeline stages:
    1.  DataGenerator      -> raw_df
    2.  DataQualityChecker -> quality_report
    3.  FeatureEngineer    -> feature_df
    4.  Train/test split
    5.  RuleEngine         -> rule-scored test_df
    6.  ModelTrainer       -> trained models, eval_results
    7.  ThresholdOptimiser -> optimal thresholds
    8.  FraudDecisionEngine-> final decisions on test set
    9.  ExplainabilityEngine -> permutation importances, PDPs
    10. MonitoringDaemon   -> PSI/CSI drift checks
    11. Persist artefacts  -> config.out_dir

    Parameters
    ----------
    config : PipelineConfig

    Attributes
    ----------
    raw_df : pd.DataFrame
    feature_df : pd.DataFrame
    train_df, test_df : pd.DataFrame
    results : dict
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg          = config
        self.raw_df:     Optional[pd.DataFrame] = None
        self.feature_df: Optional[pd.DataFrame] = None
        self.train_df:   Optional[pd.DataFrame] = None
        self.test_df:    Optional[pd.DataFrame] = None
        self.results:    Dict[str, Any]          = {}
        self._generator = DataGenerator(config)
        self._quality   = DataQualityChecker(config)
        self._features  = FeatureEngineer(config)
        self._rules     = RuleEngine(config)
        self._trainer   = ModelTrainer(config)
        self._threshold = ThresholdOptimiser(config)
        self._decision  = FraudDecisionEngine(config)
        self._explain   = ExplainabilityEngine(config)
        self._monitor   = MonitoringDaemon(config)
        logger.info("PipelineOrchestrator ready (n_transactions=%d)",
                    config.n_transactions)

    def run_batch(self) -> Dict[str, Any]:
        """
        Execute the full batch pipeline end-to-end.

        Returns
        -------
        dict
            quality_report, cv_results, eval_results, thresholds,
            decision_stats, feature_importances, drift_check,
            monitoring_summary, elapsed_s, primary_model.
        """
        t0 = time.monotonic()
        logger.info("=" * 60 + "
Pipeline batch run started
" + "=" * 60)

        # Stage 1 -- Generate
        logger.info("[1/10] Generating transactions...")
        self.raw_df = self._generator.generate()

        # Stage 2 -- Quality checks
        logger.info("[2/10] Data quality checks...")
        quality_report = self._quality.run(self.raw_df)

        # Stage 3 -- Feature engineering
        logger.info("[3/10] Feature engineering...")
        self.feature_df = self._features.fit_transform(self.raw_df)

        # Stage 4 -- Split
        logger.info("[4/10] Splitting train/test...")
        feat_cols = [c for c in self._features.feature_columns
                     if c in self.feature_df.columns]
        self.cfg._feature_cols = feat_cols
        X = self.feature_df[feat_cols].fillna(0).values
        y = self.feature_df[self.cfg._label_col].values
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=self.cfg.test_size,
            stratify=y, random_state=self.cfg.random_seed,
        )

        # Stage 5 -- Rules
        logger.info("[5/10] Rule engine scoring...")
        self.test_df = self.feature_df.iloc[len(X_tr):].copy()
        self.test_df = self._rules.score_batch(self.test_df)

        # Stage 6 -- Train
        logger.info("[6/10] Training ML models...")
        self._trainer.fit(X_tr, y_tr, X_te, y_te, feature_names=feat_cols)

        # Stage 7 -- Threshold
        logger.info("[7/10] Optimising decision thresholds...")
        primary = ("ensemble" if "ensemble" in self._trainer.models
                   else list(self._trainer.models)[0])
        y_score = self._trainer.predict_proba(X_te, model_name=primary)
        opt_t, rev_t = self._threshold.optimise(y_te, y_score)
        self._decision.decline_threshold = opt_t
        self._decision.review_threshold  = rev_t

        # Stage 8 -- Decisions
        logger.info("[8/10] Applying decision engine...")
        self.test_df = self.test_df.copy()
        self.test_df["ml_score"] = y_score
        if "rule_decision" not in self.test_df.columns:
            self.test_df["rule_decision"] = "pass"
            self.test_df["rule_score"]    = 0.0
        self.test_df = self._decision.decide_batch(self.test_df)

        # Stage 9 -- Explainability
        logger.info("[9/10] Computing explainability metrics...")
        model_obj = self._trainer.models[primary]
        perm_imp  = self._explain.compute_permutation_importance(
            model_obj, X_te, y_te, feat_cols, n_repeats=3
        )

        # Stage 10 -- Monitoring
        logger.info("[10/10] Running monitoring checks...")
        train_scores = self._trainer.predict_proba(X_tr, model_name=primary)
        drift_result = self._monitor.check_score_drift(y_score, train_scores)

        # Persist artefacts
        self._trainer.save(self.cfg.out_dir / "models")
        self.cfg.to_json(self.cfg.out_dir / "config.json")
        perm_imp.to_csv(self.cfg.out_dir / "feature_importance.csv", index=False)
        if self._threshold.sweep_df is not None:
            self._threshold.sweep_df.to_csv(
                self.cfg.out_dir / "threshold_sweep.csv", index=False
            )

        elapsed = time.monotonic() - t0
        logger.info("Pipeline complete in %.1fs", elapsed)
        self.results = {
            "quality_report":     quality_report,
            "cv_results":         self._trainer.cv_results,
            "eval_results":       self._trainer.eval_results,
            "thresholds":         {"decline": opt_t, "review": rev_t},
            "decision_stats":     self._decision.stats,
            "feature_importances": perm_imp.head(15).to_dict("records"),
            "drift_check":        drift_result,
            "monitoring_summary": self._monitor.alert_summary(),
            "elapsed_s":          round(elapsed, 2),
            "primary_model":      primary,
        }
        return self.results

    def print_summary(self) -> None:
        """Print a human-readable run summary to stdout."""
        if not self.results:
            print("No results. Call run_batch() first.")
            return
        print("
" + "=" * 60)
        print("  FRAUD PIPELINE -- RUN SUMMARY")
        print("=" * 60)
        print(f"  Elapsed:       {self.results.get('elapsed_s', '?'):.1f}s")
        print(f"  Primary model: {self.results.get('primary_model', '?')}")
        print()
        print("  Model Evaluation (test set):")
        for name, m in self.results.get("eval_results", {}).items():
            print(f"    {name:<12s}  ROC-AUC={m.get('roc_auc', 0):.4f}  "
                  f"F1={m.get('f1', 0):.4f}  AP={m.get('avg_precision', 0):.4f}")
        print()
        th = self.results.get("thresholds", {})
        print(f"  Thresholds:    decline>={th.get('decline', '?')}  review>={th.get('review', '?')}")
        print()
        print("  Decision Counts:")
        for k, v in self.results.get("decision_stats", {}).items():
            print(f"    {k:<10s} {v:>8d}")
        mon = self.results.get("monitoring_summary", {})
        print(f"
  Monitoring Alerts: {mon.get('total_alerts', 0)}")
        print("=" * 60 + "
")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description="Fintech Payments Fraud Detection Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["batch", "stream"], default="batch")
    p.add_argument("--n-transactions", type=int, default=50_000)
    p.add_argument("--fraud-rate",     type=float, default=0.023)
    p.add_argument("--random-seed",    type=int,   default=42)
    p.add_argument("--out-dir",        type=str,   default="pipeline_output")
    p.add_argument("--models", nargs="+", default=["rf", "gbt", "ensemble"],
                   choices=["rf", "gbt", "lr", "iso", "ensemble"])
    p.add_argument("--threshold-metric", default="f2",
                   choices=["f1", "f2", "precision", "recall", "cost_savings"])
    p.add_argument("--tps",      type=int, default=200, help="Streaming TPS")
    p.add_argument("--duration", type=int, default=60,  help="Streaming seconds")
    p.add_argument("--config-file", type=str, default=None)
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """
    CLI entry-point for the fraud detection pipeline.

    Parameters
    ----------
    argv : list of str, optional
        CLI arguments (defaults to sys.argv[1:]).

    Returns
    -------
    int
        0 = success, 1 = error.
    """
    parser = _build_arg_parser()
    args   = parser.parse_args(argv)
    if args.config_file:
        cfg = PipelineConfig.from_json(args.config_file)
    else:
        cfg = PipelineConfig(
            n_transactions    = args.n_transactions,
            fraud_rate        = args.fraud_rate,
            random_seed       = args.random_seed,
            out_dir           = Path(args.out_dir),
            model_names       = args.models,
            threshold_metric  = args.threshold_metric,
            streaming_tps     = args.tps,
            streaming_duration_s = args.duration,
            log_level         = args.log_level,
        )
    logger.info("
%s", cfg.summary())
    try:
        orch = PipelineOrchestrator(cfg)
        if args.mode == "batch":
            orch.run_batch()
            orch.print_summary()
        else:
            logger.info("Streaming mode -- not fully implemented in CLI demo.")
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        return 0
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
