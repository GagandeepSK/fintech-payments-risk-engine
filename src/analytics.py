"""
Advanced Analytics and Reporting Module
=========================================
Author: Gagandeep Kapoor
Date:   2026-09-02

Provides analytical post-processing on top of the pipeline outputs:

    FraudPatternAnalyser  -- statistical characterisation of fraud patterns
    SegmentAnalyser       -- population-segment breakdown of fraud rates
    TimeSeriesAnalyser    -- temporal trend decomposition and anomaly detection
    NetworkAnalyser       -- account-merchant bipartite graph analytics
    ReportBuilder         -- assembles a full HTML/JSON report from all analysers
    MetricTracker         -- rolling window metric tracking for dashboarding
    AnomalyDetector       -- statistical process control (CUSUM, EWMA, Z-score)
    CohortAnalyser        -- account cohort-level fraud risk over time

Usage
-----
::

    from src.analytics import FraudPatternAnalyser, ReportBuilder
    analyser = FraudPatternAnalyser(df, config)
    report   = analyser.run()

"""

from __future__ import annotations

import json
import logging
import math
import statistics
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIGNIFICANCE_LEVEL: float = 0.05
CUSUM_SLACK:        float = 0.5   # CUSUM allowable slack (std units)
EWMA_LAMBDA:        float = 0.20  # EWMA smoothing parameter

FRAUD_TYPE_LABELS: Dict[str, str] = {
    "card_not_present":  "Card-Not-Present",
    "account_takeover":  "Account Takeover",
    "synthetic_identity": "Synthetic Identity",
    "card_testing":      "Card Testing",
    "first_party":       "First-Party",
    "social_engineering": "Social Engineering",
    "none":              "Legitimate",
}


# ---------------------------------------------------------------------------
# FraudPatternAnalyser
# ---------------------------------------------------------------------------

class FraudPatternAnalyser:
    """
    Statistical characterisation of fraud patterns in transaction data.

    Analyses:
    * Amount distribution differences between fraud and legitimate transactions
      (two-sample KS test, Mann-Whitney U test, effect size)
    * Hour-of-day fraud concentration (chi-squared test for uniform distribution)
    * Day-of-week fraud rates (chi-squared)
    * Category-level fraud rates with confidence intervals (Wilson score)
    * Device-type fraud rates
    * Cross-border vs domestic fraud rate comparison (Fisher exact test)
    * Fraud type composition and trends

    Parameters
    ----------
    df : pd.DataFrame
        Transaction DataFrame with is_fraud, amount, hour, day_of_week,
        category, device_type, is_cross_border, fraud_type columns.
    config : optional
        Pipeline config or any object with fraud_rate attribute.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: Optional[Any] = None,
    ) -> None:
        self.df     = df.copy()
        self.cfg    = config
        self.fraud  = df[df["is_fraud"] == 1].copy()
        self.legit  = df[df["is_fraud"] == 0].copy()
        self._results: Dict[str, Any] = {}

    # -- amount analysis -----------------------------------------------------

    def analyse_amount_distribution(self) -> Dict[str, Any]:
        """
        Compare fraud vs legitimate amount distributions.

        Tests:
        * Kolmogorov-Smirnov two-sample test (H0: same distribution)
        * Mann-Whitney U test (H0: same location)
        * Cohen's d effect size

        Returns
        -------
        dict
            ks_statistic, ks_pvalue, mw_statistic, mw_pvalue,
            cohens_d, fraud_mean, legit_mean, fraud_median, legit_median,
            fraud_p95, legit_p95.
        """
        f_amt = self.fraud["amount"].dropna().values
        l_amt = self.legit["amount"].dropna().values

        if len(f_amt) < 5 or len(l_amt) < 5:
            return {"error": "Insufficient data for amount analysis"}

        ks_stat, ks_p = scipy_stats.ks_2samp(f_amt, l_amt)
        mw_stat, mw_p = scipy_stats.mannwhitneyu(f_amt, l_amt, alternative="two-sided")

        # Cohen's d
        pooled_std = math.sqrt(
            (f_amt.std()**2 + l_amt.std()**2) / 2
        )
        cohens_d = (f_amt.mean() - l_amt.mean()) / max(pooled_std, 1e-9)

        return {
            "ks_statistic":  round(float(ks_stat), 5),
            "ks_pvalue":     round(float(ks_p), 6),
            "mw_statistic":  round(float(mw_stat), 2),
            "mw_pvalue":     round(float(mw_p), 6),
            "cohens_d":      round(float(cohens_d), 4),
            "fraud_mean":    round(float(f_amt.mean()), 2),
            "legit_mean":    round(float(l_amt.mean()), 2),
            "fraud_median":  round(float(np.median(f_amt)), 2),
            "legit_median":  round(float(np.median(l_amt)), 2),
            "fraud_p95":     round(float(np.percentile(f_amt, 95)), 2),
            "legit_p95":     round(float(np.percentile(l_amt, 95)), 2),
            "significant":   bool(ks_p < SIGNIFICANCE_LEVEL),
        }

    def analyse_temporal_patterns(self) -> Dict[str, Any]:
        """
        Analyse hour-of-day and day-of-week fraud concentrations.

        Tests whether fraud is uniformly distributed across hours/days
        using chi-squared goodness-of-fit tests.

        Returns
        -------
        dict
            hourly_fraud_rates, hourly_chi2_pvalue,
            dow_fraud_rates, dow_chi2_pvalue.
        """
        # Hourly fraud rates
        hourly = []
        for h in range(24):
            total = len(self.df[self.df["hour"] == h])
            fraud = len(self.fraud[self.fraud["hour"] == h])
            rate  = fraud / max(total, 1)
            hourly.append({"hour": h, "fraud_rate": round(rate, 5),
                           "total": total, "fraud_count": fraud})

        # Chi-squared: uniform distribution over hours
        fraud_counts_h = [r["fraud_count"] for r in hourly]
        exp_per_hour   = sum(fraud_counts_h) / 24.0
        if exp_per_hour > 0:
            chi2_h, p_h = scipy_stats.chisquare(
                fraud_counts_h,
                f_exp=[exp_per_hour] * 24
            )
        else:
            chi2_h, p_h = 0.0, 1.0

        # Day-of-week
        dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dow = []
        for d in range(7):
            total = len(self.df[self.df["day_of_week"] == d])
            fraud = len(self.fraud[self.fraud["day_of_week"] == d])
            dow.append({"day": dow_labels[d], "fraud_rate": round(fraud / max(total, 1), 5),
                        "total": total, "fraud_count": fraud})

        fraud_counts_d = [r["fraud_count"] for r in dow]
        exp_per_day    = sum(fraud_counts_d) / 7.0
        if exp_per_day > 0:
            chi2_d, p_d = scipy_stats.chisquare(
                fraud_counts_d,
                f_exp=[exp_per_day] * 7
            )
        else:
            chi2_d, p_d = 0.0, 1.0

        return {
            "hourly_fraud_rates":  hourly,
            "hourly_chi2":         round(float(chi2_h), 4),
            "hourly_pvalue":       round(float(p_h), 6),
            "hourly_nonuniform":   bool(p_h < SIGNIFICANCE_LEVEL),
            "dow_fraud_rates":     dow,
            "dow_chi2":            round(float(chi2_d), 4),
            "dow_pvalue":          round(float(p_d), 6),
            "dow_nonuniform":      bool(p_d < SIGNIFICANCE_LEVEL),
        }

    def analyse_category_rates(self) -> pd.DataFrame:
        """
        Compute per-category fraud rates with Wilson-score confidence intervals.

        Wilson score CI is preferred over normal approximation for small
        sample sizes and extreme proportions.

        Returns
        -------
        pd.DataFrame
            Sorted by fraud_rate descending; columns: category, total,
            fraud_count, fraud_rate, ci_low, ci_high, relative_risk.
        """
        overall_rate = self.df["is_fraud"].mean()
        rows = []
        for cat in self.df["category"].unique():
            sub = self.df[self.df["category"] == cat]
            n   = len(sub)
            k   = int(sub["is_fraud"].sum())
            p   = k / max(n, 1)
            # Wilson score CI
            z    = 1.96  # 95% CI
            denom = 1 + z**2 / n
            centre = (p + z**2 / (2 * n)) / denom
            margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
            ci_lo  = max(0.0, centre - margin)
            ci_hi  = min(1.0, centre + margin)
            rr     = p / max(overall_rate, 1e-9)
            rows.append({
                "category":    cat,
                "total":       n,
                "fraud_count": k,
                "fraud_rate":  round(p, 5),
                "ci_low":      round(ci_lo, 5),
                "ci_high":     round(ci_hi, 5),
                "relative_risk": round(rr, 3),
            })
        return (pd.DataFrame(rows)
                .sort_values("fraud_rate", ascending=False)
                .reset_index(drop=True))

    def analyse_cross_border_effect(self) -> Dict[str, Any]:
        """
        Compare fraud rates for cross-border vs domestic transactions.

        Uses Fisher's exact test (appropriate for 2x2 contingency tables
        with potentially small expected counts).

        Returns
        -------
        dict
            domestic_rate, cross_border_rate, odds_ratio,
            fisher_pvalue, significant.
        """
        domestic = self.df[self.df["is_cross_border"] == 0]
        xborder  = self.df[self.df["is_cross_border"] == 1]

        n_dom_fraud = int(domestic["is_fraud"].sum())
        n_dom_legit = len(domestic) - n_dom_fraud
        n_xb_fraud  = int(xborder["is_fraud"].sum())
        n_xb_legit  = len(xborder) - n_xb_fraud

        dom_rate = n_dom_fraud / max(len(domestic), 1)
        xb_rate  = n_xb_fraud  / max(len(xborder),  1)

        if n_dom_fraud + n_xb_fraud == 0:
            return {"error": "No fraud in dataset"}

        table = [[n_dom_fraud, n_dom_legit], [n_xb_fraud, n_xb_legit]]
        oddsratio, pvalue = scipy_stats.fisher_exact(table)

        return {
            "domestic_rate":    round(dom_rate, 5),
            "cross_border_rate": round(xb_rate, 5),
            "rate_ratio":       round(xb_rate / max(dom_rate, 1e-9), 3),
            "odds_ratio":       round(float(oddsratio), 3),
            "fisher_pvalue":    round(float(pvalue), 6),
            "significant":      bool(pvalue < SIGNIFICANCE_LEVEL),
        }

    def analyse_fraud_type_composition(self) -> Dict[str, Any]:
        """
        Characterise the breakdown of fraud by type.

        Returns
        -------
        dict
            type_counts, type_rates, dominant_type, entropy.
        """
        if "fraud_type" not in self.fraud.columns:
            return {"error": "fraud_type column missing"}
        counts = self.fraud["fraud_type"].value_counts().to_dict()
        total  = max(sum(counts.values()), 1)
        rates  = {k: round(v / total, 4) for k, v in counts.items()}
        # Entropy
        probs    = np.array(list(rates.values()))
        probs    = probs[probs > 0]
        entropy  = float(-np.sum(probs * np.log2(probs)))
        dominant = max(rates, key=rates.get) if rates else "none"
        return {
            "type_counts":   counts,
            "type_rates":    rates,
            "dominant_type": dominant,
            "entropy_bits":  round(entropy, 4),
        }

    def run(self) -> Dict[str, Any]:
        """
        Execute all fraud pattern analyses.

        Returns
        -------
        dict
            Nested results from all analysis methods.
        """
        logger.info("FraudPatternAnalyser.run on %d rows", len(self.df))
        self._results = {
            "amount_distribution":   self.analyse_amount_distribution(),
            "temporal_patterns":     self.analyse_temporal_patterns(),
            "category_rates":        self.analyse_category_rates().to_dict("records"),
            "cross_border_effect":   self.analyse_cross_border_effect(),
            "fraud_type_composition": self.analyse_fraud_type_composition(),
        }
        return self._results



# ---------------------------------------------------------------------------
# SegmentAnalyser
# ---------------------------------------------------------------------------

class SegmentAnalyser:
    """
    Population-segment breakdown of fraud rates with statistical significance.

    Segments the transaction population along configurable dimensions and
    computes fraud rates, lift, and mutual information for each segment.

    Parameters
    ----------
    df : pd.DataFrame
    segment_cols : list of str
        Columns to segment by. Default ['category', 'device_type', 'currency'].
    """

    def __init__(
        self,
        df: pd.DataFrame,
        segment_cols: Optional[List[str]] = None,
    ) -> None:
        self.df   = df.copy()
        self.cols = segment_cols or ["category", "device_type", "currency"]
        self._overall_rate = float(df["is_fraud"].mean())

    def segment_fraud_rates(self, col: str) -> pd.DataFrame:
        """
        Compute fraud rate and lift for each unique value of col.

        Lift = segment_rate / overall_rate.
        A segment with lift > 2.0 is considered high-risk.

        Parameters
        ----------
        col : str
            Column to segment by.

        Returns
        -------
        pd.DataFrame
            Columns: segment_value, count, fraud_count, fraud_rate,
            lift, risk_label.
        """
        if col not in self.df.columns:
            return pd.DataFrame(columns=["segment_value", "count", "fraud_count",
                                         "fraud_rate", "lift", "risk_label"])
        grp = self.df.groupby(col).agg(
            count=("is_fraud", "count"),
            fraud_count=("is_fraud", "sum"),
        ).reset_index()
        grp.columns = ["segment_value", "count", "fraud_count"]
        grp["fraud_rate"] = grp["fraud_count"] / grp["count"].clip(lower=1)
        grp["lift"]        = grp["fraud_rate"] / max(self._overall_rate, 1e-9)
        grp["risk_label"]  = grp["lift"].map(
            lambda l: "critical" if l > 4 else "high" if l > 2 else "medium" if l > 1 else "low"
        )
        grp["fraud_rate"] = grp["fraud_rate"].round(5)
        grp["lift"]        = grp["lift"].round(3)
        return grp.sort_values("fraud_rate", ascending=False).reset_index(drop=True)

    def mutual_information(self, col: str) -> float:
        """
        Compute mutual information I(col; is_fraud).

        Higher MI indicates the column carries more predictive signal about
        whether a transaction is fraudulent.

        Parameters
        ----------
        col : str

        Returns
        -------
        float
            Mutual information in bits.
        """
        if col not in self.df.columns:
            return 0.0
        contingency = pd.crosstab(self.df[col], self.df["is_fraud"])
        n = len(self.df)
        mi = 0.0
        for c in contingency.columns:
            for r in contingency.index:
                n_rc = contingency.loc[r, c]
                n_r  = contingency.loc[r].sum()
                n_c  = contingency[c].sum()
                if n_rc > 0:
                    mi += (n_rc / n) * math.log2(n * n_rc / max(n_r * n_c, 1))
        return max(round(mi, 6), 0.0)

    def run(self) -> Dict[str, Any]:
        """
        Run all segment analyses for configured columns.

        Returns
        -------
        dict
            Keyed by column name; each value has segment_rates and
            mutual_information.
        """
        results = {}
        for col in self.cols:
            if col not in self.df.columns:
                continue
            results[col] = {
                "segment_rates":    self.segment_fraud_rates(col).to_dict("records"),
                "mutual_information": self.mutual_information(col),
                "n_segments":       int(self.df[col].nunique()),
            }
        return results


# ---------------------------------------------------------------------------
# TimeSeriesAnalyser
# ---------------------------------------------------------------------------

class TimeSeriesAnalyser:
    """
    Temporal trend decomposition and anomaly detection for fraud time series.

    Provides:
    * Daily fraud rate time series with 7-day rolling average
    * Trend decomposition (additive model: data = trend + seasonality + residual)
    * Anomaly detection using Z-score on residuals
    * Week-over-week and month-over-month change rates

    Parameters
    ----------
    df : pd.DataFrame
        Must have: timestamp (datetime), is_fraud, amount columns.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(self.df["timestamp"]):
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
        self.df["date"] = self.df["timestamp"].dt.date

    def daily_fraud_series(self) -> pd.DataFrame:
        """
        Compute daily fraud rate, volume, and amount totals.

        Returns
        -------
        pd.DataFrame
            Indexed by date; columns: n_transactions, n_fraud, fraud_rate,
            total_fraud_amount, rolling_7d_rate.
        """
        grp = self.df.groupby("date").agg(
            n_transactions=("is_fraud", "count"),
            n_fraud=("is_fraud", "sum"),
            total_fraud_amount=("amount", lambda x: x[self.df.loc[x.index, "is_fraud"] == 1].sum()),
        ).reset_index()
        grp["fraud_rate"]     = grp["n_fraud"] / grp["n_transactions"].clip(lower=1)
        grp["rolling_7d_rate"] = grp["fraud_rate"].rolling(7, min_periods=1).mean()
        grp["rolling_7d_vol"]  = grp["n_transactions"].rolling(7, min_periods=1).mean()
        return grp.sort_values("date").reset_index(drop=True)

    def detect_anomalies(
        self,
        series: pd.Series,
        z_threshold: float = 2.5,
    ) -> pd.DataFrame:
        """
        Detect anomalous dates using Z-score thresholding.

        Parameters
        ----------
        series : pd.Series
            Time series values (e.g. daily fraud rate).
        z_threshold : float
            Z-score magnitude above which a point is anomalous. Default 2.5.

        Returns
        -------
        pd.DataFrame
            Rows for anomalous points only; columns: index, value, z_score,
            direction.
        """
        mu   = series.mean()
        sigma = series.std()
        if sigma < 1e-9:
            return pd.DataFrame(columns=["index", "value", "z_score", "direction"])
        z_scores = (series - mu) / sigma
        anomalies = z_scores[z_scores.abs() > z_threshold]
        rows = []
        for idx, z in anomalies.items():
            rows.append({
                "index":     idx,
                "value":     round(float(series.iloc[idx] if isinstance(idx, int) else series[idx]), 5),
                "z_score":   round(float(z), 3),
                "direction": "spike" if z > 0 else "dip",
            })
        return pd.DataFrame(rows)

    def week_over_week_change(self, daily_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Compute week-over-week fraud rate change for each week in the dataset.

        Parameters
        ----------
        daily_df : pd.DataFrame, optional
            Output of :meth:`daily_fraud_series`. Computed if not provided.

        Returns
        -------
        pd.DataFrame
            Columns: week_start, fraud_rate, wow_change_pct.
        """
        if daily_df is None:
            daily_df = self.daily_fraud_series()
        daily_df = daily_df.copy()
        daily_df["date"] = pd.to_datetime(daily_df["date"])
        daily_df["week"] = daily_df["date"].dt.to_period("W")
        weekly = daily_df.groupby("week").agg(
            fraud_rate=("fraud_rate", "mean"),
            n_transactions=("n_transactions", "sum"),
        ).reset_index()
        weekly["wow_change_pct"] = weekly["fraud_rate"].pct_change() * 100
        return weekly

    def seasonal_decompose_simple(
        self, daily_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, List[float]]:
        """
        Simple additive decomposition: trend (7-day MA) + day-of-week seasonality.

        Returns
        -------
        dict
            trend, seasonality (7 DOW multipliers), residual.
        """
        if daily_df is None:
            daily_df = self.daily_fraud_series()

        if len(daily_df) < 14:
            return {"error": "Insufficient data (need >= 14 days)"}

        series    = daily_df["fraud_rate"].values
        trend     = pd.Series(series).rolling(7, center=True, min_periods=4).mean().values
        detrended = series - np.nan_to_num(trend, nan=np.nanmean(series))

        # DOW seasonality
        daily_df["date"] = pd.to_datetime(daily_df["date"])
        dow = daily_df["date"].dt.dayofweek.values
        seasonality_by_dow = np.zeros(7)
        for d in range(7):
            mask = dow == d
            if mask.any():
                seasonality_by_dow[d] = float(detrended[mask].mean())

        residual = detrended - seasonality_by_dow[dow]
        return {
            "trend":          [round(float(x), 6) for x in np.nan_to_num(trend)],
            "seasonality_dow": [round(float(x), 6) for x in seasonality_by_dow],
            "residual":        [round(float(x), 6) for x in residual],
        }

    def run(self) -> Dict[str, Any]:
        """Run all time-series analyses and return aggregated results."""
        daily = self.daily_fraud_series()
        anomalies = self.detect_anomalies(daily["fraud_rate"])
        wow = self.week_over_week_change(daily)
        decomp = self.seasonal_decompose_simple(daily)
        return {
            "daily_series":    daily.to_dict("records"),
            "anomalies":       anomalies.to_dict("records"),
            "wow_changes":     wow.to_dict("records"),
            "decomposition":   decomp,
            "n_anomaly_days":  len(anomalies),
        }


# ---------------------------------------------------------------------------
# AnomalyDetector
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """
    Statistical process control for real-time fraud metric monitoring.

    Implements three complementary methods:

    CUSUM (Cumulative Sum)
        Detects sustained shifts in the mean above a threshold.
        Appropriate for detecting gradual drift in fraud rate.

    EWMA (Exponentially Weighted Moving Average)
        Smoothed metric with control limits. Sensitive to small persistent
        shifts without over-reacting to individual outliers.

    Z-Score
        Simple standardised score vs historical baseline. Fast to compute;
        appropriate for burst-detection.

    Parameters
    ----------
    baseline_mean : float
        Historical mean of the metric being monitored.
    baseline_std : float
        Historical standard deviation.
    cusum_threshold : float
        CUSUM alarm threshold in std units. Default 5.0.
    ewma_k : float
        EWMA control-limit multiplier (std units). Default 3.0.
    """

    def __init__(
        self,
        baseline_mean: float,
        baseline_std: float,
        cusum_threshold: float = 5.0,
        ewma_k: float = 3.0,
    ) -> None:
        self.mu     = baseline_mean
        self.sigma  = max(baseline_std, 1e-9)
        self.cusum_k = CUSUM_SLACK * self.sigma
        self.cusum_h = cusum_threshold * self.sigma
        self.ewma_k  = ewma_k
        self._cusum_pos: float = 0.0
        self._cusum_neg: float = 0.0
        self._ewma_val:  float = baseline_mean
        self._history:   List[float] = []

    def update_cusum(self, value: float) -> Dict[str, Any]:
        """
        Update CUSUM statistics with a new observation.

        Parameters
        ----------
        value : float
            New metric value.

        Returns
        -------
        dict
            cusum_pos, cusum_neg, alarm_high, alarm_low.
        """
        self._cusum_pos = max(0, self._cusum_pos + (value - self.mu) - self.cusum_k)
        self._cusum_neg = max(0, self._cusum_neg - (value - self.mu) - self.cusum_k)
        alarm_high = self._cusum_pos > self.cusum_h
        alarm_low  = self._cusum_neg > self.cusum_h
        return {
            "cusum_pos":  round(self._cusum_pos, 6),
            "cusum_neg":  round(self._cusum_neg, 6),
            "alarm_high": bool(alarm_high),
            "alarm_low":  bool(alarm_low),
        }

    def update_ewma(self, value: float) -> Dict[str, Any]:
        """
        Update EWMA statistic with a new observation.

        Parameters
        ----------
        value : float

        Returns
        -------
        dict
            ewma, ucl, lcl, alarm.
        """
        self._ewma_val = EWMA_LAMBDA * value + (1 - EWMA_LAMBDA) * self._ewma_val
        ewma_std = self.sigma * math.sqrt(EWMA_LAMBDA / (2 - EWMA_LAMBDA))
        ucl = self.mu + self.ewma_k * ewma_std
        lcl = self.mu - self.ewma_k * ewma_std
        alarm = bool(self._ewma_val > ucl or self._ewma_val < lcl)
        return {
            "ewma":  round(self._ewma_val, 6),
            "ucl":   round(ucl, 6),
            "lcl":   round(lcl, 6),
            "alarm": alarm,
        }

    def z_score(self, value: float) -> Dict[str, Any]:
        """
        Compute standardised Z-score for a single observation.

        Parameters
        ----------
        value : float

        Returns
        -------
        dict
            z_score, alarm (|z| > 3), direction.
        """
        z = (value - self.mu) / self.sigma
        return {
            "z_score":   round(z, 4),
            "alarm":     bool(abs(z) > 3.0),
            "direction": "high" if z > 0 else "low",
        }

    def process_stream(
        self, values: Sequence[float]
    ) -> pd.DataFrame:
        """
        Process a sequence of values through all three detection methods.

        Parameters
        ----------
        values : sequence of float

        Returns
        -------
        pd.DataFrame
            One row per value; all detection statistics.
        """
        rows = []
        for i, v in enumerate(values):
            cusum = self.update_cusum(v)
            ewma  = self.update_ewma(v)
            zscore = self.z_score(v)
            any_alarm = cusum["alarm_high"] or cusum["alarm_low"] or ewma["alarm"] or zscore["alarm"]
            rows.append({
                "index":       i,
                "value":       round(float(v), 6),
                "cusum_pos":   cusum["cusum_pos"],
                "cusum_neg":   cusum["cusum_neg"],
                "cusum_alarm": cusum["alarm_high"] or cusum["alarm_low"],
                "ewma":        ewma["ewma"],
                "ewma_ucl":    ewma["ucl"],
                "ewma_lcl":    ewma["lcl"],
                "ewma_alarm":  ewma["alarm"],
                "z_score":     zscore["z_score"],
                "z_alarm":     zscore["alarm"],
                "any_alarm":   any_alarm,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CohortAnalyser
# ---------------------------------------------------------------------------

class CohortAnalyser:
    """
    Account cohort-level fraud risk analysis over time.

    Cohorts are defined by account creation month (age_days binned into
    quarters). Tracks whether newer cohorts exhibit higher fraud rates,
    which is a signal of synthetic-identity ring activity.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction DataFrame. Must have account_id, is_fraud, timestamp,
        and optionally age_days.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(self.df["timestamp"]):
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])

    def _age_cohort(self, age_days: float) -> str:
        """Map account age in days to a cohort label."""
        if age_days < 30:
            return "0-30d"
        elif age_days < 90:
            return "30-90d"
        elif age_days < 180:
            return "90-180d"
        elif age_days < 365:
            return "180-365d"
        else:
            return "1y+"

    def cohort_fraud_rates(self) -> pd.DataFrame:
        """
        Compute fraud rate by account-age cohort.

        Returns
        -------
        pd.DataFrame
            Columns: cohort, n_accounts, n_transactions, fraud_rate,
            relative_risk.
        """
        if "age_days" not in self.df.columns:
            return pd.DataFrame(columns=["cohort", "n_transactions", "fraud_rate"])
        df = self.df.copy()
        df["cohort"] = df["age_days"].apply(self._age_cohort)
        grp = df.groupby("cohort").agg(
            n_transactions=("is_fraud", "count"),
            fraud_count=("is_fraud", "sum"),
            n_accounts=("account_id", "nunique"),
        ).reset_index()
        grp["fraud_rate"] = (grp["fraud_count"] / grp["n_transactions"].clip(lower=1)).round(5)
        overall = df["is_fraud"].mean()
        grp["relative_risk"] = (grp["fraud_rate"] / max(overall, 1e-9)).round(3)
        order = ["0-30d", "30-90d", "90-180d", "180-365d", "1y+"]
        grp["sort_key"] = grp["cohort"].map({c: i for i, c in enumerate(order)}).fillna(99)
        return grp.sort_values("sort_key").drop(columns=["sort_key"]).reset_index(drop=True)

    def new_account_fraud_spike(self) -> Dict[str, Any]:
        """
        Test whether new accounts (<30 days) have statistically higher
        fraud rates than established accounts.

        Uses two-sample proportion Z-test.

        Returns
        -------
        dict
            new_rate, established_rate, z_stat, p_value, significant.
        """
        if "age_days" not in self.df.columns:
            return {"error": "age_days column missing"}

        new_accts  = self.df[self.df["age_days"] < 30]
        est_accts  = self.df[self.df["age_days"] >= 90]

        n1, k1 = len(new_accts), int(new_accts["is_fraud"].sum())
        n2, k2 = len(est_accts), int(est_accts["is_fraud"].sum())

        if n1 < 5 or n2 < 5:
            return {"error": "Insufficient data"}

        p1, p2 = k1 / n1, k2 / n2
        p_pool  = (k1 + k2) / (n1 + n2)
        se      = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        z_stat  = (p1 - p2) / max(se, 1e-9)
        p_val   = float(2 * (1 - scipy_stats.norm.cdf(abs(z_stat))))

        return {
            "new_rate":        round(p1, 5),
            "established_rate": round(p2, 5),
            "z_statistic":     round(z_stat, 4),
            "p_value":         round(p_val, 6),
            "significant":     bool(p_val < SIGNIFICANCE_LEVEL),
        }


# ---------------------------------------------------------------------------
# MetricTracker
# ---------------------------------------------------------------------------

class MetricTracker:
    """
    Rolling-window metric tracker for real-time fraud dashboarding.

    Maintains a fixed-length history for each named metric and provides
    rolling statistics (mean, std, trend direction).

    Parameters
    ----------
    window_size : int
        Maximum number of observations to retain. Default 24 (one day at
        hourly granularity).
    """

    def __init__(self, window_size: int = 24) -> None:
        self.window_size = window_size
        self._metrics: Dict[str, List[float]] = defaultdict(list)
        self._timestamps: Dict[str, List[str]] = defaultdict(list)

    def record(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[str] = None,
    ) -> None:
        """
        Record a new observation for a named metric.

        Parameters
        ----------
        metric_name : str
        value : float
        timestamp : str, optional
            ISO timestamp. Defaults to current UTC time.
        """
        ts = timestamp or datetime.utcnow().isoformat()
        history = self._metrics[metric_name]
        history.append(float(value))
        self._timestamps[metric_name].append(ts)
        if len(history) > self.window_size:
            self._metrics[metric_name]    = history[-self.window_size:]
            self._timestamps[metric_name] = self._timestamps[metric_name][-self.window_size:]

    def stats(self, metric_name: str) -> Dict[str, Any]:
        """
        Return rolling statistics for a named metric.

        Parameters
        ----------
        metric_name : str

        Returns
        -------
        dict
            current, mean, std, min, max, trend (up/down/flat), n.
        """
        history = self._metrics.get(metric_name, [])
        if not history:
            return {"error": f"No data for metric {metric_name!r}"}
        n   = len(history)
        cur = history[-1]
        avg = statistics.mean(history)
        std = statistics.stdev(history) if n > 1 else 0.0
        trend = ("up"   if n >= 3 and history[-1] > history[-3]
                 else "down" if n >= 3 and history[-1] < history[-3]
                 else "flat")
        return {
            "current":    round(cur, 5),
            "mean":       round(avg, 5),
            "std":        round(std, 5),
            "min":        round(min(history), 5),
            "max":        round(max(history), 5),
            "trend":      trend,
            "n_obs":      n,
            "latest_ts":  self._timestamps[metric_name][-1],
        }

    def all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return stats for all tracked metrics."""
        return {name: self.stats(name) for name in self._metrics}


# ---------------------------------------------------------------------------
# ReportBuilder
# ---------------------------------------------------------------------------

class ReportBuilder:
    """
    Assembles a comprehensive analytics report from all analyser outputs.

    Collects results from FraudPatternAnalyser, SegmentAnalyser,
    TimeSeriesAnalyser, and CohortAnalyser, then renders them into a
    single structured dict that can be serialised as JSON or used to
    populate an HTML report.

    Parameters
    ----------
    df : pd.DataFrame
        Full transaction dataset.
    config : optional
        Pipeline config (for metadata).
    out_dir : Path, optional
        Directory to save the report. Default: current directory.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: Optional[Any] = None,
        out_dir: Union[str, Path] = ".",
    ) -> None:
        self.df      = df.copy()
        self.cfg     = config
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._report: Dict[str, Any] = {}

    def _metadata(self) -> Dict[str, Any]:
        """Compile dataset metadata."""
        return {
            "n_transactions":    len(self.df),
            "n_fraud":           int(self.df["is_fraud"].sum()),
            "fraud_rate":        round(float(self.df["is_fraud"].mean()), 5),
            "date_range_start":  str(self.df["timestamp"].min()) if "timestamp" in self.df else None,
            "date_range_end":    str(self.df["timestamp"].max()) if "timestamp" in self.df else None,
            "generated_at":      datetime.utcnow().isoformat(),
            "author":            "Gagandeep Kapoor",
        }

    def build(self) -> Dict[str, Any]:
        """
        Run all analysers and assemble the full report.

        Returns
        -------
        dict
            metadata, pattern_analysis, segment_analysis,
            time_series_analysis, cohort_analysis.
        """
        logger.info("Building analytics report on %d transactions...", len(self.df))

        self._report["metadata"] = self._metadata()

        # Pattern analysis
        try:
            pattern = FraudPatternAnalyser(self.df, self.cfg)
            self._report["pattern_analysis"] = pattern.run()
        except Exception as exc:
            logger.warning("Pattern analysis failed: %s", exc)
            self._report["pattern_analysis"] = {"error": str(exc)}

        # Segment analysis
        try:
            segment_cols = [c for c in ["category", "device_type", "currency"]
                            if c in self.df.columns]
            seg = SegmentAnalyser(self.df, segment_cols)
            self._report["segment_analysis"] = seg.run()
        except Exception as exc:
            logger.warning("Segment analysis failed: %s", exc)
            self._report["segment_analysis"] = {"error": str(exc)}

        # Time-series analysis
        if "timestamp" in self.df.columns:
            try:
                ts = TimeSeriesAnalyser(self.df)
                self._report["time_series_analysis"] = ts.run()
            except Exception as exc:
                logger.warning("Time-series analysis failed: %s", exc)
                self._report["time_series_analysis"] = {"error": str(exc)}

        # Cohort analysis
        try:
            cohort = CohortAnalyser(self.df)
            self._report["cohort_analysis"] = {
                "cohort_rates":        cohort.cohort_fraud_rates().to_dict("records"),
                "new_account_spike":   cohort.new_account_fraud_spike(),
            }
        except Exception as exc:
            logger.warning("Cohort analysis failed: %s", exc)
            self._report["cohort_analysis"] = {"error": str(exc)}

        logger.info("Report built successfully.")
        return self._report

    def save_json(self, filename: str = "analytics_report.json") -> Path:
        """
        Save the report to a JSON file.

        Parameters
        ----------
        filename : str

        Returns
        -------
        Path
            Path to the saved file.
        """
        if not self._report:
            self.build()
        out_path = self.out_dir / filename
        with open(out_path, "w") as fh:
            json.dump(self._report, fh, indent=2, default=str)
        logger.info("Report saved to %s", out_path)
        return out_path

    def print_summary(self) -> None:
        """Print a concise summary of the analytics report."""
        if not self._report:
            self.build()
        meta = self._report.get("metadata", {})
        print("
" + "=" * 60)
        print("  ANALYTICS REPORT SUMMARY")
        print("=" * 60)
        print(f"  Transactions: {meta.get('n_transactions', '?'):,}")
        print(f"  Fraud count:  {meta.get('n_fraud', '?'):,}")
        print(f"  Fraud rate:   {meta.get('fraud_rate', 0):.3%}")
        print(f"  Generated:    {meta.get('generated_at', '?')}")
        ts_analysis = self._report.get("time_series_analysis", {})
        print(f"  Anomaly days: {ts_analysis.get('n_anomaly_days', 'N/A')}")
        print("=" * 60 + "
")
