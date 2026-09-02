"""
Monte Carlo Fraud Simulation Engine
=====================================
Author: Gagandeep Kapoor
Date:   2026-09-02

Provides a configurable Monte Carlo framework for simulating fraud detection
economics under different operational strategies. The engine models:

* Transaction volume distributions (Poisson-driven intraday seasonality)
* Multi-channel fraud attack scenarios (card-testing, ATO, synthetic ID)
* Analyst throughput and queue dynamics for manual review workloads
* Cost-benefit curves for detect / review / decline strategies
* Confidence intervals via bootstrapped repetitions

Usage
-----
Import as library::

    from src.simulation import MonteCarloSimulator, SimulationConfig
    cfg = SimulationConfig(n_days=90, n_simulations=500)
    sim = MonteCarloSimulator(cfg)
    report = sim.run()

"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import statistics
import time
from collections import defaultdict, namedtuple
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------

# Intraday transaction volume profile -- multiplier vs daily average per hour
INTRADAY_VOLUME_PROFILE: List[float] = [
    0.25, 0.18, 0.14, 0.12, 0.13, 0.22,
    0.55, 0.85, 1.10, 1.20, 1.18, 1.15,
    1.10, 1.05, 1.00, 0.98, 1.00, 1.10,
    1.15, 1.12, 1.05, 0.90, 0.70, 0.45,
]

# Day-of-week multipliers (Mon=0 ... Sun=6)
DOW_VOLUME_MULTIPLIER: List[float] = [0.90, 0.95, 1.00, 1.00, 1.10, 1.20, 0.85]

# Fraud attack type definitions
AttackType = namedtuple("AttackType",
    ["name", "base_rate_uplift", "avg_amount", "std_amount",
     "burst_multiplier", "duration_hours", "probability"])

ATTACK_TYPES: List[AttackType] = [
    AttackType("card_testing",      4.0, 2.5,  2.0, 20.0, 2,  0.15),
    AttackType("account_takeover",  6.0, 250,  180, 5.0,  4,  0.20),
    AttackType("synthetic_identity",3.0, 400,  300, 2.0,  72, 0.10),
    AttackType("card_not_present",  2.5, 80,   60,  3.0,  8,  0.35),
    AttackType("first_party",       1.8, 350,  250, 1.5,  168, 0.12),
    AttackType("social_engineering",3.5, 150,  120, 2.5,  6,  0.08),
]

# Analyst performance parameters
ANALYST_REVIEW_TIME_MIN:  float = 4.0   # minutes per review (mean)
ANALYST_REVIEW_TIME_STD:  float = 2.0   # std dev
ANALYST_ACCURACY:         float = 0.92  # P(correct classification)
ANALYST_SHIFTS_PER_DAY:   int   = 2
ANALYST_HOURS_PER_SHIFT:  float = 7.5

# ---------------------------------------------------------------------------
# SimulationConfig
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """
    Configuration for the Monte Carlo fraud simulation.

    Parameters
    ----------
    n_days : int
        Number of simulation days. Default 90.
    n_simulations : int
        Number of Monte Carlo repetitions for confidence intervals. Default 200.
    daily_transaction_volume : int
        Mean daily transaction count. Default 50_000.
    base_fraud_rate : float
        Baseline fraud rate (no active attack). Default 0.015.
    n_analysts : int
        Number of fraud analysts available for manual review. Default 5.
    decline_threshold : float
        ML score above which transactions are auto-declined. Default 0.60.
    review_threshold : float
        ML score above which transactions are flagged for review. Default 0.35.
    model_roc_auc : float
        Assumed ROC-AUC of the deployed model. Default 0.95.
    model_precision_at_threshold : float
        Model precision at the decline threshold. Default 0.82.
    model_recall_at_threshold : float
        Model recall at the decline threshold. Default 0.78.
    attack_frequency : float
        Mean number of attack events per simulation day. Default 0.1.
    false_negative_cost : float
        Expected loss (GBP) per missed fraud. Default 85.0.
    false_positive_cost : float
        Revenue impact (GBP) per wrongly declined legitimate. Default 12.0.
    review_cost : float
        Analyst cost (GBP) per reviewed transaction. Default 2.50.
    random_seed : int
        Master random seed. Default 42.
    out_dir : Path
        Output directory for simulation results. Default simulation_output.
    """

    n_days:                       int   = 90
    n_simulations:                int   = 200
    daily_transaction_volume:     int   = 50_000
    base_fraud_rate:              float = 0.015
    n_analysts:                   int   = 5
    decline_threshold:            float = 0.60
    review_threshold:             float = 0.35
    model_roc_auc:                float = 0.95
    model_precision_at_threshold: float = 0.82
    model_recall_at_threshold:    float = 0.78
    attack_frequency:             float = 0.10
    false_negative_cost:          float = 85.0
    false_positive_cost:          float = 12.0
    review_cost:                  float = 2.50
    random_seed:                  int   = 42
    out_dir: Path = field(default_factory=lambda: Path("simulation_output"))

    def __post_init__(self) -> None:
        assert 1 <= self.n_days <= 3650, "n_days must be in [1, 3650]"
        assert 1 <= self.n_simulations <= 10_000, "n_simulations must be in [1, 10000]"
        assert 0 < self.base_fraud_rate < 0.5
        assert 0 < self.decline_threshold <= 1.0
        assert 0 < self.review_threshold < self.decline_threshold
        assert 0 <= self.model_roc_auc <= 1.0
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["out_dir"] = str(d["out_dir"])
        return d



# ---------------------------------------------------------------------------
# TransactionVolumeModel
# ---------------------------------------------------------------------------

class TransactionVolumeModel:
    """
    Stochastic intraday transaction volume model.

    Combines Poisson inter-arrival times with a deterministic intraday
    volume profile (INTRADAY_VOLUME_PROFILE) and day-of-week seasonality
    (DOW_VOLUME_MULTIPLIER).

    Parameters
    ----------
    config : SimulationConfig
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.cfg = config
        self.rng = np.random.default_rng(config.random_seed)

    def daily_volume(self, day_index: int) -> int:
        """
        Sample total transaction count for a given simulation day.

        The daily volume follows a Negative Binomial distribution centred
        on the configured daily mean, with variance proportional to the mean
        (overdispersion factor = 1.2).

        Parameters
        ----------
        day_index : int
            Day within the simulation (0-indexed).

        Returns
        -------
        int
            Total transactions for the day.
        """
        dow = day_index % 7
        mean_vol = (self.cfg.daily_transaction_volume
                    * DOW_VOLUME_MULTIPLIER[dow])
        # Negative binomial parametrisation: n = r, p = r / (r + mu)
        r   = mean_vol / 0.2   # overdispersion
        p   = r / (r + mean_vol)
        vol = int(self.rng.negative_binomial(int(r), p))
        return max(vol, 1)

    def hourly_volumes(self, total_volume: int, day_index: int) -> List[int]:
        """
        Distribute a day's total volume across 24 hours using the
        intraday profile, with Poisson noise at each hour.

        Parameters
        ----------
        total_volume : int
            Total transactions for the day.
        day_index : int
            Used to compute day-of-week profile offset.

        Returns
        -------
        List[int]
            24-element list of hourly transaction counts.
        """
        profile = np.array(INTRADAY_VOLUME_PROFILE)
        weights = profile / profile.sum()
        expected_per_hour = weights * total_volume
        hourly = [int(self.rng.poisson(lam=max(e, 0.1)))
                  for e in expected_per_hour]
        # Clip to reasonable bounds
        return [max(0, h) for h in hourly]

    def simulate_arrivals(self, n_days: int) -> pd.DataFrame:
        """
        Generate a full arrival schedule for n_days.

        Returns
        -------
        pd.DataFrame
            Columns: day, hour, volume, cumulative_volume.
        """
        rows = []
        cumul = 0
        for d in range(n_days):
            total = self.daily_volume(d)
            hourly = self.hourly_volumes(total, d)
            for h, vol in enumerate(hourly):
                cumul += vol
                rows.append({"day": d, "hour": h, "volume": vol,
                              "cumulative_volume": cumul})
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# FraudAttackSimulator
# ---------------------------------------------------------------------------

class FraudAttackSimulator:
    """
    Simulates discrete fraud attack events superimposed on baseline fraud.

    Attack events follow a Poisson process with rate config.attack_frequency
    per day. Each event selects an attack type from ATTACK_TYPES (weighted
    by AttackType.probability) and elevates the fraud rate during the attack
    duration.

    Parameters
    ----------
    config : SimulationConfig
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.cfg     = config
        self.rng     = np.random.default_rng(config.random_seed + 1)
        self._py_rng = random.Random(config.random_seed + 1)

    def _sample_attack_type(self) -> AttackType:
        """Sample an attack type according to probability weights."""
        weights = [a.probability for a in ATTACK_TYPES]
        return self._py_rng.choices(ATTACK_TYPES, weights=weights, k=1)[0]

    def generate_attack_schedule(self) -> pd.DataFrame:
        """
        Generate a schedule of attack events over the simulation period.

        Each row represents one attack event with its start day/hour,
        duration, type, and instantaneous fraud-rate uplift factor.

        Returns
        -------
        pd.DataFrame
            Columns: attack_id, start_day, start_hour, duration_hours,
            attack_type, base_rate_uplift, burst_multiplier, avg_amount.
        """
        n_days    = self.cfg.n_days
        # Poisson number of attacks over the period
        n_attacks = int(self.rng.poisson(self.cfg.attack_frequency * n_days))
        rows      = []
        for i in range(n_attacks):
            attack    = self._sample_attack_type()
            start_day = int(self.rng.integers(0, n_days))
            start_hr  = int(self.rng.integers(0, 24))
            rows.append({
                "attack_id":        f"ATK{i:05d}",
                "start_day":        start_day,
                "start_hour":       start_hr,
                "duration_hours":   attack.duration_hours,
                "attack_type":      attack.name,
                "base_rate_uplift": attack.base_rate_uplift,
                "burst_multiplier": attack.burst_multiplier,
                "avg_amount":       attack.avg_amount,
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["attack_id", "start_day", "start_hour", "duration_hours",
                     "attack_type", "base_rate_uplift", "burst_multiplier", "avg_amount"])

    def fraud_rate_at(
        self,
        day: int,
        hour: int,
        attack_schedule: pd.DataFrame,
    ) -> float:
        """
        Compute the effective fraud rate at a given (day, hour).

        Baseline rate is uplifted by any active attacks at that time.
        Multiple concurrent attacks are combined multiplicatively
        (capped at 0.50).

        Parameters
        ----------
        day : int
        hour : int
        attack_schedule : pd.DataFrame
            Output of :meth:`generate_attack_schedule`.

        Returns
        -------
        float
            Effective fraud rate in [base_fraud_rate, 0.50].
        """
        rate = self.cfg.base_fraud_rate
        if attack_schedule.empty:
            return rate
        absolute_hour = day * 24 + hour
        for _, atk in attack_schedule.iterrows():
            atk_start = atk["start_day"] * 24 + atk["start_hour"]
            atk_end   = atk_start + atk["duration_hours"]
            if atk_start <= absolute_hour < atk_end:
                rate = rate * atk["base_rate_uplift"]
        return float(min(rate, 0.50))


# ---------------------------------------------------------------------------
# AnalystQueueModel
# ---------------------------------------------------------------------------

class AnalystQueueModel:
    """
    M/G/c queuing model for the manual review analyst pool.

    Simulates queue dynamics as transactions flow through the review process.
    Tracks:
    * Queue length (transactions awaiting review)
    * Wait time per transaction (hours from arrival to review completion)
    * Analyst utilisation rate
    * Review accuracy and its downstream economic impact

    Parameters
    ----------
    config : SimulationConfig
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.cfg = config
        self.rng = np.random.default_rng(config.random_seed + 2)
        # Analyst capacity: transactions per hour per analyst
        review_time_hrs = ANALYST_REVIEW_TIME_MIN / 60.0
        self.capacity_per_analyst_hr = 1.0 / review_time_hrs
        self.total_capacity_hr = (self.cfg.n_analysts
                                  * self.capacity_per_analyst_hr
                                  * ANALYST_HOURS_PER_SHIFT)

    def _sample_review_time(self, n: int) -> np.ndarray:
        """Sample n review times (minutes) from a truncated normal distribution."""
        times = self.rng.normal(ANALYST_REVIEW_TIME_MIN, ANALYST_REVIEW_TIME_STD, n)
        return np.clip(times, 1.0, 30.0)

    def simulate_day(
        self,
        n_review_arrivals: int,
        queue_backlog: int = 0,
    ) -> Dict[str, Any]:
        """
        Simulate one day of analyst review queue dynamics.

        Parameters
        ----------
        n_review_arrivals : int
            Transactions arriving for review during this day.
        queue_backlog : int
            Carryover queue from the previous day.

        Returns
        -------
        dict
            reviewed_count, queue_end_of_day, mean_wait_hr,
            utilisation_rate, tp_from_review, fp_from_review,
            cost_savings_from_review.
        """
        total_to_process = n_review_arrivals + queue_backlog
        # How many can be processed today?
        capacity    = int(self.total_capacity_hr)
        reviewed    = min(total_to_process, capacity)
        backlog_eod = total_to_process - reviewed

        # Wait time: backlog items waited longer
        if reviewed > 0:
            review_times  = self._sample_review_time(reviewed)
            mean_wait     = float(review_times.mean()) / 60.0   # hours
            util_rate     = reviewed / max(capacity, 1)
        else:
            mean_wait = 0.0
            util_rate = 0.0

        # Accuracy: TP = correctly caught fraud, FP = wrongly flagged legit
        if reviewed > 0:
            # Approximate: among reviewed, assume 30% are actual fraud
            n_actual_fraud = int(reviewed * 0.30)
            n_actual_legit = reviewed - n_actual_fraud
            tp = int(n_actual_fraud * ANALYST_ACCURACY)
            fp = n_actual_legit - int(n_actual_legit * ANALYST_ACCURACY)
            fp = max(fp, 0)
            fn = n_actual_fraud - tp
            cost_savings = (tp * self.cfg.false_negative_cost
                            - fp * self.cfg.false_positive_cost
                            - reviewed * self.cfg.review_cost)
        else:
            tp = fp = fn = 0
            cost_savings = 0.0

        return {
            "reviewed_count":         reviewed,
            "queue_end_of_day":       backlog_eod,
            "mean_wait_hr":           round(mean_wait, 3),
            "utilisation_rate":       round(util_rate, 4),
            "tp_from_review":         tp,
            "fp_from_review":         fp,
            "cost_savings_from_review": round(cost_savings, 2),
        }

    def simulate_period(
        self, daily_review_counts: List[int]
    ) -> pd.DataFrame:
        """
        Simulate analyst queue over multiple days.

        Parameters
        ----------
        daily_review_counts : List[int]
            Number of review-flagged transactions per day.

        Returns
        -------
        pd.DataFrame
            One row per day with queue dynamics and economics.
        """
        rows    = []
        backlog = 0
        for day, n_arrivals in enumerate(daily_review_counts):
            result  = self.simulate_day(n_arrivals, backlog)
            backlog = result["queue_end_of_day"]
            rows.append({"day": day, **result})
        return pd.DataFrame(rows)



# ---------------------------------------------------------------------------
# EconomicModel
# ---------------------------------------------------------------------------

class EconomicModel:
    """
    Cost-benefit economics model for fraud detection strategy evaluation.

    Computes gross fraud losses, recovered amounts, operational costs, and
    net savings for a given detection strategy over a simulation run.

    The model distinguishes three transaction disposition categories:

    Auto-Decline
        Transactions scoring above decline_threshold are blocked instantly.
        True positives here save the full fraud loss; false positives incur
        revenue impact.

    Manual Review
        Transactions between review_threshold and decline_threshold are
        queued for analyst review. Analysts catch a fraction (ANALYST_ACCURACY)
        of true fraud.

    Pass
        Transactions scoring below review_threshold are authorised.
        A fraction of actual fraud in this bucket is missed (false negatives).

    Parameters
    ----------
    config : SimulationConfig
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.cfg = config
        self.rng = np.random.default_rng(config.random_seed + 3)

    def daily_economics(
        self,
        n_transactions:    int,
        fraud_rate:        float,
        n_auto_declined:   int,
        n_reviewed:        int,
        n_passed:          int,
    ) -> Dict[str, float]:
        """
        Compute daily economic outcomes for a given disposition split.

        Uses the configured model precision/recall to apportion outcomes
        within each bucket.

        Parameters
        ----------
        n_transactions : int
            Total transactions for the day.
        fraud_rate : float
            Effective fraud rate (after any attack uplift).
        n_auto_declined : int
            Count of auto-declined transactions.
        n_reviewed : int
            Count of transactions sent to manual review.
        n_passed : int
            Count of authorised transactions.

        Returns
        -------
        dict
            gross_fraud_loss, recovered_auto, recovered_review,
            false_positive_cost, review_cost, net_savings.
        """
        n_fraud_today     = int(n_transactions * fraud_rate)
        avg_fraud_amount  = 95.0    # GBP: weighted average fraud transaction
        gross_loss        = n_fraud_today * avg_fraud_amount

        # Auto-decline bucket
        precision = self.cfg.model_precision_at_threshold
        recall    = self.cfg.model_recall_at_threshold
        if n_auto_declined > 0:
            tp_auto = int(n_auto_declined * precision)
            fp_auto = n_auto_declined - tp_auto
        else:
            tp_auto = fp_auto = 0
        recovered_auto    = tp_auto * avg_fraud_amount
        fp_cost_auto      = fp_auto * self.cfg.false_positive_cost

        # Review bucket (analyst catches accuracy fraction)
        if n_reviewed > 0:
            n_fraud_in_review = int(n_reviewed * fraud_rate * 2.0)  # elevated in review
            tp_review = int(n_fraud_in_review * ANALYST_ACCURACY)
        else:
            tp_review = 0
        recovered_review = tp_review * avg_fraud_amount
        review_cost_total = n_reviewed * self.cfg.review_cost

        # Pass bucket: remaining fraud leaks through
        n_fraud_passed    = max(n_fraud_today - tp_auto - tp_review, 0)
        fn_loss           = n_fraud_passed * avg_fraud_amount

        net_savings = (recovered_auto + recovered_review
                       - fp_cost_auto - review_cost_total - fn_loss)
        return {
            "gross_fraud_loss":  round(gross_loss, 2),
            "recovered_auto":    round(recovered_auto, 2),
            "recovered_review":  round(recovered_review, 2),
            "false_positive_cost": round(fp_cost_auto, 2),
            "review_cost":       round(review_cost_total, 2),
            "fn_loss":           round(fn_loss, 2),
            "net_savings":       round(net_savings, 2),
            "capture_rate":      round(
                (tp_auto + tp_review) / max(n_fraud_today, 1), 4
            ),
        }

    def period_summary(self, daily_results: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Aggregate daily economics over a simulation period.

        Parameters
        ----------
        daily_results : List[dict]
            One dict per day from :meth:`daily_economics`.

        Returns
        -------
        dict
            Totals and averages across the period.
        """
        keys = ["gross_fraud_loss", "recovered_auto", "recovered_review",
                "false_positive_cost", "review_cost", "fn_loss", "net_savings"]
        totals = {k: sum(d[k] for d in daily_results) for k in keys}
        n_days = max(len(daily_results), 1)
        avgs   = {f"avg_daily_{k}": round(v / n_days, 2) for k, v in totals.items()}
        avg_capture = statistics.mean(d["capture_rate"] for d in daily_results)
        return {**totals, **avgs, "mean_capture_rate": round(avg_capture, 4)}


# ---------------------------------------------------------------------------
# MonteCarloSimulator
# ---------------------------------------------------------------------------

class MonteCarloSimulator:
    """
    Monte Carlo engine: runs the full simulation config.n_simulations times
    and aggregates results with confidence intervals.

    Each simulation run:
    1. Generates a transaction volume schedule.
    2. Generates a fraud attack schedule.
    3. For each day / hour: computes fraud rate, disposition split.
    4. Computes daily economics.
    5. Simulates analyst queue dynamics.

    After all runs, computes mean, std, and 95% CI for key metrics.

    Parameters
    ----------
    config : SimulationConfig
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.cfg = config
        self._rng = np.random.default_rng(config.random_seed)
        self._results_cache: Optional[pd.DataFrame] = None
        logger.info("MonteCarloSimulator initialised (%d runs, %d days)",
                    config.n_simulations, config.n_days)

    def _run_single_simulation(self, sim_id: int) -> Dict[str, Any]:
        """
        Execute one full simulation run.

        Parameters
        ----------
        sim_id : int
            Simulation index (used to offset random seeds for independence).

        Returns
        -------
        dict
            Aggregated metrics for this simulation run.
        """
        seed_offset = sim_id * 137  # ensure independence
        cfg_copy    = SimulationConfig(
            **{k: v for k, v in asdict(self.cfg).items()
               if k != "random_seed"},
            random_seed=self.cfg.random_seed + seed_offset,
            out_dir=self.cfg.out_dir,
        )
        vol_model    = TransactionVolumeModel(cfg_copy)
        attack_sim   = FraudAttackSimulator(cfg_copy)
        econ_model   = EconomicModel(cfg_copy)
        queue_model  = AnalystQueueModel(cfg_copy)

        attack_sched = attack_sim.generate_attack_schedule()
        daily_review_counts: List[int] = []
        daily_econ:         List[Dict[str, float]] = []

        for day in range(self.cfg.n_days):
            n_txn = vol_model.daily_volume(day)
            hourly = vol_model.hourly_volumes(n_txn, day)

            day_declined = 0
            day_reviewed = 0
            day_passed   = 0
            day_fraud_rate = 0.0

            for hour, n_hr in enumerate(hourly):
                if n_hr == 0:
                    continue
                fr = attack_sim.fraud_rate_at(day, hour, attack_sched)
                day_fraud_rate += fr / 24.0

                # Disposition split (simplified model)
                # Transactions above decline_threshold score: fraction = recall
                n_decline = int(n_hr * self.cfg.model_recall_at_threshold
                                * (fr / max(self.cfg.base_fraud_rate, 1e-6)) * 0.1)
                n_decline = min(n_decline, n_hr)
                n_review  = int((n_hr - n_decline) * 0.05)  # 5% review rate
                n_pass    = n_hr - n_decline - n_review
                day_declined += n_decline
                day_reviewed += n_review
                day_passed   += max(n_pass, 0)

            daily_review_counts.append(day_reviewed)
            econ = econ_model.daily_economics(
                n_txn, day_fraud_rate, day_declined, day_reviewed, day_passed
            )
            daily_econ.append(econ)

        # Queue simulation
        queue_df  = queue_model.simulate_period(daily_review_counts)
        econ_summ = econ_model.period_summary(daily_econ)

        return {
            "sim_id":              sim_id,
            "total_net_savings":   econ_summ["net_savings"],
            "mean_capture_rate":   econ_summ["mean_capture_rate"],
            "total_recovered":     econ_summ["recovered_auto"] + econ_summ["recovered_review"],
            "total_fn_loss":       econ_summ["fn_loss"],
            "total_fp_cost":       econ_summ["false_positive_cost"],
            "total_review_cost":   econ_summ["review_cost"],
            "mean_queue_length":   float(queue_df["queue_end_of_day"].mean()),
            "max_queue_length":    float(queue_df["queue_end_of_day"].max()),
            "mean_utilisation":    float(queue_df["utilisation_rate"].mean()),
            "n_attack_events":     len(attack_sched),
        }

    def run(self) -> Dict[str, Any]:
        """
        Execute all Monte Carlo simulations and compile the aggregate report.

        Returns
        -------
        dict
            Comprehensive report with per-metric mean, std, p5, p25,
            median, p75, p95, and confidence intervals.
        """
        t0 = time.monotonic()
        logger.info("Starting %d simulation runs...", self.cfg.n_simulations)
        sim_results = []
        for i in range(self.cfg.n_simulations):
            if i % 50 == 0:
                logger.info("  Run %d/%d...", i, self.cfg.n_simulations)
            try:
                sim_results.append(self._run_single_simulation(i))
            except Exception as exc:
                logger.warning("Simulation %d failed: %s", i, exc)

        if not sim_results:
            return {"error": "All simulation runs failed"}

        df = pd.DataFrame(sim_results)
        self._results_cache = df

        # Build aggregate statistics for each numeric metric
        metrics = [c for c in df.columns if c != "sim_id"]
        summary: Dict[str, Any] = {}
        for m in metrics:
            col = df[m].dropna().values
            if len(col) == 0:
                continue
            summary[m] = {
                "mean":    round(float(col.mean()),  2),
                "std":     round(float(col.std()),   2),
                "p5":      round(float(np.percentile(col, 5)),  2),
                "p25":     round(float(np.percentile(col, 25)), 2),
                "median":  round(float(np.median(col)), 2),
                "p75":     round(float(np.percentile(col, 75)), 2),
                "p95":     round(float(np.percentile(col, 95)), 2),
                "ci95_lo": round(float(np.percentile(col, 2.5)),  2),
                "ci95_hi": round(float(np.percentile(col, 97.5)), 2),
            }

        elapsed = time.monotonic() - t0
        report = {
            "config":           self.cfg.to_dict(),
            "n_successful_runs": len(sim_results),
            "elapsed_s":         round(elapsed, 2),
            "metrics":           summary,
        }
        self._save_report(report)
        logger.info("Simulation complete in %.1fs", elapsed)
        return report

    def _save_report(self, report: Dict[str, Any]) -> None:
        """Persist the simulation report as JSON."""
        out_path = self.cfg.out_dir / "simulation_report.json"
        with open(out_path, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        logger.info("Report saved to %s", out_path)
        if self._results_cache is not None:
            csv_path = self.cfg.out_dir / "simulation_runs.csv"
            self._results_cache.to_csv(csv_path, index=False)

    def sensitivity_analysis(
        self,
        parameter: str,
        values: List[Any],
    ) -> pd.DataFrame:
        """
        Sweep a single configuration parameter and compare net savings.

        For each value in 'values', runs n_simulations simulations and
        records the mean and 95% CI of total_net_savings.

        Parameters
        ----------
        parameter : str
            Attribute name of SimulationConfig to vary.
        values : list
            Values to test.

        Returns
        -------
        pd.DataFrame
            One row per value: parameter_value, mean_net_savings,
            p5_net_savings, p95_net_savings.
        """
        logger.info("Sensitivity analysis: %s over %d values", parameter, len(values))
        rows = []
        base = asdict(self.cfg)
        base["out_dir"] = str(self.cfg.out_dir)
        for val in values:
            cfg_dict = {**base, parameter: val}
            try:
                trial_cfg = SimulationConfig(**cfg_dict)
                trial_sim = MonteCarloSimulator(trial_cfg)
                result    = trial_sim.run()
                savings   = result.get("metrics", {}).get("total_net_savings", {})
                rows.append({
                    "parameter_value": val,
                    "mean_net_savings": savings.get("mean", 0),
                    "p5_net_savings":   savings.get("p5",   0),
                    "p95_net_savings":  savings.get("p95",  0),
                })
            except Exception as exc:
                logger.warning("Sensitivity run failed for %s=%s: %s", parameter, val, exc)
        return pd.DataFrame(rows)

    def compare_strategies(
        self,
        strategies: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Compare multiple detection strategies head-to-head.

        Each strategy is a dict of SimulationConfig overrides. Runs
        n_simulations per strategy and ranks by mean net savings.

        Parameters
        ----------
        strategies : list of dict
            Each dict may override: decline_threshold, review_threshold,
            n_analysts, model_recall_at_threshold, etc.

        Returns
        -------
        pd.DataFrame
            Ranked by mean_net_savings; one row per strategy.
        """
        logger.info("Comparing %d strategies...", len(strategies))
        rows = []
        base = asdict(self.cfg)
        base["out_dir"] = str(self.cfg.out_dir)
        for i, strategy in enumerate(strategies):
            cfg_dict = {**base, **strategy}
            try:
                trial_cfg = SimulationConfig(**cfg_dict)
                trial_sim = MonteCarloSimulator(trial_cfg)
                result    = trial_sim.run()
                savings   = result.get("metrics", {}).get("total_net_savings", {})
                capture   = result.get("metrics", {}).get("mean_capture_rate",  {})
                rows.append({
                    "strategy_id":      f"S{i+1:02d}",
                    "description":      str(strategy),
                    "mean_net_savings": savings.get("mean", 0),
                    "p5_net_savings":   savings.get("p5",   0),
                    "p95_net_savings":  savings.get("p95",  0),
                    "mean_capture_rate": capture.get("mean", 0),
                    "decline_threshold": strategy.get("decline_threshold",
                                                       self.cfg.decline_threshold),
                    "review_threshold":  strategy.get("review_threshold",
                                                       self.cfg.review_threshold),
                })
            except Exception as exc:
                logger.warning("Strategy %d failed: %s", i, exc)
        df = pd.DataFrame(rows)
        return df.sort_values("mean_net_savings", ascending=False).reset_index(drop=True)

    def print_summary(self, report: Optional[Dict[str, Any]] = None) -> None:
        """
        Print a human-readable Monte Carlo simulation summary.

        Parameters
        ----------
        report : dict, optional
            Output of :meth:`run`. If None, runs a new simulation.
        """
        if report is None:
            report = self.run()
        metrics = report.get("metrics", {})
        print("
" + "=" * 60)
        print("  MONTE CARLO SIMULATION SUMMARY")
        print("=" * 60)
        print(f"  Runs:    {report.get('n_successful_runs', 0)}")
        print(f"  Elapsed: {report.get('elapsed_s', 0):.1f}s")
        print()
        for metric, stats in metrics.items():
            print(f"  {metric:<30s}  mean={stats['mean']:>12.2f}  ",
                  end="")
            print(f"95%CI=[{stats['ci95_lo']:.2f}, {stats['ci95_hi']:.2f}]")
        print("=" * 60 + "
")


# ---------------------------------------------------------------------------
# ScenarioLibrary
# ---------------------------------------------------------------------------

class ScenarioLibrary:
    """
    Pre-built simulation scenarios for common fraud strategy comparisons.

    Scenarios are defined as lists of SimulationConfig overrides suitable
    for :meth:`MonteCarloSimulator.compare_strategies`.

    Available scenario suites:
    * threshold_sweep         -- vary decline threshold from 0.30 to 0.80
    * analyst_capacity_sweep  -- vary n_analysts from 2 to 20
    * attack_frequency_sweep  -- vary attack_frequency from 0.05 to 0.50
    """

    @staticmethod
    def threshold_sweep(
        decline_values: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Return strategy overrides for a sweep over decline threshold.

        Parameters
        ----------
        decline_values : list of float, optional
            Thresholds to test. Defaults to [0.30, 0.40, 0.50, 0.60, 0.70, 0.80].

        Returns
        -------
        List[dict]
            Strategy override dicts for compare_strategies.
        """
        thresholds = decline_values or [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
        return [
            {"decline_threshold": t,
             "review_threshold":  max(0.05, t - 0.20)}
            for t in thresholds
        ]

    @staticmethod
    def analyst_capacity_sweep(
        analyst_counts: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Return strategy overrides for a sweep over analyst headcount.

        Parameters
        ----------
        analyst_counts : list of int, optional
            Headcounts to test. Defaults to [1, 2, 3, 5, 8, 10, 15, 20].

        Returns
        -------
        List[dict]
        """
        counts = analyst_counts or [1, 2, 3, 5, 8, 10, 15, 20]
        return [{"n_analysts": n} for n in counts]

    @staticmethod
    def attack_frequency_sweep(
        frequencies: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Return strategy overrides for a sweep over attack frequency.

        Parameters
        ----------
        frequencies : list of float, optional
            Attack events per day. Defaults to [0.01, 0.05, 0.10, 0.20, 0.50].

        Returns
        -------
        List[dict]
        """
        freqs = frequencies or [0.01, 0.05, 0.10, 0.20, 0.50]
        return [{"attack_frequency": f} for f in freqs]

    @staticmethod
    def model_performance_sweep(
        auc_values: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate economics across a range of model AUC values.

        Models with higher AUC can use more aggressive thresholds
        without excessive false positives. This sweep explores that trade-off.

        Parameters
        ----------
        auc_values : list of float, optional
            AUC values to test. Defaults to [0.75, 0.80, 0.85, 0.90, 0.95].

        Returns
        -------
        List[dict]
        """
        aucs = auc_values or [0.75, 0.80, 0.85, 0.90, 0.95]
        precision_map = {0.75: 0.55, 0.80: 0.63, 0.85: 0.72, 0.90: 0.80, 0.95: 0.87}
        recall_map    = {0.75: 0.60, 0.80: 0.67, 0.85: 0.73, 0.90: 0.79, 0.95: 0.84}
        return [
            {
                "model_roc_auc":                auc,
                "model_precision_at_threshold": precision_map.get(auc, 0.75),
                "model_recall_at_threshold":    recall_map.get(auc, 0.70),
            }
            for auc in aucs
        ]


# ---------------------------------------------------------------------------
# BootstrapAnalyser
# ---------------------------------------------------------------------------

class BootstrapAnalyser:
    """
    Bootstrap resampling for confidence intervals on observed fraud metrics.

    Wraps numpy bootstrap resampling to produce empirical confidence intervals
    on any scalar statistic (mean, median, quantile, custom function) applied
    to fraud score distributions or economic outcomes.

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap samples. Default 1000.
    confidence : float
        Confidence level for intervals. Default 0.95.
    random_seed : int
        Reproducibility seed. Default 42.
    """

    def __init__(
        self,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        random_seed: int = 42,
    ) -> None:
        self.n_bootstrap = n_bootstrap
        self.confidence  = confidence
        self.rng         = np.random.default_rng(random_seed)

    def bootstrap_ci(
        self,
        data: np.ndarray,
        statistic: Callable,
    ) -> Tuple[float, float, float]:
        """
        Compute a bootstrap confidence interval for a scalar statistic.

        Parameters
        ----------
        data : np.ndarray
            Observed data.
        statistic : callable
            Function mapping an ndarray to a scalar (e.g. np.mean, np.median).

        Returns
        -------
        Tuple[float, float, float]
            (point_estimate, ci_lower, ci_upper)
        """
        point_est = float(statistic(data))
        bootstrap_stats = np.array([
            statistic(self.rng.choice(data, size=len(data), replace=True))
            for _ in range(self.n_bootstrap)
        ])
        alpha   = 1.0 - self.confidence
        ci_lo   = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
        ci_hi   = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))
        return point_est, ci_lo, ci_hi

    def bootstrap_comparison(
        self,
        data_a: np.ndarray,
        data_b: np.ndarray,
        statistic: Callable = np.mean,
    ) -> Dict[str, Any]:
        """
        Bootstrap comparison of two strategies / distributions.

        Computes the probability that strategy A outperforms B on the
        given statistic, plus individual CIs for each.

        Parameters
        ----------
        data_a : np.ndarray
            Metric values (e.g. net_savings) under strategy A.
        data_b : np.ndarray
            Metric values under strategy B.
        statistic : callable

        Returns
        -------
        dict
            point_a, ci_a, point_b, ci_b, p_a_better (float), delta.
        """
        est_a, lo_a, hi_a = self.bootstrap_ci(data_a, statistic)
        est_b, lo_b, hi_b = self.bootstrap_ci(data_b, statistic)

        # P(A > B) via bootstrap overlap
        deltas = []
        for _ in range(self.n_bootstrap):
            s_a = statistic(self.rng.choice(data_a, len(data_a), replace=True))
            s_b = statistic(self.rng.choice(data_b, len(data_b), replace=True))
            deltas.append(s_a - s_b)
        p_a_better = float(np.mean(np.array(deltas) > 0))

        return {
            "point_a": round(est_a, 4),
            "ci_a":   (round(lo_a, 4), round(hi_a, 4)),
            "point_b": round(est_b, 4),
            "ci_b":   (round(lo_b, 4), round(hi_b, 4)),
            "p_a_better": round(p_a_better, 4),
            "delta":      round(est_a - est_b, 4),
            "delta_ci":   (round(float(np.percentile(deltas, 2.5)), 4),
                            round(float(np.percentile(deltas, 97.5)), 4)),
        }

