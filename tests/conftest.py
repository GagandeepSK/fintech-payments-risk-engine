"""
Shared pytest configuration and fixtures for the fintech-payments-risk-engine test suite.
Author: Gagandeep Kapoor
"""
import pytest
import numpy as np
import pandas as pd


@pytest.fixture(scope='session')
def rng():
    """Session-scoped random generator for reproducible tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture(scope='session')
def small_txn_df(rng):
    """Minimal 100-row transaction DataFrame used in fast unit tests."""
    n = 100
    return pd.DataFrame({
        'transaction_id': [f'T{i:04d}' for i in range(n)],
        'amount': rng.lognormal(3.5, 1.2, n),
        'is_fraud': rng.binomial(1, 0.05, n),
        'merchant_category': rng.choice(['retail', 'travel', 'online'], n),
        'channel': rng.choice(['card_present', 'card_not_present'], n),
        'fraud_score': rng.uniform(0, 1, n),
        'velocity_1h': rng.integers(0, 8, n),
        'velocity_24h': rng.integers(0, 25, n),
        'customer_id': rng.integers(1, 50, n),
        'timestamp': pd.date_range('2024-06-01', periods=n, freq='30min'),
    })


@pytest.fixture(scope='session')
def fraud_labels(small_txn_df):
    """Boolean fraud mask aligned to small_txn_df."""
    return small_txn_df['is_fraud'].astype(bool).values


@pytest.fixture
def score_array(rng):
    """1-D array of fraud probability scores for threshold/metric tests."""
    return rng.beta(0.5, 9.0, 1000).astype(float)


@pytest.fixture
def binary_labels(rng):
    """Binary ground-truth labels aligned in expectation with score_array."""
    return rng.binomial(1, 0.05, 1000)


@pytest.fixture
def high_risk_txns(small_txn_df):
    """Subset of transactions with fraud_score above 0.7 (high-risk tier)."""
    return small_txn_df[small_txn_df['fraud_score'] > 0.7].copy().reset_index(drop=True)


@pytest.fixture
def low_risk_txns(small_txn_df):
    """Subset of transactions with fraud_score below 0.3 (low-risk tier)."""
    return small_txn_df[small_txn_df['fraud_score'] < 0.3].copy().reset_index(drop=True)

# end of shared fixtures
# done

# fixtures complete -- 100 tests across 6 modules
