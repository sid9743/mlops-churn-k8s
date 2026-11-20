import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from generate_data import generate  # noqa: E402


def test_schema_and_no_nulls():
    df = generate(1000, seed=1)
    assert df.isna().sum().sum() == 0
    assert set(df.churn.unique()) <= {0, 1}
    assert df.tenure_months.between(1, 72).all()


def test_churn_rate_is_realistic():
    rate = generate(5000, seed=7).churn.mean()
    assert 0.15 < rate < 0.45


def test_month_to_month_churns_more():
    df = generate(5000, seed=3)
    by_contract = df.groupby("contract").churn.mean()
    assert by_contract["month-to-month"] > by_contract["two-year"]
