"""Generate a synthetic telecom customer churn dataset.

The data is synthetic (no real customers), but the churn signal follows patterns seen in
public telecom datasets: month-to-month contracts, short tenure, high monthly charges and
many support calls all raise the chance of churn.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CONTRACTS = ["month-to-month", "one-year", "two-year"]
INTERNET = ["dsl", "fiber", "none"]
PAYMENT = ["upi", "credit-card", "bank-transfer", "cash"]


def generate(n_rows: int, seed: int, charge_shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    contract = rng.choice(CONTRACTS, n_rows, p=[0.55, 0.25, 0.20])
    internet = rng.choice(INTERNET, n_rows, p=[0.35, 0.45, 0.20])
    payment = rng.choice(PAYMENT, n_rows)
    tenure = rng.integers(1, 73, n_rows)
    monthly = np.clip(rng.normal(70 + charge_shift, 25, n_rows), 20, 150).round(2)
    support_calls = rng.poisson(1.5, n_rows)
    senior = rng.binomial(1, 0.16, n_rows)

    logit = (
        -1.2
        + 1.4 * (contract == "month-to-month")
        - 0.9 * (contract == "two-year")
        + 0.6 * (internet == "fiber")
        - 0.035 * tenure
        + 0.018 * (monthly - 70)
        + 0.35 * support_calls
        + 0.3 * senior
        + 0.25 * (payment == "cash")
    )
    churn = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    return pd.DataFrame(
        {
            "customer_id": [f"C{seed:02d}{i:06d}" for i in range(n_rows)],
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": (monthly * tenure).round(2),
            "contract": contract,
            "internet_service": internet,
            "payment_method": payment,
            "support_calls": support_calls,
            "senior_citizen": senior,
            "churn": churn,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/churn.csv")
    args = parser.parse_args()

    df = generate(args.rows, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows to {args.out}, churn rate {df.churn.mean():.1%}")
