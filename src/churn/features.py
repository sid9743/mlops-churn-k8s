"""Feature definitions shared by training and serving, so both always agree."""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["tenure_months", "monthly_charges", "total_charges", "support_calls", "senior_citizen"]
CATEGORICAL = ["contract", "internet_service", "payment_method"]
TARGET = "churn"
FEATURES = NUMERIC + CATEGORICAL


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
