"""Train the churn model, log everything to MLflow and enforce a quality gate.

Exit code is non-zero when the model does not meet the minimum ROC AUC, which stops
the CI pipeline before a weak model is packaged and deployed.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from churn.features import FEATURES, TARGET, build_preprocessor

MODELS = {
    "logreg": lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
    "gbm": lambda: GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
}


def train(data_path: str, model_name: str, out_dir: str, min_auc: float) -> dict:
    df = pd.read_csv(data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, stratify=df[TARGET], random_state=42
    )

    pipeline = Pipeline([("prep", build_preprocessor()), ("model", MODELS[model_name]())])

    mlflow.set_experiment("churn-prediction")
    with mlflow.start_run(run_name=model_name) as run:
        pipeline.fit(x_train, y_train)
        proba = pipeline.predict_proba(x_test)[:, 1]
        preds = (proba >= 0.5).astype(int)

        metrics = {
            "roc_auc": round(roc_auc_score(y_test, proba), 4),
            "f1": round(f1_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds), 4),
            "recall": round(recall_score(y_test, preds), 4),
        }
        mlflow.log_params({"model": model_name, "rows": len(df), "features": len(FEATURES)})
        mlflow.log_metrics(metrics)
        # The tree type is trusted because this pipeline built the model itself
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            input_example=x_test.head(3),
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        # Reference distribution for drift monitoring in the serving layer
        edges = np.quantile(df["monthly_charges"], np.linspace(0, 1, 11)).tolist()
        reference = {"monthly_charges_bin_edges": edges}

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, out / "model.joblib")
        (out / "metrics.json").write_text(json.dumps({**metrics, "run_id": run.info.run_id}, indent=2))
        (out / "reference.json").write_text(json.dumps(reference))

    print(json.dumps(metrics, indent=2))
    if metrics["roc_auc"] < min_auc:
        print(f"Quality gate FAILED: roc_auc {metrics['roc_auc']} < {min_auc}", file=sys.stderr)
        sys.exit(1)
    print("Quality gate passed")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/churn.csv")
    parser.add_argument("--model", choices=MODELS, default="gbm")
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--min-auc", type=float, default=0.75)
    args = parser.parse_args()
    train(args.data, args.model, args.out, args.min_auc)
