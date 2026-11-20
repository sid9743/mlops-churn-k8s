import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def trained_artifacts(tmp_path_factory):
    """Generate a small dataset and train once for the whole test session."""
    work = tmp_path_factory.mktemp("run")
    data = work / "churn.csv"
    out = work / "artifacts"
    env_py = sys.executable
    subprocess.run([env_py, "data/generate_data.py", "--rows", "4000", "--out", str(data)], cwd=ROOT, check=True)
    subprocess.run(
        [env_py, "-m", "churn.train", "--data", str(data), "--out", str(out), "--model", "logreg"],
        cwd=ROOT,
        check=True,
        env={"PYTHONPATH": "src", "MLFLOW_TRACKING_URI": f"sqlite:///{work}/mlflow.db", "MLFLOW_DISABLE_AGENT_HINT": "1",
             "PATH": "/usr/bin:/bin"},
    )
    return out
