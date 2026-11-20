.PHONY: install data train test serve docker

export PYTHONPATH := src
export MLFLOW_TRACKING_URI ?= sqlite:///mlflow.db

install:
	pip install -r requirements-train.txt

data:
	python data/generate_data.py --rows 20000

train: data
	python -m churn.train --model gbm --min-auc 0.75

test:
	pytest -q

serve:
	PYTHONPATH=.:src uvicorn serving.app:app --reload

ui:
	mlflow ui --backend-store-uri $(MLFLOW_TRACKING_URI)

docker: train
	docker build -t churn-api:local .
