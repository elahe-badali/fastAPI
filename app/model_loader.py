from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# TODO: import os, mlflow, mlflow.sklearn, MlflowClient when implementing.
import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from . import config


@dataclass
class LoadedModelState:
    model: Any = None
    loaded: bool = False
    error: Optional[str] = None
    model_uri: Optional[str] = None
    run_id: Optional[str] = None
    run_name: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, Any] = field(default_factory=dict)


class ModelService:
    """TODO: Load the HW02 model from MLflow.

    Required behavior:
    - Use MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, and MLFLOW_TRACKING_PASSWORD.
    - If MLFLOW_RUN_ID is set, load runs:/<run_id>/model.
    - Otherwise auto-select a clean/selected run from MLFLOW_EXPERIMENT_NAME.
    - Do not crash the API on startup. Store the error in self.state.error.
    """

    def __init__(self) -> None:
        self.state = LoadedModelState()

    def load(self) -> None:
        # TODO: Replace this placeholder with real MLflow loading.
        # Hint:
        #   os.environ["MLFLOW_TRACKING_USERNAME"] = config.MLFLOW_TRACKING_USERNAME
        #   os.environ["MLFLOW_TRACKING_PASSWORD"] = config.MLFLOW_TRACKING_PASSWORD
        #   mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        #   model = mlflow.sklearn.load_model(f"runs:/{config.MLFLOW_RUN_ID}/model")

        try:

            self.state.loaded = False
            self.state.error = "TODO: load model from MLflow."
            os.environ["MLFLOW_TRACKING_USERNAME"] = config.MLFLOW_TRACKING_USERNAME
            os.environ["MLFLOW_TRACKING_PASSWORD"] = config.MLFLOW_TRACKING_PASSWORD
            mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
            client = MlflowClient()
            run_id = getattr(config, "MLFLOW_RUN_ID", None)


            if not run_id:
                experiment = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
                if experiment is None:
                    raise RuntimeError(f"Experiment not found: {config.MLFLOW_EXPERIMENT_NAME}")

                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string="tags.selected_for_serving = 'true' AND tags.leakage_status = 'clean'",
                    order_by=["metrics.f1 DESC"],
                    max_results=1
                )
                if not runs:
                    runs = client.search_runs(
                        experiment_ids=[experiment.experiment_id],
                        filter_string="tags.leakage_status = 'clean'",
                        order_by=["metrics.f1 DESC"],
                        max_results=1
                    )
                if not runs:
                    raise RuntimeError("No clean MLflow run found.")
                run_id = runs[0].info.run_id

            model_uri_candidates = [
                f"runs:/{run_id}/sklearn_pipeline_model",
                f"runs:/{run_id}/model"
            ]

            model = None
            last_error = None
            for uri in model_uri_candidates:
                try:
                    model = mlflow.sklearn.load_model(uri)
                    self.state.model_uri = uri
                    break
                except Exception as e:
                    last_error = e

            if model is None:
                raise RuntimeError(f"Could not load model from MLflow. Last error: {last_error}")

            run = client.get_run(run_id)
            self.state.model = model
            self.state.loaded = True
            self.state.error = None
            self.state.run_id = run_id
            self.state.run_name = run.data.tags.get("mlflow.runName")
            self.state.metrics = dict(run.data.metrics)
            self.state.params = dict(run.data.params)
            self.state.tags = dict(run.data.tags)

        except Exception as e:
            self.state.model = None
            self.state.loaded = False
            self.state.error = str(e)

    def require_model(self):
        if not self.state.loaded or self.state.model is None:
            raise RuntimeError(self.state.error or "Model is not loaded.")
        return self.state.model

    def model_info(self) -> dict:
        return {
            "model_loaded": self.state.loaded,
            "tracking_uri": config.MLFLOW_TRACKING_URI,
            "experiment_name": config.MLFLOW_EXPERIMENT_NAME,
            "model_uri": self.state.model_uri,
            "run_id": self.state.run_id,
            "run_name": self.state.run_name,
            "dataset_version": config.DATASET_VERSION,
            "target": config.TARGET_NAME,
            "threshold": config.PREDICTION_THRESHOLD,
            "metrics": self.state.metrics,
            "params": self.state.params,
            "tags": self.state.tags,
            "error": self.state.error,
        }
