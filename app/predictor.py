from __future__ import annotations

from typing import Iterable, List

import pandas as pd
from fastapi import HTTPException, status

from . import config
from .schemas import ListingFeatures, PredictionResponse


def records_to_dataframe(records: Iterable[ListingFeatures]) -> pd.DataFrame:
    """Convert validated API payloads into the exact DataFrame expected by the model."""
    rows = [record.model_dump() for record in records]
    df = pd.DataFrame(rows)

    # TODO 1: reject unknown fields and forbidden leakage fields.
    forbidden_found = [c for c in df.columns if c in config.FORBIDDEN_FIELDS]
    if forbidden_found:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Forbidden fields in request.",
                "forbidden_fields": forbidden_found
            }
        )

    if "host_is_superhost" in df.columns:
        df["is_superhost"] = df["host_is_superhost"]

    if "has_reviews_before_cutoff" not in df.columns:
        df["has_reviews_before_cutoff"] = (
            df["total_reviews_before_cutoff"].fillna(0) > 0
        ).astype(int)


    # TODO 2: check missing fields against config.EXPECTED_FEATURE_COLUMNS.
    
    missing_cols = [c for c in config.EXPECTED_FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "TODO: missing feature handling", "missing_fields": missing_cols},
        )
    
    # TODO 3: return df[config.EXPECTED_FEATURE_COLUMNS].
    return df[config.EXPECTED_FEATURE_COLUMNS]


def predict_records(model, records: List[ListingFeatures]) -> List[PredictionResponse]:
    """TODO: Run model prediction and return API responses."""
    X = records_to_dataframe(records)

    # TODO:
    # - if model has predict_proba, use positive-class probability.
    # - apply config.PREDICTION_THRESHOLD.
    # - return one PredictionResponse per record.
    # Temporary placeholder so the endpoint shape is clear:

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
    else:
        prob = model.predict(X).astype(float)

    threshold = config.PREDICTION_THRESHOLD
    predictions = (prob >= threshold).astype(int)

    return [
        PredictionResponse(
            prediction=int(pred),
            prediction_label=config.POSITIVE_LABEL if pred == 1 else config.NEGATIVE_LABEL,
            probability=float(p),
            threshold=threshold,
        )
        for pred, p in zip(predictions, prob)
    ]
