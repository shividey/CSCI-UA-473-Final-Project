from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from clustering_pipeline import (
    apply_saved_pca_clustering,
    build_cluster_summary as build_runtime_cluster_summary,
    load_latest_district_data,
    resolve_kmeans_artifact,
)


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_artifacts(artifact_dir: str | Path = ARTIFACT_DIR) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    feature_config = _load_json(artifact_dir / "feature_list.json")
    cluster_summary_path = artifact_dir / "cluster_summary.csv"
    cluster_summary = (
        pd.read_csv(cluster_summary_path) if cluster_summary_path.exists() else pd.DataFrame()
    )
    try:
        latest_df = load_latest_district_data(artifact_dir=artifact_dir)
        processed_df = apply_saved_pca_clustering(latest_df, artifact_dir=artifact_dir)
    except FileNotFoundError:
        processed_df = pd.read_csv(artifact_dir / "processed_neighborhoods.csv")

    if cluster_summary.empty:
        cluster_summary = build_runtime_cluster_summary(processed_df, feature_config=feature_config)

    return {
        "processed_df": processed_df,
        "scaler": joblib.load(artifact_dir / "scaler.pkl"),
        "pca": joblib.load(artifact_dir / "pca.pkl"),
        "kmeans": joblib.load(resolve_kmeans_artifact(artifact_dir)),
        "feature_config": feature_config,
        "cluster_summary": cluster_summary,
    }


def get_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    feature_config = {
        "category_features": {
            "Safety": ["crime_all_rt", "crime_viol_rt"],
            "Schools": ["pct_prof_math", "pct_prof_ela"],
            "Transit": [
                "mta_station_count",
                "mta_route_count_log1p",
                "mta_station_density_per_sq_km_log1p",
            ],
            "Parks": [
                "parks_total_acres_log1p",
                "parks_neighborhood_park_count",
                "parks_playground_count",
                "parks_acres_per_sq_km_log1p",
            ],
        }
    }
    summary = build_runtime_cluster_summary(df, feature_config=feature_config)
    for expected_col in ["median_2_3br_rent", "avg_school_score", "avg_violent_crime_rate", "avg_transit_count", "avg_parks_count"]:
        if expected_col not in summary.columns:
            summary[expected_col] = np.nan
    return summary
