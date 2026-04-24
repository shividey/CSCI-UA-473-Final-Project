from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.linalg import qr
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler

from clustering_pipeline import DEFAULT_IMPUTED_DATA_PATH, load_latest_district_data


ROOT = Path(__file__).resolve().parent
DATA_PATH = DEFAULT_IMPUTED_DATA_PATH
RAW_DATA_PATH = ROOT / "data" / "final_district_feature_table_latest_year_raw.csv"
GEOJSON_PATH = ROOT / "data" / "nyc_community_districts.geojson"

FEATURES = [
    "crime_all_rt",
    "crime_viol_rt",
    "pct_prof_math",
    "pct_prof_ela",
    "mta_station_count",
    "mta_route_count_log1p",
    "mta_station_density_per_sq_km_log1p",
    "parks_total_acres_log1p",
    "parks_neighborhood_park_count",
    "parks_playground_count",
    "parks_acres_per_sq_km_log1p",
]

CATEGORIES = ["Safety", "Schools", "Transit", "Parks"]

PRIORITY_VECTORS = {
    "Safety": np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float),
    "Schools": np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=float),
    "Transit": np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0], dtype=float),
    "Parks": np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=float),
}

BEDROOM_MAP = {
    0: ("gross_rent_0_1beds_usd", "ah_studio_share"),
    1: ("gross_rent_0_1beds_usd", "ah_1br_share"),
    2: ("gross_rent_2_3beds_usd", "ah_2br_share"),
    3: ("gross_rent_2_3beds_usd", "ah_3br_share"),
}

VALID_BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]


def _normalize_priority_vectors() -> dict[str, np.ndarray]:
    return {
        key: values / values.sum()
        for key, values in PRIORITY_VECTORS.items()
    }


def _load_retrieval_dataframe() -> pd.DataFrame:
    return load_latest_district_data(
        data_path=DATA_PATH,
        raw_data_path=RAW_DATA_PATH,
        geojson_path=GEOJSON_PATH,
    )


def _budget_fit(
    district_rent: float,
    budget: float,
    over_steepness: float = 0.004,
    over_center_pct: float = 0.05,
    under_steepness: float = 0.001,
    under_center_pct: float = 0.50,
) -> float:
    if pd.isna(district_rent):
        return np.nan

    if district_rent > budget:
        center = budget * over_center_pct
        return float(1 / (1 + np.exp(over_steepness * (district_rent - budget - center))))

    center = budget * under_center_pct
    return float(1 / (1 + np.exp(under_steepness * (budget - district_rent - center))))


def _build_explanation(
    row: pd.Series,
    ordered_categories: list[str],
    total_count: int,
    user_budget: float,
) -> str:
    best_categories = sorted(ordered_categories, key=lambda category: row[f"{category.lower()}_rank"])[:2]
    strengths_text = f"Strongest on {best_categories[0].lower()} and {best_categories[1].lower()}."

    if pd.notna(row["selected_rent"]):
        if row["selected_rent"] > user_budget:
            budget_note = "over budget"
        elif row["selected_rent"] >= 0.8 * user_budget:
            budget_note = "near target budget"
        else:
            budget_note = "well below target budget"
        rent_text = f"Rent: ${row['selected_rent']:,.0f}/mo ({budget_note})"
    else:
        rent_text = "Rent unknown"

    ranking_text = " | ".join(
        f"{category}: #{int(row[f'{category.lower()}_rank'])} of {total_count}"
        for category in ordered_categories
    )
    return f"{strengths_text} {rent_text}. {ranking_text}"


def retrieve_with_model_logic(
    safety: float = 3.0,
    schools: float = 3.0,
    transit: float = 3.0,
    parks: float = 3.0,
    top_k: int = 5,
    budget: float = 1800,
    bedrooms: int = 2,
    boroughs: list[str] | None = None,
    cluster_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    boroughs = boroughs or None
    if boroughs is not None:
        invalid = [borough for borough in boroughs if borough not in VALID_BOROUGHS]
        if invalid:
            raise ValueError(f"Invalid borough filter: {invalid}")

    df = _load_retrieval_dataframe()
    work = df.copy()
    work["crime_all_rt"] = -work["crime_all_rt"]
    work["crime_viol_rt"] = -work["crime_viol_rt"]

    for column in FEATURES:
        work[column] = np.asarray(winsorize(work[column], limits=[0.017, 0.017]), dtype=float)

    work["school_avg"] = (work["pct_prof_math"] + work["pct_prof_ela"]) / 2

    scaler = StandardScaler()
    x = work[FEATURES].to_numpy(dtype=float)
    x_scaled = scaler.fit_transform(x)

    priority_vectors = _normalize_priority_vectors()
    basis = np.array([priority_vectors[category] for category in CATEGORIES])
    q_matrix, _ = qr(basis.T)
    q_matrix = q_matrix[:, : len(CATEGORIES)].T
    for idx, category in enumerate(CATEGORIES):
        if np.dot(q_matrix[idx], basis[idx]) < 0:
            q_matrix[idx] *= -1

    x_proj = x_scaled @ q_matrix.T
    category_weights = np.array([safety, schools, transit, parks], dtype=float)
    if category_weights.sum() <= 0:
        category_weights = np.ones_like(category_weights)
    category_weights /= category_weights.sum()

    rent_col, share_col = BEDROOM_MAP[bedrooms]
    raw_share = work[share_col].fillna(0).clip(0, 1)
    work["budget_fit"] = work[rent_col].apply(lambda rent: _budget_fit(rent, budget))
    work["unit_availability"] = np.sqrt(raw_share)
    work["price_fit_factor"] = work["budget_fit"] * (0.75 + 0.25 * work["unit_availability"])

    preference_scores = (x_proj * category_weights).sum(axis=1)
    work["preference_score"] = np.clip(preference_scores, 0, None)
    work["final_score"] = preference_scores * work["price_fit_factor"]
    work["selected_rent"] = work[rent_col]

    category_score_df = pd.DataFrame(x_proj, columns=CATEGORIES, index=work.index)
    category_rank_df = category_score_df.rank(ascending=False, method="min").astype(int)
    for category in CATEGORIES:
        work[f"{category.lower()}_rank"] = category_rank_df[category].values

    if boroughs is not None:
        work = work[work["borough"].isin(boroughs)].copy()

    if cluster_df is not None:
        cluster_columns = [
            "region_id",
            "cluster_label",
            "cluster_type",
            "latitude",
            "longitude",
            "geometry_json",
        ]
        available_cluster_columns = [column for column in cluster_columns if column in cluster_df.columns]
        work = work.merge(
            cluster_df[available_cluster_columns].drop_duplicates("region_id"),
            on="region_id",
            how="left",
            suffixes=("", "_cluster"),
        )

    ordered_categories = sorted(
        CATEGORIES,
        key=lambda category: {"Safety": safety, "Schools": schools, "Transit": transit, "Parks": parks}[category],
        reverse=True,
    )
    total_count = len(df)
    work["explanation"] = work.apply(
        lambda row: _build_explanation(row, ordered_categories, total_count, budget),
        axis=1,
    )

    work = work.sort_values("final_score", ascending=False).reset_index(drop=True)
    work["rank"] = np.arange(1, len(work) + 1)

    result_columns = [
        "rank",
        "region_id",
        "region_name",
        "region_display",
        "borough",
        "cluster_label",
        "cluster_type",
        "preference_score",
        "price_fit_factor",
        "final_score",
        "selected_rent",
        "school_avg",
        "crime_viol_rt",
        "mta_station_count",
        "parks_neighborhood_park_count",
        "latitude",
        "longitude",
        "geometry_json",
        "explanation",
        "safety_rank",
        "schools_rank",
        "transit_rank",
        "parks_rank",
    ]
    available_columns = [column for column in result_columns if column in work.columns]
    return work[available_columns].head(top_k).copy()
