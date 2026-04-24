from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize

from artifact_objects import ArrayKMeans, ArrayPCA, ArrayStandardScaler


ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"
DATA_DIR = ROOT / "data"
LATEST_DATA_PATH = DATA_DIR / "final_district_feature_table_latest_year_ml.csv"
RAW_DATA_PATH = DATA_DIR / "final_district_feature_table_latest_year_raw.csv"
DEFAULT_GEOJSON_PATH = DATA_DIR / "nyc_community_districts.geojson"
DEFAULT_IMPUTED_DATA_PATH = DATA_DIR / "final_district_feature_table_latest_year_affordability_imputed.xlsx"

RETRIEVAL_FEATURES = [
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

CATEGORY_FEATURES = {
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

CLUSTER_COLORS = {
    0: "#2A9D8F",
    1: "#E9C46A",
    2: "#457B9D",
    3: "#E76F51",
}

DEFAULT_BUDGET = 1800
DEFAULT_BEDROOMS = 2
DEFAULT_WINSOR_LIMIT = 0.017
DEFAULT_PCA_COMPONENTS = 5
DEFAULT_CLUSTER_COUNT = 4


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type for table input: {path}")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def build_feature_config() -> dict[str, Any]:
    priority_vectors = {}
    for key, values in PRIORITY_VECTORS.items():
        priority_vectors[key] = (values / values.sum()).tolist()

    return {
        "retrieval_features": RETRIEVAL_FEATURES,
        "model_feature_columns": [f"model_{feature}" for feature in RETRIEVAL_FEATURES],
        "category_features": CATEGORY_FEATURES,
        "priority_vectors": priority_vectors,
        "bedroom_map": {str(key): list(value) for key, value in BEDROOM_MAP.items()},
        "default_budget": DEFAULT_BUDGET,
        "default_bedrooms": DEFAULT_BEDROOMS,
        "availability_weight": 0.25,
        "cluster_colors": {str(key): value for key, value in CLUSTER_COLORS.items()},
    }


def resolve_kmeans_artifact(artifact_dir: str | Path = ARTIFACT_DIR) -> Path:
    artifact_dir = Path(artifact_dir)
    matches = sorted(artifact_dir.glob("kmeans_k*.pkl"))
    if not matches:
        raise FileNotFoundError(f"No KMeans artifact found in {artifact_dir}")
    return matches[0]


def borough_from_display(region_display: str) -> str | None:
    prefix = str(region_display).split()[0]
    return {
        "MN": "Manhattan",
        "BX": "Bronx",
        "BK": "Brooklyn",
        "QN": "Queens",
        "SI": "Staten Island",
    }.get(prefix)


def polygon_centroid(ring: list[list[float]]) -> tuple[float, float]:
    if len(ring) < 3:
        xs = [pt[0] for pt in ring]
        ys = [pt[1] for pt in ring]
        return (float(np.mean(xs)), float(np.mean(ys)))

    area = 0.0
    cx = 0.0
    cy = 0.0
    points = ring[:]
    if points[0] != points[-1]:
        points = points + [points[0]]

    for idx in range(len(points) - 1):
        x0, y0 = points[idx]
        x1, y1 = points[idx + 1]
        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross

    area *= 0.5
    if abs(area) < 1e-12:
        xs = [pt[0] for pt in ring]
        ys = [pt[1] for pt in ring]
        return (float(np.mean(xs)), float(np.mean(ys)))

    cx /= 6 * area
    cy /= 6 * area
    return (float(cx), float(cy))


def geometry_centroid(geometry: dict[str, Any]) -> tuple[float, float]:
    geom_type = geometry["type"]
    coords = geometry["coordinates"]

    if geom_type == "Polygon":
        return polygon_centroid(coords[0])

    if geom_type == "MultiPolygon":
        centroids: list[tuple[float, float]] = []
        weights: list[float] = []
        for polygon in coords:
            ring = polygon[0]
            centroid = polygon_centroid(ring)
            xs = [pt[0] for pt in ring]
            ys = [pt[1] for pt in ring]
            weight = max((max(xs) - min(xs)) * (max(ys) - min(ys)), 1e-9)
            centroids.append(centroid)
            weights.append(weight)

        lons = np.average([centroid[0] for centroid in centroids], weights=weights)
        lats = np.average([centroid[1] for centroid in centroids], weights=weights)
        return (float(lons), float(lats))

    raise ValueError(f"Unsupported geometry type: {geom_type}")


def attach_geometry(
    df: pd.DataFrame,
    artifact_dir: str | Path = ARTIFACT_DIR,
    geojson_path: str | Path | None = None,
) -> pd.DataFrame:
    artifact_dir = Path(artifact_dir)
    candidate_geojson = Path(geojson_path) if geojson_path else artifact_dir / "nyc_community_districts.geojson"
    if not candidate_geojson.exists():
        candidate_geojson = DEFAULT_GEOJSON_PATH

    geojson = json.loads(candidate_geojson.read_text())
    geo_rows = []
    for feature in geojson["features"]:
        region_id = int(feature["properties"]["boro_cd"])
        lon, lat = geometry_centroid(feature["geometry"])
        geo_rows.append(
            {
                "region_id": region_id,
                "boro_cd": region_id,
                "latitude": lat,
                "longitude": lon,
                "geometry_json": json.dumps(feature["geometry"]),
            }
        )

    geo_df = pd.DataFrame(geo_rows)
    merged = df.merge(geo_df, on="region_id", how="left")
    if "borough" not in merged.columns:
        merged["borough"] = merged["region_display"].map(borough_from_display)
    return merged


def load_latest_district_data(
    data_path: str | Path = DEFAULT_IMPUTED_DATA_PATH,
    raw_data_path: str | Path | None = RAW_DATA_PATH,
    artifact_dir: str | Path = ARTIFACT_DIR,
    geojson_path: str | Path | None = None,
) -> pd.DataFrame:
    base_df = read_table(data_path)
    if (
        "mta_station_count" not in base_df.columns
        and raw_data_path is not None
        and Path(raw_data_path).exists()
    ):
        raw_df = read_table(raw_data_path)[["region_id", "mta_station_count"]]
        base_df = base_df.merge(raw_df, on="region_id", how="left")

    base_df["borough"] = base_df["region_display"].map(borough_from_display)
    return attach_geometry(base_df, artifact_dir=artifact_dir, geojson_path=geojson_path)


def prepare_model_features(
    df: pd.DataFrame,
    feature_config: dict[str, Any],
    winsor_limit: float = 0.017,
) -> pd.DataFrame:
    retrieval_features = feature_config["retrieval_features"]
    model_feature_columns = feature_config.get(
        "model_feature_columns",
        [f"model_{feature}" for feature in retrieval_features],
    )

    work = df.copy()
    for feature_name, model_column in zip(retrieval_features, model_feature_columns):
        if feature_name not in work.columns:
            raise KeyError(f"Missing required feature column: {feature_name}")

        values = work[feature_name].astype(float)
        if feature_name in {"crime_all_rt", "crime_viol_rt"}:
            values = -values

        work[model_column] = np.asarray(
            winsorize(values, limits=[winsor_limit, winsor_limit]),
            dtype=float,
        )

    if "school_avg" not in work.columns and {"pct_prof_math", "pct_prof_ela"}.issubset(work.columns):
        work["school_avg"] = (work["pct_prof_math"] + work["pct_prof_ela"]) / 2

    return work


def apply_saved_pca_clustering(
    source_df: pd.DataFrame | None = None,
    artifact_dir: str | Path = ARTIFACT_DIR,
) -> pd.DataFrame:
    artifact_dir = Path(artifact_dir)
    feature_config = _load_json(artifact_dir / "feature_list.json")
    scaler = joblib.load(artifact_dir / "scaler.pkl")
    pca = joblib.load(artifact_dir / "pca.pkl")
    kmeans = joblib.load(resolve_kmeans_artifact(artifact_dir))

    base_df = source_df.copy() if source_df is not None else load_latest_district_data(artifact_dir=artifact_dir)
    work = prepare_model_features(base_df, feature_config=feature_config)

    model_feature_columns = feature_config.get(
        "model_feature_columns",
        [f"model_{feature}" for feature in feature_config["retrieval_features"]],
    )
    x = work[model_feature_columns].to_numpy(dtype=float)
    x_scaled = scaler.transform(x)
    x_pca = pca.transform(x_scaled)

    pca_mask = np.asarray(feature_config.get("pca_mask", [True] * x_pca.shape[1]), dtype=bool)
    x_reduced = x_pca[:, pca_mask]

    for idx in range(x_reduced.shape[1]):
        work[f"pca_{idx + 1}"] = x_reduced[:, idx]

    work["cluster_label"] = kmeans.predict(x_reduced).astype(int)

    cluster_type_map = {
        int(label): cluster_type
        for label, cluster_type in feature_config.get("cluster_type_map", {}).items()
    }
    work["cluster_type"] = work["cluster_label"].map(cluster_type_map)
    return work


def build_cluster_summary(
    df: pd.DataFrame,
    feature_config: dict[str, Any],
) -> pd.DataFrame:
    category_features = feature_config["category_features"]
    rows = []
    total_count = len(df)

    for cluster_label, group in df.groupby("cluster_label"):
        row: dict[str, Any] = {
            "cluster_label": int(cluster_label),
            "size": int(len(group)),
            "share_of_districts": round(len(group) / total_count, 3) if total_count else 0.0,
            "boroughs": ", ".join(sorted(group["borough"].dropna().unique().tolist())),
        }

        for category, features in category_features.items():
            model_columns = [f"model_{feature}" for feature in features if f"model_{feature}" in df.columns]
            if not model_columns:
                row[category] = 0.0
                continue

            reference = df[model_columns].mean(axis=1)
            cluster_score = group[model_columns].mean(axis=1).mean()
            denom = float(reference.std(ddof=0))
            row[category] = round(0.0 if denom <= 0 else (cluster_score - reference.mean()) / denom, 2)

        if "gross_rent_2_3beds_usd" in group.columns:
            row["gross_rent_2_3beds_usd"] = round(group["gross_rent_2_3beds_usd"].mean(), 1)
        if "rent_burden_sev_pct" in group.columns:
            row["rent_burden_sev_pct"] = round(group["rent_burden_sev_pct"].mean(), 1)
        if "hh_inc_med_adj_usd" in group.columns:
            row["hh_inc_med_adj_usd"] = round(group["hh_inc_med_adj_usd"].mean(), 1)
        if "cluster_type" in group.columns:
            modes = group["cluster_type"].dropna().mode()
            row["cluster_type"] = modes.iloc[0] if not modes.empty else f"Cluster {cluster_label}"

        rows.append(row)

    return pd.DataFrame(rows).sort_values("cluster_label").reset_index(drop=True)


def build_cluster_membership_table(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "region_name",
        "region_display",
        "borough",
        "cluster_label",
        "cluster_type",
        "gross_rent_2_3beds_usd",
        "school_avg",
        "mta_station_count",
    ]
    available_columns = [column for column in columns if column in df.columns]
    return (
        df[available_columns]
        .sort_values(["cluster_label", "borough", "region_name"])
        .reset_index(drop=True)
    )


def interpret_cluster(row: pd.Series) -> str:
    safety = row.get("Safety", 0.0)
    schools = row.get("Schools", 0.0)
    transit = row.get("Transit", 0.0)
    parks = row.get("Parks", 0.0)
    rent = row.get("gross_rent_2_3beds_usd", np.nan)

    if transit > 1.0 and schools > 0.8 and pd.notna(rent) and rent > 3000:
        return "elite transit-rich core"
    if safety < -0.5 and schools < -0.5:
        return "lower-resource districts"
    if safety > 0.4 and schools > 0.2 and abs(transit) < 0.5:
        return "stable middle-resource residential districts"
    if transit > 0.0 and abs(safety) < 0.4 and abs(schools) < 0.4:
        return "mixed middle districts"
    return "mixed middle districts"


def build_affordability_factor(
    df: pd.DataFrame,
    budget: float = DEFAULT_BUDGET,
    bedrooms: int = DEFAULT_BEDROOMS,
    availability_weight: float = 0.25,
) -> pd.Series:
    rent_col, share_col = BEDROOM_MAP[bedrooms]

    def rent_feasibility(district_rent: float) -> float:
        if pd.isna(district_rent):
            return 0.0
        if district_rent <= budget:
            return 1.0
        return float(1 / (1 + np.exp(-0.003 * (budget - district_rent))))

    rent_component = df[rent_col].apply(rent_feasibility)
    raw_share = df[share_col].fillna(0)
    share_max = raw_share.max()
    availability = raw_share / share_max if share_max and share_max > 0 else 0.0
    return rent_component * (1 - availability_weight + availability_weight * availability)


def fit_pca_clustering(
    source_df: pd.DataFrame | None = None,
    n_pca_components: int = DEFAULT_PCA_COMPONENTS,
    n_clusters: int = DEFAULT_CLUSTER_COUNT,
    winsor_limit: float = DEFAULT_WINSOR_LIMIT,
    random_state: int = 42,
    data_path: str | Path = LATEST_DATA_PATH,
    raw_data_path: str | Path = RAW_DATA_PATH,
    geojson_path: str | Path | None = None,
) -> dict[str, Any]:
    feature_config = build_feature_config()
    base_df = (
        source_df.copy()
        if source_df is not None
        else load_latest_district_data(
            data_path=data_path,
            raw_data_path=raw_data_path,
            geojson_path=geojson_path,
        )
    )
    work = prepare_model_features(base_df, feature_config=feature_config, winsor_limit=winsor_limit)

    model_feature_columns = feature_config["model_feature_columns"]
    x = work[model_feature_columns].to_numpy(dtype=float)

    scaler = ArrayStandardScaler.fit(x)
    x_scaled = scaler.transform(x)

    pca = ArrayPCA.fit(x_scaled, variance_threshold=0.90)
    x_pca = pca.transform(x_scaled)
    if x_pca.shape[1] < n_pca_components:
        raise ValueError(
            f"PCA only produced {x_pca.shape[1]} components, fewer than requested {n_pca_components}."
        )

    x_reduced = x_pca[:, :n_pca_components]
    kmeans = ArrayKMeans.fit(
        x_reduced,
        n_clusters=n_clusters,
        n_init=100,
        random_state=random_state,
    )

    for idx in range(n_pca_components):
        work[f"pca_{idx + 1}"] = x_reduced[:, idx]

    work["cluster_label"] = kmeans.labels_.astype(int)
    cluster_summary = build_cluster_summary(work, feature_config=feature_config)
    cluster_summary["cluster_type"] = cluster_summary.apply(interpret_cluster, axis=1)

    cluster_type_map = dict(
        zip(cluster_summary["cluster_label"].astype(int), cluster_summary["cluster_type"])
    )
    work["cluster_type"] = work["cluster_label"].map(cluster_type_map)
    work[f"affordability_factor_default_{DEFAULT_BUDGET}_{DEFAULT_BEDROOMS}br"] = build_affordability_factor(work)

    feature_config["pca_mask"] = [idx < n_pca_components for idx in range(pca.n_components_)]
    feature_config["component_weights"] = pca.explained_variance_ratio_[:n_pca_components].tolist()
    feature_config["cluster_type_map"] = {
        str(label): cluster_type for label, cluster_type in cluster_type_map.items()
    }
    feature_config["training_config"] = {
        "n_pca_components": n_pca_components,
        "n_clusters": n_clusters,
        "winsor_limit": winsor_limit,
        "random_state": random_state,
    }

    return {
        "processed_df": work,
        "cluster_summary": cluster_summary,
        "feature_config": feature_config,
        "scaler": scaler,
        "pca": pca,
        "kmeans": kmeans,
    }


def export_cluster_artifacts(
    artifact_dir: str | Path = ARTIFACT_DIR,
    source_df: pd.DataFrame | None = None,
    n_pca_components: int = DEFAULT_PCA_COMPONENTS,
    n_clusters: int = DEFAULT_CLUSTER_COUNT,
    data_path: str | Path = LATEST_DATA_PATH,
    raw_data_path: str | Path = RAW_DATA_PATH,
    geojson_path: str | Path | None = None,
) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    bundle = fit_pca_clustering(
        source_df=source_df,
        n_pca_components=n_pca_components,
        n_clusters=n_clusters,
        data_path=data_path,
        raw_data_path=raw_data_path,
        geojson_path=geojson_path,
    )
    processed_df = bundle["processed_df"]
    cluster_summary = bundle["cluster_summary"]
    feature_config = bundle["feature_config"]

    processed_df.to_csv(artifact_dir / "processed_neighborhoods.csv", index=False)
    cluster_summary.to_csv(artifact_dir / "cluster_summary.csv", index=False)
    joblib.dump(bundle["scaler"], artifact_dir / "scaler.pkl")
    joblib.dump(bundle["pca"], artifact_dir / "pca.pkl")
    joblib.dump(bundle["kmeans"], artifact_dir / f"kmeans_k{n_clusters}.pkl")
    (artifact_dir / "feature_list.json").write_text(json.dumps(feature_config, indent=2))

    geojson_target = artifact_dir / "nyc_community_districts.geojson"
    if not geojson_target.exists() and DEFAULT_GEOJSON_PATH.exists():
        geojson_target.write_text(DEFAULT_GEOJSON_PATH.read_text())

    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit and export the PCA + clustering artifacts for the NYC district explorer."
    )
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--pca-components", type=int, default=DEFAULT_PCA_COMPONENTS)
    parser.add_argument("--clusters", type=int, default=DEFAULT_CLUSTER_COUNT)
    parser.add_argument("--data-path", type=Path, default=LATEST_DATA_PATH)
    parser.add_argument("--raw-data-path", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--geojson-path", type=Path, default=DEFAULT_GEOJSON_PATH)
    args = parser.parse_args()

    bundle = export_cluster_artifacts(
        artifact_dir=args.artifact_dir,
        n_pca_components=args.pca_components,
        n_clusters=args.clusters,
        data_path=args.data_path,
        raw_data_path=args.raw_data_path,
        geojson_path=args.geojson_path,
    )
    print(
        f"Saved PCA + clustering artifacts to {args.artifact_dir} "
        f"for {len(bundle['processed_df'])} districts "
        f"using {args.pca_components} PCA components and {args.clusters} clusters."
    )


if __name__ == "__main__":
    main()
