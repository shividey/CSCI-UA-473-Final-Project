# NYC District Explorer

Streamlit app for exploring NYC community districts with two separate workflows:

- `PCA + clustering` for the opening webpage and cluster analysis
- `model-based retrieval` for personalized district recommendations

The two workflows are intentionally separated:

- the opening map uses saved PCA + KMeans artifacts
- the recommendation panel uses the scoring logic from `model/model.py`

## Main Files

```text
.
├── app.py
├── clustering_pipeline.py
├── retrieval_adapter.py
├── model_utils.py
├── export_artifacts.py
├── artifact_objects.py
├── artifacts/
├── data/
├── model/
├── requirements.txt
└── README.md
```

## What Each Part Does

### App

- `app.py`: Streamlit entrypoint
- `retrieval_adapter.py`: reproduces the retrieval logic from `model/model.py` for app use

### Opening Webpage

- `clustering_pipeline.py`: fits and exports the PCA + clustering artifacts used only for the opening webpage
- `model_utils.py`: loads the saved artifacts and cluster summaries for the app
- `artifact_objects.py`: lightweight array-based classes for the saved scaler, PCA, and KMeans artifacts
- `export_artifacts.py`: simple wrapper script that calls the clustering artifact export function
- `artifacts/`: saved models

### Data

The app currently runs from:

- `data/final_district_feature_table_latest_year_affordability_imputed.xlsx`
- `data/nyc_community_districts.geojson`

Important note:

- the current processed table used by the app is `final_district_feature_table_latest_year_affordability_imputed.xlsx`
- the other remaining source files in `data/` are raw inputs or preprocessing scripts


## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

## Rebuild Opening-Webpage Artifacts

The opening webpage uses the processed imputed table directly.

To regenerate the PCA + clustering artifacts:

```bash
python clustering_pipeline.py \
  --artifact-dir artifacts \
  --data-path data/final_district_feature_table_latest_year_affordability_imputed.xlsx \
  --geojson-path data/nyc_community_districts.geojson \
  --pca-components 5 \
  --clusters 4
```

This writes:

- `artifacts/processed_neighborhoods.csv`
- `artifacts/cluster_summary.csv`
- `artifacts/scaler.pkl`
- `artifacts/pca.pkl`
- `artifacts/kmeans_k4.pkl`
- `artifacts/feature_list.json`

## Retrieval Logic

The recommendation panel does **not** use PCA retrieval.

It uses the logic from `model/model.py`, including:

- winsorization
- category projection
- weighted preference scoring
- two-sided budget fit
- borough filtering

