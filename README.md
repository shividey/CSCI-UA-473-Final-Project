# NYC Housing Community Recommender

An interpretable, neighborhood-level housing recommendation system for New York City.

This repository implements a four-stage framework:

1. Data integration across affordable housing, subsidized housing, cost of living, poverty, and rent context.
2. PCA-based dimensionality reduction to summarize the main axes of community variation.
3. Community clustering with K-means to identify neighborhood archetypes.
4. Personalized recommendation with hard filters plus weighted ranking.

The project is designed for two use cases:

- User-facing housing search: given income, household structure, commute preferences, and budget, identify neighborhoods that best match the user's needs.
- Policy support: summarize neighborhood clusters and surface structural gaps that can guide housing development and affordability planning.

## Repository layout

```text
.
nyc_housing_app/
├── data/
│   ├── raw/                # Original CSVs: HPD Production, CoreData, AMI Tables.
│   └── processed/          # Merged and cleaned "Feature Matrix" ready for PCA.
├── models/
│   ├── scaler.pkl          # Saved StandardScaler to normalize new user inputs.
│   ├── pca_model.pkl       # Trained PCA transformer for dimensionality reduction.
│   └── kmeans_model.pkl    # Saved K-Means clusters for neighborhood categorization.
├── src/
│   ├── __init__.py         # Makes the src folder a Python package.
│   ├── integration.py      # Logic for merging HPD, CoreData, and AMI datasets via BBL/NTA.
│   ├── processing.py       # Scripts for standardization, PCA transformation, and de-noising.
│   ├── clustering.py       # K-Means implementation and archetype labeling (e.g., "Stability Anchor").
│   └── recommender.py      # The weighted ranking engine and hard-constraint filtering logic.
├── pages/
│   ├── 1.Explorer.py     # Streamlit page for visualizing NYC housing trends and maps.
│   └── 2.Housing_Recommander.py   # Streamlit page for the personalized recommendation interface.
├── app.py                  # The main entry point and "Home" page for the Streamlit dashboard.
├── requirements.txt        # List of dependencies (streamlit, scikit-learn, pandas, plotly).
└── README.md               # Documentation of the framework, data sources, and setup instructions.

```

## Quickstart

```bash
conda env create -f environment.yml
conda activate nyc-housing
pip install -r requirements.txt
PYTHONPATH=src python -m nyc_housing_recommender.cli recommend 
PYTHONPATH=src streamlit run streamlit_app.py
```

To inspect cluster summaries and policy signals:

```bash
PYTHONPATH=src python -m nyc_housing_recommender.cli summarize --use-sample-data
```

## Installation options

Recommended classroom-friendly setup:

```bash
conda env create -f environment.yml
conda activate nyc-housing
pip install -r requirements.txt
```

Alternative package-style setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Real data integration

Put cleaned CSV files into `data/raw/` and point the CLI to them with `--data-dir data/raw`.

Expected source files:

- `affordable_housing.csv`
- `subsidized_housing.csv`
- `living_cost.csv`
- `poverty.csv`
- `fair_market_rents.csv`
- `community_resources.csv`

The loader expects neighborhood-level records with a `borough` and `neighborhood` column. Additional schema details are documented in [`src/nyc_housing_recommender/data_loader.py`](/Users/zhaoyingru/Documents/New project/src/nyc_housing_recommender/data_loader.py).

If you start from raw open-data exports instead of already standardized files, run:

```bash
PYTHONPATH=src python3 -m nyc_housing_recommender.cli prepare-community-data --raw-dir data/raw --output-dir data/prepared
```

This command:

- reads the raw source files from `data/raw/`
- maps source-specific column names into the repository's standard schema
- coerces numeric fields
- aggregates records to `borough + community_name`
- writes pipeline-ready CSVs to `data/prepared/`
- writes a provenance report to `data/prepared/prepare_report.json`

## Download official source files

You can download the public source files used in this project with:

```bash
python3 scripts/download_open_data.py
```

This script currently fetches:

- NYC Open Data: Affordable Housing Production by Building
- NYC Open Data: `LC2016`
- NYC Open Data: NYCgov Poverty Measure Data (2016)
- HUD USER: FY 2026 county-level Fair Market Rents

It also writes a manifest file under `data/raw/` and leaves a note for `CoreData.nyc`, since a stable bulk download endpoint was not automatically confirmed.

## API-based ingestion

For the sources that expose an official API, you can fetch data directly instead of downloading files first:

```bash
PYTHONPATH=src python3 -m nyc_housing_recommender.cli fetch-api
```

Current API support:

- NYC Open Data via official Socrata API:
  - Affordable Housing Production by Building: [hg8x-zxpr](https://data.cityofnewyork.us/Housing-Development/Affordable-Housing-Production-by-Building/hg8x-zxpr)
  - LC2016: [etzh-883j](https://data.cityofnewyork.us/widgets/etzh-883j?mobile_redirect=true)
  - NYCgov Poverty Measure Data (2016): [tpt8-yikk](https://data.cityofnewyork.us/d/tpt8-yikk)
- HUD USER official FMR API:
  - docs: [FMR IL Dataset API Documentation](https://www.huduser.gov/portal/dataset/fmr-api.html)
  - base URL: `https://www.huduser.gov/hudapi/public/fmr`

Environment variables:

- `NYC_OPEN_DATA_APP_TOKEN`: optional Socrata app token for higher-rate access
- `HUD_USER_API_TOKEN`: required for HUD FMR API calls

At the moment:

- the repository can directly fetch the NYC Open Data tables through API
- HUD FMR is documented and token-ready in code, but the main pipeline still uses the file-based HUD input
- `CoreData.nyc` is still treated as a manual input because a stable official public API endpoint was not confirmed

## Official map boundaries

The app can now use official NYC community boundaries instead of the demo polygons.

Recommended source:

- 2020 Community District Tabulation Areas (CDTAs): [catalog metadata](https://catalog-beta.data.gov/dataset/2020-community-district-tabulation-areas-cdtas)

To download and convert the official boundary layer into app-ready GeoJSON:

```bash
python3 scripts/download_official_boundaries.py
```

This script writes:

- `data/raw/official_cdta_boundaries.geojson`

The Streamlit app will automatically prefer that file if it exists.

## Community-based unit

The repository now standardizes each source table into a shared `community_name` unit before integration. If a source only has `neighborhood`, that value is promoted into `community_name` so the sample pipeline still works. For real NYC data, the intended target geography is community district or another community-level geography that can be aligned across sources.

## Recommendation logic

The ranking module combines:

- Hard constraints: borough preference, max rent, minimum affordability, and optional cluster filtering.
- Weighted scoring over affordability, subsidized housing access, commute convenience, family fit, and neighborhood support.

This keeps recommendations transparent: every result includes a breakdown of the score and the cluster label it belongs to.

## Modeling approach

Model code now lives under [`src/nyc_housing_recommender/models/`](/Users/zhaoyingru/Documents/New project/src/nyc_housing_recommender/models).

The end-to-end method is:

1. Construct community-level housing, rent, hardship, and resource features.
2. Normalize the feature space.
3. Train a `PCA` model to compress correlated indicators into a smaller latent representation.
4. Train `K-means` on the PCA representation to identify community types.
5. Apply a transparent weighted ranking layer for user-specific recommendation.

Why PCA is included:

- it is a trainable model that captures the main structure of community variation
- it reduces redundancy across correlated affordability and support indicators
- it provides a 2D projection for visualization in the Streamlit interface
- it supports a cleaner clustering step before recommendation

## Policy analysis

The policy module produces cluster-level summaries and identifies neighborhoods with:

- high rent burden risk
- limited affordable housing supply
- low subsidized housing coverage
- weak access to community resources

These outputs can support development targeting, voucher strategy, and infrastructure planning.

## Streamlit interface

There was no UI in the earlier version of the repository; it only exposed a CLI. The project now includes a Streamlit interface in [`streamlit_app.py`](/Users/zhaoyingru/Documents/New project/streamlit_app.py).

Run it with:

```bash
PYTHONPATH=src streamlit run streamlit_app.py
```

You can also specify a data directory:

```bash
PYTHONPATH=src streamlit run streamlit_app.py -- --data-dir data/sample
```

Map support:

- The Streamlit app can render community polygons and highlight the top recommended communities.
- For the demo, the repository includes a sample boundary file at [`data/sample/community_boundaries.geojson`](/Users/zhaoyingru/Documents/New project/data/sample/community_boundaries.geojson).
- For real NYC deployment, use `data/raw/official_cdta_boundaries.geojson` and keep `borough` plus `community_name` aligned with the cleaned dataset.

## Data gap notes

A project-specific assessment of the remaining open-data gaps is included in [`docs/data_gap_analysis.md`](/Users/zhaoyingru/Documents/New project/docs/data_gap_analysis.md).

## Sources and processing

The implemented `prepare-community-data` step is designed around these sources:

- Affordable housing: [Affordable Housing Production by Building](https://data.cityofnewyork.us/Housing-Development/Affordable-Housing-Production-by-Building/hg8x-zxpr)
- Subsidized housing: [CoreData.nyc](https://app.coredata.nyc/) or an equivalent subsidized-housing export
- Cost of living: [`LC2016`](https://data.cityofnewyork.us/widgets/etzh-883j?mobile_redirect=true)
- Poverty: [NYCgov Poverty Measure Data (2016)](https://data.cityofnewyork.us/d/tpt8-yikk)
- Fair Market Rents: [HUD USER FMR API](https://www.huduser.gov/portal/dataset/fmr-api.html)
- Community geometry / geography reference: [2020 Community District Tabulation Areas](https://catalog-beta.data.gov/dataset/2020-community-district-tabulation-areas-cdtas)

The processing logic is implemented in [`src/nyc_housing_recommender/prepare.py`](/Users/zhaoyingru/Documents/New project/src/nyc_housing_recommender/prepare.py). For each source it:

- selects the best available geography and metric columns from known candidate names
- renames them into the repository's standard schema
- coerces strings like `$2,350` or `31%` into numeric values
- aggregates lower-level rows to community level
- outputs standardized tables ready for clustering and ranking
