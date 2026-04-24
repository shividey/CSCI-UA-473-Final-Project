from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

try:
    import folium
    from streamlit_folium import st_folium

    MAP_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover
    folium = None
    st_folium = None
    MAP_IMPORT_ERROR = exc

from clustering_pipeline import build_cluster_membership_table
from model_utils import load_artifacts
from retrieval_adapter import VALID_BOROUGHS, retrieve_with_model_logic


st.set_page_config(page_title="NYC District Explorer", layout="wide")

if MAP_IMPORT_ERROR is not None:
    st.title("NYC District Explorer")
    st.error(
        "Map dependencies are missing in this environment. Install the packages from "
        "`requirements.txt` before running the Streamlit app."
    )
    st.code("pip install -r requirements.txt")
    st.stop()


@st.cache_resource
def load_bundle() -> dict:
    return load_artifacts()


@st.cache_data
def load_processed_df() -> pd.DataFrame:
    return load_bundle()["processed_df"].copy()


@st.cache_data
def load_cluster_table() -> pd.DataFrame:
    return build_cluster_membership_table(load_processed_df())


@st.cache_data
def build_pca_variance_table(explained_variance_ratio: tuple[float, ...]) -> pd.DataFrame:
    ratios = list(explained_variance_ratio)
    cumulative = pd.Series(ratios).cumsum()
    return pd.DataFrame(
        {
            "PC": [f"PC{i + 1}" for i in range(len(ratios))],
            "explained_variance_ratio": ratios,
            "cumulative_explained_variance": cumulative,
        }
    )


def make_feature_collection(df: pd.DataFrame) -> dict:
    features = []
    for _, row in df.iterrows():
        geometry_value = row.get("geometry_json")
        if pd.isna(geometry_value):
            continue
        properties = {
            "region_name": row.get("region_name"),
            "borough": row.get("borough"),
            "cluster_label": int(row.get("cluster_label", -1)),
            "cluster_type": row.get("cluster_type"),
        }
        features.append(
            {
                "type": "Feature",
                "geometry": json.loads(geometry_value),
                "properties": properties,
            }
        )
    return {"type": "FeatureCollection", "features": features}


def build_cluster_map(
    base_df: pd.DataFrame,
    cluster_colors: dict[str, str],
    result_df: pd.DataFrame | None = None,
) -> folium.Map:
    center_lat = float(base_df["latitude"].dropna().mean()) if "latitude" in base_df else 40.7128
    center_lon = float(base_df["longitude"].dropna().mean()) if "longitude" in base_df else -74.0060
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="CartoDB positron")

    if "geometry_json" in base_df.columns and base_df["geometry_json"].notna().any():
        feature_collection = make_feature_collection(base_df)

        def style_function(feature: dict) -> dict:
            cluster_label = str(feature["properties"].get("cluster_label", -1))
            return {
                "fillColor": cluster_colors.get(cluster_label, "#8d99ae"),
                "color": "#1f2937",
                "weight": 1,
                "fillOpacity": 0.52,
            }

        tooltip = folium.GeoJsonTooltip(
            fields=["region_name", "borough", "cluster_type"],
            aliases=["District", "Borough", "Cluster label"],
            sticky=False,
        )

        folium.GeoJson(
            feature_collection,
            style_function=style_function,
            tooltip=tooltip,
            name="District clusters",
        ).add_to(fmap)

    if result_df is not None and not result_df.empty:
        for _, row in result_df.dropna(subset=["latitude", "longitude"]).iterrows():
            popup = (
                f"<b>#{int(row['rank'])} {row['region_name']}</b><br>"
                f"Cluster: {row['cluster_type']}<br>"
                f"Final score: {row['final_score']:.3f}<br>"
                f"Preference: {row['preference_score']:.3f}<br>"
                f"Price fit: {row['price_fit_factor']:.3f}"
            )
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=9,
                fill=True,
                color="#111827",
                weight=2,
                fill_color="#d62828",
                fill_opacity=0.95,
                tooltip=f"#{int(row['rank'])} {row['region_name']}",
                popup=popup,
            ).add_to(fmap)

    folium.LayerControl(collapsed=True).add_to(fmap)
    return fmap


def render_results_panel(results_df: pd.DataFrame | None) -> None:
    st.subheader("Retrieval Results")
    if results_df is None:
        st.info("Use the sidebar to run retrieval. Results will appear here next to the map.")
        return

    top_row = results_df.iloc[0]
    st.metric("Top Match", top_row["region_name"], delta=f"Score {top_row['final_score']:.3f}")

    for _, row in results_df.iterrows():
        with st.container(border=True):
            st.markdown(
                f"**#{int(row['rank'])} {row['region_name']}**  \n"
                f"{row.get('borough', 'Unknown borough')} · {row.get('cluster_type', 'Unknown cluster')}"
            )
            score_col, sim_col, aff_col = st.columns(3)
            with score_col:
                st.metric("Final", f"{row['final_score']:.3f}")
            with sim_col:
                st.metric("Preference", f"{row['preference_score']:.3f}")
            with aff_col:
                st.metric("Price Fit", f"{row['price_fit_factor']:.3f}")

            details = []
            if "selected_rent" in row and pd.notna(row["selected_rent"]):
                details.append(f"Rent ${row['selected_rent']:,.0f}")
            if "school_avg" in row and pd.notna(row["school_avg"]):
                details.append(f"Schools {row['school_avg']:.1f}")
            if "mta_station_count" in row and pd.notna(row["mta_station_count"]):
                details.append(f"Transit {int(row['mta_station_count'])} stations")
            if details:
                st.caption(" | ".join(details))

            st.write(row.get("explanation", ""))

    with st.expander("See ranking table", expanded=False):
        ranking_columns = [
            "rank",
            "region_name",
            "borough",
            "cluster_type",
            "final_score",
            "preference_score",
            "price_fit_factor",
            "selected_rent",
        ]
        available_columns = [column for column in ranking_columns if column in results_df.columns]
        st.dataframe(results_df[available_columns], use_container_width=True, hide_index=True)


def render_pca_variance_plot(variance_df: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    x = range(1, len(variance_df) + 1)
    ax1.bar(x, variance_df["explained_variance_ratio"], color="#457B9D", alpha=0.85)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_xticks(list(x))

    ax2 = ax1.twinx()
    ax2.plot(
        list(x),
        variance_df["cumulative_explained_variance"],
        color="#E76F51",
        marker="o",
        linewidth=2,
    )
    ax2.set_ylabel("Cumulative Explained Variance")
    ax2.set_ylim(0, 1.05)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_cluster_scatter_2d(processed_df: pd.DataFrame, cluster_colors: dict[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for cluster_label, group in processed_df.groupby("cluster_label"):
        color = cluster_colors.get(str(int(cluster_label)), "#8d99ae")
        ax.scatter(
            group["pca_1"],
            group["pca_2"],
            s=60,
            alpha=0.85,
            label=f"Cluster {int(cluster_label)}",
            color=color,
            edgecolor="#1f2937",
            linewidth=0.5,
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Clusters in PCA Space (PC1 vs PC2)")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_cluster_scatter_3d(processed_df: pd.DataFrame, cluster_colors: dict[str, str]) -> None:
    fig = plt.figure(figsize=(8.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    for cluster_label, group in processed_df.groupby("cluster_label"):
        color = cluster_colors.get(str(int(cluster_label)), "#8d99ae")
        ax.scatter(
            group["pca_1"],
            group["pca_2"],
            group["pca_3"],
            s=50,
            alpha=0.9,
            label=f"Cluster {int(cluster_label)}",
            color=color,
            edgecolors="#1f2937",
            linewidths=0.4,
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("3D PCA Clustering View")
    ax.view_init(elev=22, azim=38)
    ax.legend(loc="upper left")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


bundle = load_bundle()
processed_df = load_processed_df()
membership_df = load_cluster_table()
cluster_summary_df = bundle["cluster_summary"].copy().sort_values("cluster_label").reset_index(drop=True)
cluster_colors = bundle["feature_config"].get("cluster_colors", {})
pca_dimensions = int(sum(bundle["feature_config"].get("pca_mask", [])))
cluster_count = int(processed_df["cluster_label"].nunique())
variance_df = build_pca_variance_table(tuple(bundle["pca"].explained_variance_ratio_))

st.title("NYC District Explorer")
st.write(
    "The first page focuses on the map and retrieval workflow. Cluster interpretation and PCA evaluation "
    "live on the second page."
)

top_metrics = st.columns(3)
top_metrics[0].metric("Districts grouped", len(processed_df))
top_metrics[1].metric("PCA dimensions used", pca_dimensions)
top_metrics[2].metric("Cluster labels", cluster_count)

with st.sidebar:
    st.header("Neighborhood Retrieval")
    st.caption("The saved artifacts use 5 PCA dimensions and 4 clusters.")
    safety = st.slider("Safety priority", 0, 5, 5)
    schools = st.slider("Schools priority", 0, 5, 4)
    transit = st.slider("Transit priority", 0, 5, 3)
    parks = st.slider("Parks priority", 0, 5, 2)
    budget = st.slider("Monthly rent budget ($)", 1000, 6000, 1800, step=100)
    bedroom_label = st.selectbox("Bedroom type", ["Studio", "1BR", "2BR", "3BR"], index=2)
    boroughs = st.multiselect("Borough filter", VALID_BOROUGHS, default=[])
    top_k = st.slider("Top matches", 3, 10, 5)
    run_search = st.button("Run retrieval", type="primary")

bedroom_lookup = {"Studio": 0, "1BR": 1, "2BR": 2, "3BR": 3}

if "results_df" not in st.session_state:
    st.session_state.results_df = None

if run_search:
    st.session_state.results_df = retrieve_with_model_logic(
        safety=safety,
        schools=schools,
        transit=transit,
        parks=parks,
        top_k=top_k,
        budget=budget,
        bedrooms=bedroom_lookup[bedroom_label],
        boroughs=boroughs or None,
        cluster_df=processed_df,
    )

results_df = st.session_state.results_df

legend_html = "<div style='display:flex;gap:10px;flex-wrap:wrap;margin:0 0 12px 0;'>"
for cluster_id, color in sorted(cluster_colors.items(), key=lambda item: int(item[0])):
    label_match = cluster_summary_df.loc[cluster_summary_df["cluster_label"] == int(cluster_id), "cluster_type"]
    label_text = label_match.iloc[0] if not label_match.empty else f"Cluster {cluster_id}"
    legend_html += (
        "<div style='display:flex;align-items:center;gap:6px;'>"
        f"<span style='display:inline-block;width:14px;height:14px;background:{color};border:1px solid #111827;'></span>"
        f"{label_text}</div>"
    )
legend_html += "</div>"
st.markdown(legend_html, unsafe_allow_html=True)

opening_tab, cluster_tab = st.tabs(["Opening Webpage", "Cluster Analysis"])

with opening_tab:
    map_col, result_col = st.columns([1.7, 1], gap="large")

    with map_col:
        st.subheader("Clustered District Map")
        cluster_map = build_cluster_map(processed_df, cluster_colors, result_df=results_df)
        st_folium(cluster_map, use_container_width=True, height=760)

    with result_col:
        render_results_panel(results_df)

with cluster_tab:
    st.subheader("Cluster Explanations")
    for _, row in cluster_summary_df.iterrows():
        with st.container(border=True):
            st.markdown(f"**Cluster {int(row['cluster_label'])}: {row['cluster_type']}**")
            st.caption(
                f"{int(row['size'])} districts · "
                f"share {row.get('share_of_districts', 0):.1%} · "
                f"avg 2-3BR rent ${row.get('gross_rent_2_3beds_usd', 0):,.0f}"
            )
            st.write(
                f"Safety {row.get('Safety', 0):.2f} | "
                f"Schools {row.get('Schools', 0):.2f} | "
                f"Transit {row.get('Transit', 0):.2f} | "
                f"Parks {row.get('Parks', 0):.2f}"
            )

    st.subheader("District Grouping Table")
    cluster_filter_options = ["All clusters"] + [
        f"Cluster {int(row['cluster_label'])}: {row['cluster_type']}"
        for _, row in cluster_summary_df.iterrows()
    ]
    selected_filter = st.selectbox("Browse by cluster", cluster_filter_options)
    if selected_filter == "All clusters":
        filtered_membership_df = membership_df
    else:
        selected_cluster = int(selected_filter.split(":")[0].replace("Cluster", "").strip())
        filtered_membership_df = membership_df[membership_df["cluster_label"] == selected_cluster]
    st.dataframe(filtered_membership_df, use_container_width=True, hide_index=True)

    st.subheader("PCA Evaluation")
    variance_col, variance_table_col = st.columns([1.4, 1], gap="large")
    with variance_col:
        render_pca_variance_plot(variance_df)
    with variance_table_col:
        st.dataframe(
            variance_df.style.format(
                {
                    "explained_variance_ratio": "{:.4f}",
                    "cumulative_explained_variance": "{:.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Clustering Evaluation")
    eval_left, eval_right = st.columns(2, gap="large")
    with eval_left:
        render_cluster_scatter_2d(processed_df, cluster_colors)
    with eval_right:
        render_cluster_scatter_3d(processed_df, cluster_colors)

    cluster_size_df = cluster_summary_df[["cluster_label", "cluster_type", "size", "share_of_districts"]].copy()
    st.dataframe(
        cluster_size_df.style.format({"share_of_districts": "{:.1%}"}),
        use_container_width=True,
        hide_index=True,
    )
