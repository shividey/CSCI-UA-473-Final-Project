import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import winsorize
from numpy.linalg import qr


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

VALID_BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]

BEDROOM_MAP = {
    0: ("gross_rent_0_1beds_usd", "ah_studio_share"),
    1: ("gross_rent_0_1beds_usd", "ah_1br_share"),
    2: ("gross_rent_2_3beds_usd", "ah_2br_share"),
    3: ("gross_rent_2_3beds_usd", "ah_3br_share"),
}


BOROUGH_MAP = {
    "Battery Park/Tribeca": "Manhattan",
    "Central Harlem": "Manhattan",
    "Chelsea/Clinton": "Manhattan",
    "Clinton/Chelsea": "Manhattan",
    "East Harlem": "Manhattan",
    "Financial District": "Manhattan",
    "Gramercy/Murray Hill": "Manhattan",
    "Greenwich Village/Soho": "Manhattan",
    "Inwood/Washington Heights": "Manhattan",
    "Lower East Side/Chinatown": "Manhattan",
    "Midtown": "Manhattan",
    "Morningside Heights/Hamilton": "Manhattan",
    "Stuyvesant Town/Turtle Bay": "Manhattan",
    "Upper East Side": "Manhattan",
    "Upper West Side": "Manhattan",
    "Washington Heights/Inwood": "Manhattan",

    "Bay Ridge/Dyker Heights": "Brooklyn",
    "Bedford Stuyvesant": "Brooklyn",
    "Bensonhurst": "Brooklyn",
    "Borough Park": "Brooklyn",
    "Brownsville": "Brooklyn",
    "Bushwick": "Brooklyn",
    "Canarsie/Flatlands": "Brooklyn",
    "Coney Island": "Brooklyn",
    "Crown Heights/Prospect Heights": "Brooklyn",
    "East Flatbush": "Brooklyn",
    "East New York": "Brooklyn",
    "Flatbush/Midwood": "Brooklyn",
    "Fort Greene/Brooklyn Heights": "Brooklyn",
    "Greenpoint/Williamsburg": "Brooklyn",
    "Park Slope/Carroll Gardens": "Brooklyn",
    "Sheepshead Bay": "Brooklyn",
    "Sunset Park": "Brooklyn",
    "Williamsburg/Bushwick": "Brooklyn",
    "East New York/Starrett City": "Brooklyn",
    "South Crown Heights/Lefferts Gardens": "Brooklyn",
    "Flatlands/Canarsie": "Brooklyn",

    "Astoria": "Queens",
    "Bayside/Little Neck": "Queens",
    "Elmhurst/Corona": "Queens",
    "Flushing/Whitestone": "Queens",
    "Hillcrest/Fresh Meadows": "Queens",
    "Jackson Heights": "Queens",
    "Jamaica/Hollis": "Queens",
    "Kew Gardens/Woodhaven": "Queens",
    "Long Island City/Astoria": "Queens",
    "Queens Village": "Queens",
    "Rego Park/Forest Hills": "Queens",
    "Ridgewood/Maspeth": "Queens",
    "Rockaway/Broad Channel": "Queens",
    "South Ozone Park/Howard Beach": "Queens",
    "Woodside/Sunnyside": "Queens",

    "Belmont/East Tremont": "Bronx",
    "Co-op City": "Bronx",
    "Fordham/University Heights": "Bronx",
    "Highbridge/Concourse": "Bronx",
    "Hunts Point/Longwood": "Bronx",
    "Kingsbridge/Riverdale": "Bronx",
    "Morris Park/Bronxdale": "Bronx",
    "Morrisania/Crotona": "Bronx",
    "Mott Haven/Melrose": "Bronx",
    "Pelham/Throgs Neck": "Bronx",
    "Wakefield/Williamsbridge": "Bronx",
    "Kingsbridge Heights/Bedford": "Bronx",
    "Riverdale/Fieldston": "Bronx",
    "Parkchester/Soundview": "Bronx",
    "Throgs Neck/Co-op City": "Bronx",
    "Williamsbridge/Baychester": "Bronx",

    "South Beach/Willowbrook": "Staten Island",
    "St. George/Stapleton": "Staten Island",
    "Tottenville/Great Kills": "Staten Island",
}


def budget_fit(
    district_rent,
    budget,
    over_steepness=0.004,
    over_center_pct=0.05,
    under_steepness=0.001,
    under_center_pct=0.50,
):
    """
    Calculates how well a neighborhood rent matches the user's budget.
    Rent slightly under budget scores well.
    Rent above budget is penalized.
    Rent far below budget is softened.
    """
    if pd.isna(district_rent):
        return np.nan

    if district_rent > budget:
        center = budget * over_center_pct
        return 1 / (1 + np.exp(over_steepness * (district_rent - budget - center)))
    else:
        center = budget * under_center_pct
        return 1 / (1 + np.exp(under_steepness * (budget - district_rent - center)))


def validate_inputs(user_weights, user_rent_budget, user_bedrooms, user_boroughs):
    """
    Checks whether user inputs are valid.
    """
    if user_rent_budget <= 0:
        raise ValueError("Rent budget must be greater than 0.")

    if user_bedrooms not in BEDROOM_MAP:
        raise ValueError("Bedrooms must be 0, 1, 2, or 3.")

    missing = [cat for cat in CATEGORIES if cat not in user_weights]
    if missing:
        raise ValueError(f"Missing user weight(s): {missing}")

    if sum(user_weights.values()) <= 0:
        raise ValueError("At least one user weight must be greater than 0.")

    if user_boroughs is not None:
        invalid = [b for b in user_boroughs if b not in VALID_BOROUGHS]
        if invalid:
            raise ValueError(f"Invalid borough(s): {invalid}")


def prepare_features(df):
    """
    Cleans and standardizes the feature table before recommendation.
    """
    df = df.copy()

    required_cols = ["region_name"] + FEATURES
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    df["borough"] = df["region_name"].map(BOROUGH_MAP)

    # Lower crime should be better, so multiply by -1.
    df["crime_all_rt"] = -df["crime_all_rt"]
    df["crime_viol_rt"] = -df["crime_viol_rt"]

    # Reduce extreme outliers.
    for col in FEATURES:
        df[col] = winsorize(df[col], limits=[0.017, 0.017])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES].values)

    df["school_avg"] = (df["pct_prof_math"] + df["pct_prof_ela"]) / 2

    return df, X_scaled


def build_category_projection(X_scaled):
    """
    Builds category-level representations for Safety, Schools, Transit, and Parks.
    """
    priority_vectors = {
        "Safety": np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float),
        "Schools": np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=float),
        "Transit": np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0], dtype=float),
        "Parks": np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=float),
    }

    for category in priority_vectors:
        priority_vectors[category] /= priority_vectors[category].sum()

    basis = np.array([priority_vectors[c] for c in CATEGORIES])

    Q, _ = qr(basis.T)
    Q = Q[:, :len(CATEGORIES)].T

    for i in range(len(CATEGORIES)):
        if np.dot(Q[i], basis[i]) < 0:
            Q[i] *= -1

    X_proj = X_scaled @ Q.T

    return X_proj


def add_preference_score(df, X_proj, user_weights):
    """
    Calculates how well each neighborhood matches user priorities.
    """
    df = df.copy()

    cat_weights = np.array([user_weights[c] for c in CATEGORIES], dtype=float)
    cat_weights = cat_weights / cat_weights.sum()

    preference_scores = (X_proj * cat_weights).sum(axis=1)

    # Keep only positive preference matches.
    df["preference_score"] = np.clip(preference_scores, 0, None)

    return df


def add_price_fit(df, user_rent_budget, user_bedrooms):
    """
    Calculates budget fit and unit availability fit.
    """
    df = df.copy()

    rent_col, share_col = BEDROOM_MAP[user_bedrooms]

    if rent_col not in df.columns:
        raise ValueError(f"Dataset is missing rent column: {rent_col}")

    if share_col not in df.columns:
        raise ValueError(f"Dataset is missing bedroom-share column: {share_col}")

    df["estimated_rent"] = df[rent_col]

    df["budget_fit"] = df["estimated_rent"].apply(
        lambda rent: budget_fit(rent, user_rent_budget)
    )

    raw_share = df[share_col].fillna(0).clip(0, 1)
    df["unit_availability"] = np.sqrt(raw_share)

    availability_weight = 0.25

    df["price_fit_factor"] = (
        df["budget_fit"]
        * (1 - availability_weight + availability_weight * df["unit_availability"])
    )

    return df


def add_final_score(df):
    """
    Combines preference match and budget fit into final score.
    """
    df = df.copy()

    df["final_score"] = df["preference_score"] * df["price_fit_factor"]

    return df


def build_explanation(row, user_weights, user_rent_budget):
    """
    Creates a short explanation for each recommendation.
    """
    top_priority = max(user_weights, key=user_weights.get)

    rent = row["estimated_rent"]

    if pd.isna(rent):
        budget_text = "rent information is unavailable"
    elif rent > user_rent_budget:
        budget_text = "slightly above your budget"
    elif rent >= 0.8 * user_rent_budget:
        budget_text = "near your target budget"
    else:
        budget_text = "below your target budget"

    return (
        f"Recommended because it has a strong {top_priority.lower()} match "
        f"and its rent is {budget_text}."
    )


def format_results(df, user_weights, user_rent_budget):
    """
    Selects and formats the columns shown in the app.
    """
    results = df[
        [
            "region_name",
            "region_display",
            "borough",
            "estimated_rent",
            "preference_score",
            "price_fit_factor",
            "final_score",
            "crime_viol_rt",
            "school_avg",
            "mta_station_count",
        ]
    ].copy()

    results["reason"] = results.apply(
        lambda row: build_explanation(row, user_weights, user_rent_budget),
        axis=1,
    )

    results["preference_score"] = results["preference_score"].round(4)
    results["price_fit_factor"] = results["price_fit_factor"].round(4)
    results["final_score"] = results["final_score"].round(4)
    results["estimated_rent"] = results["estimated_rent"].round(0)

    return results


def recommend_neighborhoods(
    df,
    user_weights,
    user_rent_budget,
    user_bedrooms,
    user_boroughs=None,
    top_k=5,
):
    """
    Main recommender function.

    This function takes:
    - cleaned district feature table
    - user weights
    - rent budget
    - bedroom need
    - borough preference

    It returns:
    - top recommended neighborhoods
    - scores
    - explanations
    """

    validate_inputs(
        user_weights=user_weights,
        user_rent_budget=user_rent_budget,
        user_bedrooms=user_bedrooms,
        user_boroughs=user_boroughs,
    )

    df_prepared, X_scaled = prepare_features(df)

    X_proj = build_category_projection(X_scaled)

    scored = add_preference_score(
        df=df_prepared,
        X_proj=X_proj,
        user_weights=user_weights,
    )

    scored = add_price_fit(
        df=scored,
        user_rent_budget=user_rent_budget,
        user_bedrooms=user_bedrooms,
    )

    scored = add_final_score(scored)

    if user_boroughs is not None:
        scored = scored[scored["borough"].isin(user_boroughs)]

    if scored.empty:
        return pd.DataFrame()

    results = format_results(
        df=scored,
        user_weights=user_weights,
        user_rent_budget=user_rent_budget,
    )

    results = results.sort_values("final_score", ascending=False).head(top_k)

    return results.reset_index(drop=True)


if __name__ == "__main__":
    df = pd.read_excel("final_district_feature_table_latest_year_affordability_imputed.xlsx")

    user_weights = {
        "Safety": 5,
        "Schools": 5,
        "Transit": 2,
        "Parks": 2,
    }

    recommendations = recommend_neighborhoods(
        df=df,
        user_weights=user_weights,
        user_rent_budget=3800,
        user_bedrooms=1,
        user_boroughs=["Manhattan", "Brooklyn"],
        top_k=5,
    )

    print(recommendations)