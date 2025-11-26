import math
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Space Mission Launch Dashboard",
    layout="wide"
)

# -----------------------------
# DATA LOADING + CLEANING
# -----------------------------
@st.cache_data
def load_and_clean_data(csv_path: str = "Data/space_missions.csv"):
    # Load
    data = pd.read_csv(csv_path, encoding="latin1")

    # Drop duplicates
    data = data.drop_duplicates()

    # Drop rows with missing key fields
    data = data.dropna(subset=["Date", "MissionStatus", "Company"])

    # Clean Price column
    if "Price" in data.columns:
        data["Price"] = (
            data["Price"]
            .replace(r"[\$,]", "", regex=True)
            .astype(float)
        )
        # keep original but also clean version
        data["Price_Clean"] = data["Price"]
        data["Price_Clean"].fillna(data["Price_Clean"].mean(), inplace=True)
    else:
        data["Price_Clean"] = np.nan

    # Create datetime column
    if "Time" in data.columns:
        dt_series = data["Date"].astype(str) + " " + data["Time"].astype(str)
    else:
        dt_series = data["Date"].astype(str)

    data["LaunchDateTime"] = pd.to_datetime(dt_series, errors="coerce")

    # Extract date parts
    data["Year"] = data["LaunchDateTime"].dt.year
    data["Month"] = data["LaunchDateTime"].dt.month
    data["Weekday"] = data["LaunchDateTime"].dt.day_name()

    # Extract country from Location
    if "Location" in data.columns:
        data["Country"] = data["Location"].apply(
            lambda x: str(x).split(",")[-1].strip()
        )
    else:
        data["Country"] = "Unknown"

    return data


data = load_and_clean_data()

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.title("Filters")

# Year filter
year_min = int(data["Year"].min())
year_max = int(data["Year"].max())
year_range = st.sidebar.slider(
    "Launch Year Range",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max),
    step=1
)

# Company filter
companies = sorted(data["Company"].dropna().unique())
selected_companies = st.sidebar.multiselect(
    "Company",
    options=companies,
    default=companies
)

# Country filter
countries = sorted(data["Country"].dropna().unique())
selected_countries = st.sidebar.multiselect(
    "Country",
    options=countries,
    default=countries
)

# Mission status filter
statuses = sorted(data["MissionStatus"].dropna().unique())
selected_statuses = st.sidebar.multiselect(
    "Mission Status",
    options=statuses,
    default=statuses
)

# Apply filters
filtered = data[
    (data["Year"] >= year_range[0]) &
    (data["Year"] <= year_range[1]) &
    (data["Company"].isin(selected_companies)) &
    (data["Country"].isin(selected_countries)) &
    (data["MissionStatus"].isin(selected_statuses))
]

# -----------------------------
# HEADER + KPIs
# -----------------------------
st.title("🚀 Space Mission Launch Analysis Dashboard")

st.markdown(
    f"Showing data from **{year_range[0]}–{year_range[1]}** "
    f"for **{len(filtered)}** launches."
)

col1, col2, col3, col4 = st.columns(4)

total_launches = len(filtered)
success_count = (filtered["MissionStatus"] == "Success").sum()
failure_count = (filtered["MissionStatus"] == "Failure").sum()

success_rate = (
    (success_count / total_launches) * 100 if total_launches > 0 else 0
)
avg_price = filtered["Price_Clean"].mean()

with col1:
    st.metric("Total Launches", f"{total_launches:,}")

with col2:
    st.metric("Successful Missions", f"{success_count:,}")

with col3:
    st.metric("Success Rate", f"{success_rate:.1f}%")

with col4:
    if not np.isnan(avg_price):
        st.metric("Avg Launch Price (M$)", f"{avg_price:,.1f}")
    else:
        st.metric("Avg Launch Price (M$)", "N/A")

st.markdown("---")

# -----------------------------
# TABS FOR DIFFERENT ANALYSES
# -----------------------------
tab_overview, tab_success, tab_price, tab_temporal, tab_corr = st.tabs(
    ["📊 Overview", "✅ Success Analysis", "💰 Price Analysis",
     "⏱ Temporal Trends", "📈 Correlation"]
)

# -----------------------------
# TAB 1: OVERVIEW
# -----------------------------
with tab_overview:
    st.subheader("Launch Overview")

    col_a, col_b = st.columns(2)

    # Launches by Year
    with col_a:
        st.markdown("**Launches by Year**")
        launches_by_year = (
            filtered["Year"].value_counts().sort_index()
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.lineplot(
            x=launches_by_year.index,
            y=launches_by_year.values,
            marker="o",
            ax=ax
        )
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Launches")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    # Top Companies
    with col_b:
        st.markdown("**Top 10 Companies by Mission Count**")
        top_companies = filtered["Company"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(
            y=top_companies.index,
            x=top_companies.values,
            ax=ax
        )
        ax.set_xlabel("Missions")
        ax.set_ylabel("Company")
        st.pyplot(fig)

    st.markdown("---")

    col_c, col_d = st.columns(2)

    # Top Countries
    with col_c:
        st.markdown("**Top 10 Countries by Launch Count**")
        top_countries = filtered["Country"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(
            y=top_countries.index,
            x=top_countries.values,
            ax=ax
        )
        ax.set_xlabel("Missions")
        ax.set_ylabel("Country")
        st.pyplot(fig)

    # Mission Status Distribution
    with col_d:
        st.markdown("**Mission Status Distribution**")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(
            x="MissionStatus",
            data=filtered,
            ax=ax,
            order=filtered["MissionStatus"].value_counts().index
        )
        ax.set_xlabel("Mission Status")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    st.markdown("### Sample Data")
    st.dataframe(filtered.head(20))

# -----------------------------
# TAB 2: SUCCESS ANALYSIS
# -----------------------------
with tab_success:
    st.subheader("Success vs Failure by Country and Company")

    if len(filtered) == 0:
        st.warning("No data for the selected filters.")
    else:
        # Country success table
        country_success = (
            filtered.groupby(["Country", "MissionStatus"])
            .size().unstack().fillna(0)
        )

        st.markdown("**Top 10 Countries by Success Count**")
        st.dataframe(
            country_success.sort_values("Success", ascending=False).head(10)
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        top_country_success = country_success.sort_values(
            "Success", ascending=False
        ).head(10)
        top_country_success[["Success", "Failure"]].plot(
            kind="barh", stacked=True, ax=ax
        )
        ax.set_xlabel("Mission Count")
        ax.set_ylabel("Country")
        ax.set_title("Top Countries: Success vs Failure")
        st.pyplot(fig)

        # Company success table
        company_success = (
            filtered.groupby(["Company", "MissionStatus"])
            .size().unstack().fillna(0)
        )

        st.markdown("**Top 10 Companies by Success Count**")
        st.dataframe(
            company_success.sort_values("Success", ascending=False).head(10)
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        top_company_success = company_success.sort_values(
            "Success", ascending=False
        ).head(10)
        top_company_success[["Success", "Failure"]].plot(
            kind="barh", stacked=True, ax=ax
        )
        ax.set_xlabel("Mission Count")
        ax.set_ylabel("Company")
        ax.set_title("Top Companies: Success vs Failure")
        st.pyplot(fig)

# -----------------------------
# TAB 3: PRICE ANALYSIS
# -----------------------------
with tab_price:
    st.subheader("Launch Price Analysis")

    price_df = filtered.dropna(subset=["Price_Clean"])

    if price_df.empty:
        st.warning("No price data available for current filters.")
    else:
        col1, col2 = st.columns(2)

        # Avg price by mission status
        with col1:
            st.markdown("**Average Launch Price by Mission Outcome**")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(
                x="MissionStatus",
                y="Price_Clean",
                data=price_df,
                ax=ax
            )
            ax.set_xlabel("Mission Status")
            ax.set_ylabel("Avg Price (M$)")
            st.pyplot(fig)

        # Avg price by year
        with col2:
            st.markdown("**Average Launch Price Over Years**")
            avg_price_year = (
                price_df.groupby("Year")["Price_Clean"].mean().dropna()
            )
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.lineplot(
                x=avg_price_year.index,
                y=avg_price_year.values,
                marker="o",
                ax=ax
            )
            ax.set_xlabel("Year")
            ax.set_ylabel("Avg Price (M$)")
            ax.grid(alpha=0.3)
            st.pyplot(fig)

        st.markdown("**Price Summary by Mission Status**")
        st.dataframe(
            price_df.groupby("MissionStatus")["Price_Clean"].describe()
        )

# -----------------------------
# TAB 4: TEMPORAL TRENDS
# -----------------------------
with tab_temporal:
    st.subheader("Temporal Trends in Launches")

    col1, col2 = st.columns(2)

    # Launches by Weekday
    with col1:
        st.markdown("**Launches by Weekday**")
        fig, ax = plt.subplots(figsize=(6, 4))
        weekday_order = [
            "Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday", "Sunday"
        ]
        sns.countplot(
            x="Weekday",
            data=filtered,
            order=weekday_order,
            ax=ax
        )
        ax.set_xlabel("Weekday")
        ax.set_ylabel("Number of Launches")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)

    # Launches by Month
    with col2:
        st.markdown("**Launches by Month**")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(
            x="Month",
            data=filtered,
            ax=ax
        )
        ax.set_xlabel("Month")
        ax.set_ylabel("Number of Launches")
        st.pyplot(fig)

    # Mission status over time
    st.markdown("**Mission Status Over Time**")
    status_over_time = (
        filtered.groupby(["Year", "MissionStatus"])
        .size().unstack().fillna(0)
    )

    if not status_over_time.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        status_over_time.plot(ax=ax, marker="o")
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Missions")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    else:
        st.info("No missions available for the selected filters.")

# -----------------------------
# TAB 5: CORRELATION
# -----------------------------
with tab_corr:
    st.subheader("Correlation Analysis")

    num_df = filtered[["Price_Clean", "Year", "Month"]].dropna()

    if num_df.empty:
        st.warning("Not enough numeric data for correlation analysis.")
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            num_df.corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            ax=ax
        )
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

        st.markdown("**Correlation Table**")
        st.dataframe(num_df.corr())
