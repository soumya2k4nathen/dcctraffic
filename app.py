import streamlit as st
st.set_page_config(page_title="Road Accident Dashboard", layout="wide")

import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
from utils.preprocessing import prepare_data
from utils.models import forecast_arima, cluster_states

# Load and clean data
df = pd.read_csv("data/traffic.csv")
df_clean = prepare_data(df)

# Sidebar navigation menu
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=[
            "Home",
            "Exploratory Data Analysis",
            "Accident Forecasting",
            "State-wise Risk Clustering",
            "Cause-based Breakdown",
            "Weather Condition Impact"
        ],
        icons=[
            "house", "bar-chart", "graph-up",
            "diagram-3", "exclamation-triangle", "cloud"
        ],
        default_index=0
    )

# App title
st.title("🚧 Road Accident Data Analysis (NCRB 2020)")

# Home: Univariate, Bivariate, Multivariate
if selected == "Home":
    st.subheader("📊 Basic Data Analysis")
    analysis_type = st.radio("Select Analysis Type", ["Univariate", "Bivariate", "Multivariate"], horizontal=True)

    if analysis_type == "Univariate":
        col = st.selectbox("Select column to analyze", df.columns[2:])
        fig = px.bar(df, x="State/UT/City", y=col, title=f"{col} across States", labels={col: col, "State/UT/City": "Location"})
        st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "Bivariate":
        x_col = st.selectbox("X-axis", df.columns[2:], key="biv_x")
        y_col = st.selectbox("Y-axis", df.columns[2:], key="biv_y")
        fig = px.scatter(df, x=x_col, y=y_col, color="State/UT/City", hover_name="State/UT/City")
        st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "Multivariate":
        x_col = st.selectbox("X-axis", df.columns[2:], key="multi_x")
        y_col = st.selectbox("Y-axis", df.columns[2:], key="multi_y")
        z_col = st.selectbox("Z-axis", df.columns[2:], key="multi_z")
        fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color="State/UT/City")
        st.plotly_chart(fig, use_container_width=True)

# EDA
elif selected == "Exploratory Data Analysis":
    st.subheader("📈 Exploratory Data Analysis")
    st.dataframe(df_clean.head())

    categorywise = df_clean.groupby("Category")["Total"].sum()
    fig1 = px.bar(categorywise, title="Total Accidents by Category")
    st.plotly_chart(fig1)

    statewise = df_clean.groupby("State/UT")["Total"].sum().sort_values(ascending=False).head(10)
    fig2 = px.bar(statewise, title="Top 10 States by Total Accidents")
    st.plotly_chart(fig2)

# Forecasting
elif selected == "Accident Forecasting":
    st.subheader("📊 Accident Forecasting using ARIMA")
    state = st.selectbox("Select a State", df_clean["State/UT"].unique())
    forecast_fig = forecast_arima(df_clean, state, by="Category")
    st.pyplot(forecast_fig)

# Clustering
elif selected == "State-wise Risk Clustering":
    st.subheader("🧪 State-wise Risk Clustering")
    fig, labels = cluster_states(df_clean)
    st.pyplot(fig)
    st.write("Cluster Labels by State:")
    st.dataframe(labels)

# Cause-based
elif selected == "Cause-based Breakdown":
    st.subheader("📊 Accidents by Cause")
    cause = st.selectbox("Select Cause of Accident", df.columns[3:])
    cause_data = df_clean.groupby("State/UT")[cause].sum().sort_values(ascending=False)
    fig3 = px.bar(cause_data, title=f"Accidents by {cause}")
    st.plotly_chart(fig3)

# Weather
elif selected == "Weather Condition Impact":
    st.subheader("🌦️ Weather Condition Impact on Accidents")
    weather_condition = st.selectbox("Select Weather Condition", [
        "Weather Condition (Poor Visibility) - Cases",
        "Weather Condition (Others Causes) - Cases"
    ])
    weather_data = df_clean.groupby("State/UT")[weather_condition].sum().sort_values(ascending=False)
    fig4 = px.bar(weather_data, title=f"Accidents Due to {weather_condition}")
    st.plotly_chart(fig4)
