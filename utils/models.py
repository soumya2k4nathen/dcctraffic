import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def forecast_arima(df, state, by="Category"):
    # Filter data for selected state
    df_state = df[df["State/UT"] == state]
    
    # Group by the specified column and sort (e.g., by "Category")
    grouped = df_state.groupby(by)["Total"].sum().sort_index()

    # Build ARIMA model
    model = ARIMA(grouped, order=(1,1,1))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=3)

    # Plot
    fig, ax = plt.subplots()
    grouped.plot(ax=ax, label="Actual")
    forecast.plot(ax=ax, label="Forecast", style='--')
    ax.set_title(f"Forecasting Accidents in {state}")
    ax.legend()
    return fig

def cluster_states(df):
    grouped = df.groupby("State/UT")["Total"].sum().reset_index()

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(grouped[["Total"]])

    # KMeans Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    grouped["Cluster"] = kmeans.fit_predict(X)

    # Plot
    fig, ax = plt.subplots()
    colors = ['red', 'green', 'blue']
    for i in range(3):
        cluster_data = grouped[grouped["Cluster"] == i]
        ax.bar(cluster_data["State/UT"], cluster_data["Total"], label=f"Cluster {i}", color=colors[i])
    ax.set_title("State-wise Accident Clusters")
    ax.set_xticklabels(grouped["State/UT"], rotation=90)
    ax.legend()

    return fig, grouped[["State/UT", "Cluster"]]
