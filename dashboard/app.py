import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Demand Forecast Intelligence", layout="wide")
st.title("Consumer Demand Forecasting & Marketing Mix Intelligence")

df     = pd.read_csv("data/processed/model_output.csv", parse_dates=["date"])
did    = pd.read_csv("data/processed/did_results.csv", index_col=0)
shap_df= pd.read_csv("data/processed/shap_values.csv")

# --- Row 1: KPI cards ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue", f"${df['total_revenue'].sum():,.0f}")
col2.metric("Avg Monthly Revenue", f"${df['total_revenue'].mean():,.0f}")
col3.metric("Promo Months", f"{df['promo_flag'].sum()}")
col4.metric("Avg Sentiment Score", f"{df['sentiment_score'].mean():.2f}")

st.divider()

# --- Row 2: Forecast vs Actuals ---
st.subheader("Demand Forecast vs Actuals")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df["date"], y=df["total_revenue"],
    mode="lines+markers", name="Actual Revenue", line=dict(color="steelblue")))
fig1.add_trace(go.Scatter(x=df["date"], y=df["predicted_revenue"],
    mode="lines+markers", name="Predicted Revenue", line=dict(color="orange", dash="dash")))
fig1.update_layout(xaxis_title="Date", yaxis_title="Revenue ($)", height=350)
st.plotly_chart(fig1, use_container_width=True)

st.divider()

# --- Row 3: Sentiment + Promo flags ---
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Consumer Sentiment Over Time")
    fig2 = px.line(df, x="date", y="sentiment_score", title="Sentiment Score (VADER)")
    fig2.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Neutral")
    st.plotly_chart(fig2, use_container_width=True)

with col_b:
    st.subheader("Promotional Impact Scorecard")
    promo    = df[df["promo_flag"]==1]["total_revenue"].mean()
    no_promo = df[df["promo_flag"]==0]["total_revenue"].mean()
    did_est  = did.loc["did", "coefficient"]
    did_pval = did.loc["did", "p_value"]

    st.metric("Avg Revenue — Promo Months", f"${promo:,.0f}")
    st.metric("Avg Revenue — Non-Promo Months", f"${no_promo:,.0f}")
    st.metric("DiD Causal Estimate", f"${did_est:,.0f}")
    st.metric("P-Value", f"{did_pval:.4f}")
    if did_pval < 0.05:
        st.success("Statistically significant promotional lift detected")
    else:
        st.warning("Lift not significant — insufficient data for causal confirmation")

st.divider()

# --- Row 4: Feature importance via SHAP ---
st.subheader("Feature Importance (SHAP Values)")
shap_means = shap_df.drop(columns=["date"]).abs().mean().sort_values(ascending=True)
fig3 = px.bar(shap_means, orientation="h",
    labels={"index": "Feature", "value": "Mean |SHAP Value|"},
    title="Which features drive revenue predictions most")
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# --- Row 5: Economic indicators ---
st.subheader("Economic Indicators Over Time")
indicator = st.selectbox("Select indicator", 
    ["consumer_sentiment", "cpi", "unemployment", "disposable_income"])
fig4 = px.line(df, x="date", y=indicator, title=indicator.replace("_", " ").title())
st.plotly_chart(fig4, use_container_width=True)