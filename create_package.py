import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Predictive Delivery Optimizer", layout="wide")

# Title and header
st.title("🚛 Predictive Delivery Optimizer")
st.caption("AI-powered Logistics Delay Prediction | Ritvik Jaiswal | OFI AI Internship 2025")

# Sidebar for uploading file
st.sidebar.header("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your Orders CSV", type=["csv"])

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("models/delivery_model.joblib")

pipe = load_model()

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📦 Uploaded Data Preview")
    st.dataframe(df.head())

    # Predict delay probabilities
    df["delay_prob"] = pipe.predict_proba(df)[:, 1]
    df["Prediction"] = np.where(df["delay_prob"] > 0.5, "Likely Delayed", "On Time")

    st.markdown("---")
    st.subheader("🧮 Prediction Results")
    st.dataframe(df[["Order_ID", "delay_prob", "Prediction"]].head(20))

    # 📊 Delay Probability Distribution
    st.markdown("---")
    st.subheader("📊 Delay Probability Distribution")
    fig, ax = plt.subplots()
    ax.hist(df["delay_prob"], bins=20, color="#4DA8DA", edgecolor="black")
    ax.set_xlabel("Predicted Delay Probability")
    ax.set_ylabel("Number of Orders")
    st.pyplot(fig)

    # 💰 Cost vs Delay Probability
    if "Order_Value_INR" in df.columns:
        st.markdown("---")
        st.subheader("💰 Cost vs Delay Probability")
        fig, ax = plt.subplots()
        ax.scatter(df["Order_Value_INR"], df["delay_prob"], color="#FFA41B", alpha=0.7)
        ax.set_xlabel("Order Value (INR)")
        ax.set_ylabel("Predicted Delay Probability")
        st.pyplot(fig)

    # 🚀 Feature Importance (if model supports it)
    try:
        st.markdown("---")
        st.subheader("🚀 Top Features Impacting Delivery Delays")

        # Extract from trained model
        pre = pipe.named_steps["pre"]
        clf = pipe.named_steps["clf"]
        feature_names = pre.get_feature_names_out()
        importances = clf.feature_importances_

        feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        feat_imp = feat_imp.sort_values("Importance", ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(feat_imp["Feature"], feat_imp["Importance"], color="#50C878")
        ax.invert_yaxis()
        st.pyplot(fig)

    except Exception as e:
        st.info("Feature importance unavailable for this model type.")

    # 🧠 Business Insights
    st.markdown("---")
    st.subheader("🧠 Key Insights Summary")

    avg_delay = round(df["delay_prob"].mean() * 100, 2)
    high_risk_orders = df[df["delay_prob"] > 0.7]
    total_orders = len(df)
    cost_exposure = 0
    if "Order_Value_INR" in df.columns:
        cost_exposure = int(high_risk_orders["Order_Value_INR"].sum())

    st.write(f"📦 Total Orders Analyzed: **{total_orders}**")
    st.write(f"⚠️ High-Risk Orders (>70% chance of delay): **{len(high_risk_orders)}**")
    st.write(f"📈 Average Predicted Delay Probability: **{avg_delay}%**")
    if cost_exposure > 0:
        st.write(f"💸 Potential Cost Exposure (High-Risk Orders): ₹{cost_exposure:,}")

    st.markdown("---")
    st.success("✅ Analysis Complete! Use the Download button below to save predictions.")

    # Allow downloading of predictions
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Predictions CSV",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv",
    )

else:
    st.info("👈 Upload your Orders CSV file to begin the analysis.")
