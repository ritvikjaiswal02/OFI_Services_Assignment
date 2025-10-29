# ============================================
# app.py
# Predictive Delivery Optimizer - NexGen Logistics
# Final stable version WITHOUT external LLMs
# Includes: simulation "What-if: Carrier Swap & Prioritization"
# Fixed: no label clipping (carrier plot + global tight_layout)
# Key Business Insight displayed as plain text
# ============================================

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="NexGen Predictive Delivery Optimizer", layout="wide")
st.title("NexGen Predictive Delivery Optimizer")
st.caption("Predict delivery delays and produce actionable, order-level recommendations")

# ---------- SIDEBAR ----------
st.sidebar.header("Upload Orders Data")
uploaded_file = st.sidebar.file_uploader("Upload your Orders CSV", type=["csv"])

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    return joblib.load("models/delivery_model.joblib")

try:
    pipe = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# ---------- HELPER: professional plain-text summary ----------
def generate_local_summary(total_orders, high_risk, med_risk, low_risk, top_features):
    """Generate a professional, plain-text business insight."""
    if top_features and top_features != "N/A":
        clean_feats = [f.replace("num__", "").replace("cat__", "").replace("_", " ").title() for f in top_features.split(", ")]
        feature_text = ", ".join(clean_feats[:6])
    else:
        feature_text = "key operational variables such as delivery time and route conditions"

    summary = (
        f"Across a total of {total_orders} shipments, the predictive model identified {high_risk} high-risk, "
        f"{med_risk} medium-risk, and {low_risk} low-risk deliveries. "
        f"The most influential factors impacting delay likelihood include {feature_text}. "
        "These generally reflect longer travel distances, suboptimal carrier performance, and higher logistical complexity. "
        "To mitigate risks, logistics teams should focus on re-routing high-risk shipments, reassigning carriers where needed, "
        "and optimizing schedules for time-critical routes."
    )
    return summary

# ---------- PLOT STYLE TUNING ----------
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 16,
    "axes.labelsize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

# utility to safely draw and return fig
def draw_and_show(fig, use_container=True):
    # apply tight layout and small adjustments to avoid clipping
    try:
        fig.tight_layout(pad=1.2)
    except Exception:
        pass
    st.pyplot(fig, use_container_width=use_container)


# ---------- MAIN ----------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # Optional enrichment from local files if uploaded file lacks columns
    required = {"Distance_KM", "Traffic_Delay_Minutes", "Carrier", "Promised_Delivery_Days", "Actual_Delivery_Days"}
    if not required.issubset(set(df.columns)):
        st.info("Integrating optional operational datasets (if available)...")
        try:
            data_dir = "data"
            delivery = pd.read_csv(os.path.join(data_dir, "delivery_performance.csv"))
            routes = pd.read_csv(os.path.join(data_dir, "routes_distance.csv"))
            cost = pd.read_csv(os.path.join(data_dir, "cost_breakdown.csv"))
            feedback = pd.read_csv(os.path.join(data_dir, "customer_feedback.csv"))
            df = df.merge(delivery, on="Order_ID", how="left")
            df = df.merge(routes[["Order_ID", "Distance_KM", "Traffic_Delay_Minutes", "Weather_Impact"]], on="Order_ID", how="left")
            df = df.merge(cost, on="Order_ID", how="left")
            if "Rating" in feedback.columns:
                df = df.merge(feedback[["Order_ID", "Rating"]], on="Order_ID", how="left")
            st.success("Optional operational data integrated successfully.")
        except Exception as e:
            st.warning(f"Optional dataset merge skipped ({e})")

    df.drop_duplicates(subset="Order_ID", inplace=True)
    df = df.copy()
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Predictions
    try:
        df["delay_prob"] = pipe.predict_proba(df)[:, 1]
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        st.stop()

    # Risk bucketing
    def risk_label(p):
        if p > 0.7:
            return "High Risk"
        elif p > 0.4:
            return "Medium Risk"
        return "Low Risk"

    df["Risk_Level"] = df["delay_prob"].apply(risk_label)

    # KPIs
    st.markdown("---")
    st.subheader("Shipment Risk Overview")
    total_orders = len(df)
    high_risk = int((df["Risk_Level"] == "High Risk").sum())
    med_risk = int((df["Risk_Level"] == "Medium Risk").sum())
    low_risk = int((df["Risk_Level"] == "Low Risk").sum())
    avg_prob = round(df["delay_prob"].mean() * 100, 2)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Orders", total_orders)
    k2.metric("High Risk", high_risk)
    k3.metric("Medium Risk", med_risk)
    k4.metric("Average Delay Probability", f"{avg_prob}%")

    # ---------- PLOTS: tuned to avoid label clipping ----------
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Delay Probability Distribution")
        fig, ax = plt.subplots(figsize=(6.2, 3.2))
        ax.hist(df["delay_prob"], bins=10, color="#4DA8DA", edgecolor="black")
        ax.set_xlabel("Predicted Delay Probability")
        ax.set_ylabel("Orders")
        ax.set_xlim(0, 1)
        # safe adjustments
        fig.subplots_adjust(bottom=0.15, left=0.12, right=0.98, top=0.92)
        draw_and_show(fig)

    with col2:
        st.subheader("Order Value vs Predicted Delay")
        fig, ax = plt.subplots(figsize=(6.2, 3.2))
        if "Order_Value_INR" in df.columns:
            ax.scatter(df["Order_Value_INR"], df["delay_prob"], s=20, alpha=0.7)
            ax.set_xlabel("Order Value (INR)")
            ax.set_ylabel("Predicted Delay Probability")
            try:
                x_max = np.percentile(df["Order_Value_INR"].dropna(), 99)
                ax.set_xlim(left=0, right=max(x_max, 1))
            except Exception:
                pass
        else:
            ax.text(0.5, 0.5, "Order_Value_INR not present", ha="center")
        fig.subplots_adjust(bottom=0.15, left=0.12, right=0.98, top=0.92)
        draw_and_show(fig)

    # Carrier & Feature importance (fixed margins)
    col3, col4 = st.columns([1.1, 1])
    with col3:
        st.subheader("Average Delay Risk by Carrier")
        if "Carrier" in df.columns:
            avg_risk_carrier = df.groupby("Carrier")["delay_prob"].mean().sort_values(ascending=False).head(6)

            # Adjusted plot size and margins to prevent clipping
            fig, ax = plt.subplots(figsize=(5.5, 2.8))
            ax.bar(avg_risk_carrier.index, avg_risk_carrier.values, color="#8EC6C5", edgecolor="k")

            # Label updated to "Average Delay Probability"
            ax.set_ylabel("Average Delay Probability", labelpad=8)
            ax.set_xlabel("Carrier", labelpad=6)
            ax.set_ylim(0, 1)

            # reduce xtick label fontsize slightly and rotate for readability
            for label in ax.get_xticklabels():
                label.set_fontsize(9)
                label.set_rotation(30)
                label.set_ha("right")

            # Tight layout + explicit margins to avoid text clipping
            try:
                fig.tight_layout(pad=1.6)
            except Exception:
                pass
            fig.subplots_adjust(left=0.18, right=0.97, top=0.92, bottom=0.30)
            draw_and_show(fig)
        else:
            st.info("Carrier column not present in uploaded data.")

    with col4:
        st.subheader("Top Factors Influencing Delay")
        feat_imp = None
        try:
            pre = pipe.named_steps["pre"]
            clf = pipe.named_steps["clf"]
            feature_names = pre.get_feature_names_out()
            importances = clf.feature_importances_
            feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
            feat_imp = feat_imp.sort_values("Importance", ascending=False).head(6)

            fig, ax = plt.subplots(figsize=(5.5, 2.8))
            ax.barh(feat_imp["Feature"], feat_imp["Importance"], color="#50C878", edgecolor="k")
            ax.invert_yaxis()
            ax.set_xlabel("Importance")
            # leave larger left margin to fit feature names
            try:
                fig.tight_layout(pad=1.6)
            except Exception:
                pass
            fig.subplots_adjust(left=0.33, right=0.98, top=0.92, bottom=0.12)
            draw_and_show(fig)
        except Exception:
            st.info("Feature importance not available for this model type.")
            feat_imp = None

    # ---------- ORDER-LEVEL ACTIONS ----------
    st.markdown("---")
    st.subheader("Order-Level Corrective Actions")

    def suggest_actions(row):
        p = row.get("delay_prob", 0)
        actions = []
        if p > 0.7:
            if row.get("Traffic_Delay_Minutes", 0) > 60:
                actions.append("Re-route: avoid peak congestion.")
            if row.get("Distance_KM", 0) > 250:
                actions.append("Use closer regional hub or split shipment.")
            if row.get("Carrier"):
                actions.append(f"Review carrier: {row['Carrier']}")
            if row.get("Rating", 5) < 3:
                actions.append("Investigate handling/quality.")
        elif 0.4 < p <= 0.7:
            actions.append("Monitor: moderate risk.")
        else:
            actions.append("Low risk: proceed normally.")
        return " | ".join(actions)

    df["Recommended_Action"] = df.apply(suggest_actions, axis=1)
    st.dataframe(df[["Order_ID", "Risk_Level", "delay_prob", "Recommended_Action"]], use_container_width=True, height=340)

    # ------------------ What-if: Carrier Swap & Prioritization (Simulations) ------------------
    st.markdown("---")
    st.subheader("What-if: Carrier Swap & Prioritization")

    # Only run if Carrier column exists
    if "Carrier" in df.columns:
        # Candidate carriers available in dataset (exclude current carrier duplicates)
        carrier_choices = sorted(df["Carrier"].dropna().unique().tolist())
        # Add a synthetic 'Express' option if you want a faster service simulation
        carrier_choices = ["(Simulate Express)"] + carrier_choices

        top_k = st.number_input("Select top K high-risk orders to simulate (by predicted prob)", min_value=1, max_value=min(200, len(df)), value=10, step=1)
        new_carrier_choice = st.selectbox("Simulate change to carrier / service:", carrier_choices)

        # select top K risky orders
        top_orders = df.sort_values("delay_prob", ascending=False).head(top_k).copy().reset_index(drop=True)
        st.write(f"Simulating for top {top_k} orders by predicted delay probability.")

        # helper to simulate a new row with changed carrier/service
        def simulate_row_change(row, new_carrier):
            r = row.copy()
            # If user selected synthetic Express, we simulate by lowering Promised_Delivery_Days if exists
            if new_carrier == "(Simulate Express)":
                # reduce promised days by 1 if numeric
                if "Promised_Delivery_Days" in r and pd.notna(r["Promised_Delivery_Days"]):
                    try:
                        r["Promised_Delivery_Days"] = max(1, float(r["Promised_Delivery_Days"]) - 1)
                    except Exception:
                        pass
                # optionally flag Priority to express if column exists
                if "Priority" in r:
                    r["Priority"] = "High"
            else:
                # set carrier field
                r["Carrier"] = new_carrier

            # Build a single-row DataFrame aligned with model input
            X_sim = r.to_frame().T.reindex(columns=df.columns, fill_value=np.nan)

            # run model predict_proba safely
            try:
                p_after = float(pipe.predict_proba(X_sim)[:, 1][0])
            except Exception:
                # if model can't handle missing features in sim, fallback to original prob
                p_after = float(row.get("delay_prob", np.nan))
            return p_after

        # Run simulation for top orders
        sim_rows = []
        for _, r in top_orders.iterrows():
            current_carrier = r.get("Carrier", "N/A")
            current_p = float(r["delay_prob"])
            new_p = simulate_row_change(r, new_carrier_choice)
            delta = current_p - new_p
            # Estimate extra cost heuristic
            extra_cost = None
            if "Delivery_Cost_INR" in df.columns and pd.notna(r.get("Delivery_Cost_INR")):
                base = float(r["Delivery_Cost_INR"])
                # express assumed +15% cost if simulate express, else switching carrier assume +5%
                if new_carrier_choice == "(Simulate Express)":
                    extra_cost = round(base * 0.15, 2)
                else:
                    extra_cost = round(base * 0.05, 2)
            else:
                # fallback: base on order value if present
                if "Order_Value_INR" in df.columns and pd.notna(r.get("Order_Value_INR")):
                    base = float(r["Order_Value_INR"])
                    extra_cost = round(base * (0.02 if new_carrier_choice != "(Simulate Express)" else 0.1), 2)
                else:
                    extra_cost = 0.0

            # ROI-like score: delta (prob reduction) per 1 INR extra cost (higher better). avoid divide by zero
            roi = (delta / extra_cost) if extra_cost and extra_cost > 0 else (delta * 1000)

            sim_rows.append({
                "Order_ID": r["Order_ID"],
                "Current_Carrier": current_carrier,
                "Current_Prob": round(current_p, 3),
                "New_Carrier": new_carrier_choice,
                "New_Prob": round(new_p, 3),
                "Delta_Prob": round(delta, 3),
                "Est_Extra_Cost_INR": extra_cost,
                "ROI_Score": round(roi, 6)
            })

        sim_df = pd.DataFrame(sim_rows).sort_values(["Delta_Prob", "ROI_Score"], ascending=[False, False]).reset_index(drop=True)

        st.markdown("**Simulation results (sorted by Delta Prob then ROI)**")
        st.dataframe(sim_df, use_container_width=True, height=300)

        # Download simulated recommendations
        st.download_button("Download Simulation Results (CSV)", data=sim_df.to_csv(index=False).encode("utf-8"),
                           file_name="carrier_simulation_results.csv", mime="text/csv")

    else:
        st.info("Carrier data not available — cannot run what-if simulations.")

    # ---------- KEY BUSINESS INSIGHT (plain text) ----------
    st.markdown("---")
    st.subheader("Key Business Insight")

    top_features = ", ".join(feat_imp["Feature"].tolist()) if (feat_imp is not None) else "N/A"
    business_insight = generate_local_summary(total_orders, high_risk, med_risk, low_risk, top_features)
    st.write(business_insight)

    # ---------- DOWNLOAD ----------
    st.markdown("---")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Predictions Report (CSV)", data=csv, file_name="predicted_delivery_risks.csv", mime="text/csv")

else:
    st.info("Upload your Orders CSV file to begin analysis.")
