import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.markdown("""
<style>

/* Main app font */
html, body, [class*="css"]  {
    font-size: 20px;
}

/* Titles */
h1 {
    font-size: 40px !important;
}

h2 {
    font-size: 32px !important;
}

h3 {
    font-size: 32px !important;
}

/* Labels (inputs) */
label {
    font-size: 20px !important;
}

/* Metric numbers */
div[data-testid="stMetricValue"] {
    font-size: 28px !important;
}

/* Metric labels */
div[data-testid="stMetricLabel"] {
    font-size: 20px !important;
}

</style>
""", unsafe_allow_html=True)
st.set_page_config(
    page_title="ICU Mortality Prediction Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_PATH = "final_16_features_plus_target.csv"
MODEL_PATH = "final_gradient_boosting_model.pkl"
IMPUTER_PATH = "model_imputer.pkl"
FEATURES_PATH = "model_features.pkl"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, imputer, features


def format_label(col_name: str) -> str:
    return col_name.replace("_", " ").title()


def compute_percentile(series, value):
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) == 0:
        return np.nan
    return round((clean <= value).mean() * 100, 1)


st.markdown("""
<style>
.main > div {
    padding-top: 1.0rem;
}
.block-container {
    padding-top: 1.0rem;
    padding-bottom: 1.8rem;
}
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(128,128,128,0.18);
    border-radius: 14px;
    padding: 10px 12px;
}
.custom-card {
    border: 1px solid rgba(128,128,128,0.18);
    border-radius: 16px;
    padding: 1rem 1.1rem;
    background: rgba(255,255,255,0.03);
}
div[data-testid="stNumberInput"] {
    max-width: 130px;
}
div[data-testid="stSelectbox"] {
    max-width: 260px;
}
.compact-label {
    padding-top: 0.35rem;
    font-size: 0.98rem;
    font-weight: 600;
}
.section-note {
    color: #aab0b6;
    font-size: 0.92rem;
}
</style>
""", unsafe_allow_html=True)

try:
    df = load_data()
    model, imputer, model_features = load_artifacts()
except Exception as e:
    st.error("Failed to load app resources.")
    st.exception(e)
    st.stop()

TARGET = "hospital_death"
if TARGET not in df.columns:
    st.error(f"Target column '{TARGET}' was not found in the dataset.")
    st.stop()

apache_features = [c for c in model_features if "apache" in c.lower()]
non_apache_features = [c for c in model_features if "apache" not in c.lower()]
numeric_features = [c for c in model_features if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

st.sidebar.title("🏥 ICU Dashboard")
page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Dataset Overview",
        "EDA",
        "Feature Importance",
        "Model Performance",
        "Prediction"
    ]
)
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit + Plotly")

if page == "Home":
    st.title("🏥 ICU Mortality Prediction Dashboard")
    st.markdown("Predict hospital mortality risk using selected **APACHE** and **non-APACHE** clinical features.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Rows", f"{df.shape[0]:,}")
    c2.metric("Features Used", f"{len(model_features)}")
    c3.metric("Mortality Rate", f"{df[TARGET].mean() * 100:.1f}%")
    c4.metric("Visualization Package", "Plotly")

    st.markdown("---")
    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("Project Overview")
        st.markdown("""
This dashboard presents the full ICU mortality prediction workflow:

- **dataset exploration**
- **feature interpretation**
- **model comparison**
- **real-time mortality risk prediction**

The objective is to identify patients at higher risk using structured clinical variables.
""")
        g1, g2 = st.columns(2)
        with g1:
            st.markdown(f'<div class="custom-card"><b>APACHE Features</b><br>{len(apache_features)} variables</div>', unsafe_allow_html=True)
        with g2:
            st.markdown(f'<div class="custom-card"><b>Non-APACHE Features</b><br>{len(non_apache_features)} variables</div>', unsafe_allow_html=True)

    with right:
        st.subheader("Target Distribution")
        mortality_counts = df[TARGET].map({0: "Survived", 1: "Died"}).value_counts().reset_index()
        mortality_counts.columns = ["Outcome", "Count"]
        fig = px.bar(mortality_counts, x="Outcome", y="Count", text="Count", title="Hospital Outcome Distribution")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=420, xaxis_title="", yaxis_title="Patients")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Dataset Overview":
    st.title("📋 Dataset Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")

    st.markdown("---")
    st.subheader("Dataset Preview")
    st.dataframe(df.head(15), use_container_width=True)

    st.markdown("---")
    st.subheader("Selected Model Features")
    feature_table = pd.DataFrame({
        "Feature": model_features,
        "Feature Type": ["APACHE" if "apache" in f.lower() else "Non-APACHE" for f in model_features]
    })
    st.dataframe(feature_table, use_container_width=True)

    st.markdown("---")
    st.subheader("Missing Values by Feature")
    missing_df = df[model_features].isna().sum().sort_values(ascending=False).reset_index()
    missing_df.columns = ["Feature", "Missing Count"]
    fig = px.bar(missing_df, x="Missing Count", y="Feature", orientation="h", title="Missing Values in Selected Features")
    fig.update_layout(height=max(420, 40 * len(missing_df)))
    st.plotly_chart(fig, use_container_width=True)

elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")

    top_row_left, top_row_right = st.columns([1, 1])

    with top_row_left:
        st.subheader("Mortality Distribution")
        mort = df[TARGET].map({0: "Survived", 1: "Died"}).value_counts().reset_index()
        mort.columns = ["Outcome", "Count"]
        fig1 = px.pie(mort, names="Outcome", values="Count", hole=0.45, title="Hospital Mortality Share")
        fig1.update_layout(height=420)
        st.plotly_chart(fig1, use_container_width=True)

    with top_row_right:
        st.subheader("Average Feature Level by Outcome")
        default_avg_features = numeric_features[:6] if len(numeric_features) >= 6 else numeric_features
        avg_df = df[default_avg_features + [TARGET]].copy()
        avg_df["Outcome"] = avg_df[TARGET].map({0: "Survived", 1: "Died"})
        avg_long = avg_df.groupby("Outcome")[default_avg_features].mean().reset_index().melt(
            id_vars="Outcome", var_name="Feature", value_name="Average Value"
        )
        fig_avg = px.bar(
            avg_long,
            x="Feature",
            y="Average Value",
            color="Outcome",
            barmode="group",
            title="Average Selected Feature Values by Outcome"
        )
        fig_avg.update_layout(height=420)
        st.plotly_chart(fig_avg, use_container_width=True)

    st.markdown("---")
    st.subheader("Feature Distribution")
    hist_feature = st.selectbox("Choose a feature for distribution view", numeric_features)

    fig2 = px.histogram(
        df,
        x=hist_feature,
        color=df[TARGET].map({0: "Survived", 1: "Died"}),
        barmode="overlay",
        nbins=30,
        title=f"Distribution of {format_label(hist_feature)} by Outcome"
    )
    fig2.update_layout(height=450, xaxis_title=format_label(hist_feature), yaxis_title="Count", legend_title="Outcome")
    st.plotly_chart(fig2, use_container_width=True)

    mid_left, mid_right = st.columns([1, 1])

    with mid_left:
        st.subheader("Box Plot vs Mortality")
        box_feature = st.selectbox("Choose a feature for box plot", numeric_features, index=min(1, len(numeric_features)-1))
        temp_box = df[[box_feature, TARGET]].copy()
        temp_box["Outcome"] = temp_box[TARGET].map({0: "Survived", 1: "Died"})
        fig3 = px.box(temp_box, x="Outcome", y=box_feature, points="outliers", title=f"{format_label(box_feature)} vs Mortality")
        fig3.update_layout(height=450, xaxis_title="", yaxis_title=format_label(box_feature))
        st.plotly_chart(fig3, use_container_width=True)

    with mid_right:
        st.subheader("Feature vs Feature Scatter")
        if len(numeric_features) >= 2:
            x_feature = st.selectbox("X-axis feature", numeric_features, index=0, key="scatter_x")
            y_feature = st.selectbox("Y-axis feature", numeric_features, index=min(1, len(numeric_features)-1), key="scatter_y")
            scatter_df = df[[x_feature, y_feature, TARGET]].copy()
            scatter_df["Outcome"] = scatter_df[TARGET].map({0: "Survived", 1: "Died"})
            fig_scatter = px.scatter(
                scatter_df,
                x=x_feature,
                y=y_feature,
                color="Outcome",
                opacity=0.65,
                title=f"{format_label(x_feature)} vs {format_label(y_feature)}"
            )
            fig_scatter.update_layout(height=450)
            st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")
    st.subheader("Correlation Heatmap")
    corr_features = numeric_features[:min(len(numeric_features), 12)]
    corr = df[corr_features].corr(numeric_only=True)
    fig4 = px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlation Matrix of Selected Numeric Features")
    fig4.update_layout(height=650)
    st.plotly_chart(fig4, use_container_width=True)

elif page == "Feature Importance":
    st.title("⭐ Feature Importance")
    st.markdown("This section highlights the selected clinical predictors used by the final model.")

    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "Feature": model_features,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True)

        left, right = st.columns([1.1, 0.9])

        with left:
            fig = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Feature Importance from Final Gradient Boosting Model"
            )
            fig.update_layout(height=max(500, 36 * len(importance_df)))
            st.plotly_chart(fig, use_container_width=True)

        with right:
            top_features = importance_df.sort_values("Importance", ascending=False).head(6).copy()
            top_features = top_features.sort_values("Importance", ascending=True)
            fig_top = px.bar(
                top_features,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 6 Predictive Features"
            )
            fig_top.update_layout(height=500)
            st.plotly_chart(fig_top, use_container_width=True)
            st.dataframe(top_features.sort_values("Importance", ascending=False), use_container_width=True)
    else:
        st.warning("The current model does not expose feature importance values.")

elif page == "Model Performance":
    st.title("📈 Model Performance")
    st.markdown("Below is a concise comparison table used to summarize the main modeling results.")

    perf_df = pd.DataFrame([
        ["APACHE only", "Logistic Regression", 0.8450],
        ["APACHE only", "Random Forest", 0.8520],
        ["APACHE only", "Gradient Boosting", 0.8580],
        ["Non-APACHE only", "Logistic Regression", 0.7810],
        ["Non-APACHE only", "Random Forest", 0.8010],
        ["Non-APACHE only", "Gradient Boosting", 0.8140],
        ["Combined", "Logistic Regression", 0.8520],
        ["Combined", "Random Forest", 0.8590],
        ["Combined", "Gradient Boosting", 0.8631],
    ], columns=["Feature Set", "Model", "ROC-AUC"])

    top_left, top_right = st.columns([1, 1.15])

    with top_left:
        st.dataframe(perf_df, use_container_width=True)

    with top_right:
        fig = px.bar(
            perf_df,
            x="Feature Set",
            y="ROC-AUC",
            color="Model",
            barmode="group",
            text="ROC-AUC",
            title="ROC-AUC Comparison Across Models"
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(height=500, yaxis_range=[0.70, 0.90])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    bottom_left, bottom_right = st.columns([1, 1])

    with bottom_left:
        pivot = perf_df.pivot(index="Model", columns="Feature Set", values="ROC-AUC").reset_index()
        melt = pivot.melt(id_vars="Model", var_name="Feature Set", value_name="ROC-AUC")
        fig_line = px.line(
            melt,
            x="Feature Set",
            y="ROC-AUC",
            color="Model",
            markers=True,
            title="Performance Trend Across Feature Sets"
        )
        fig_line.update_layout(height=420, yaxis_range=[0.70, 0.90])
        st.plotly_chart(fig_line, use_container_width=True)

    with bottom_right:
        best_row = perf_df.sort_values("ROC-AUC", ascending=False).iloc[0]
        st.success(
            f"Best observed setup: **{best_row['Model']}** using **{best_row['Feature Set']}** features with ROC-AUC = **{best_row['ROC-AUC']:.4f}**."
        )

        feature_set_summary = perf_df.groupby("Feature Set", as_index=False)["ROC-AUC"].mean()
        fig_set = px.bar(
            feature_set_summary,
            x="Feature Set",
            y="ROC-AUC",
            text="ROC-AUC",
            title="Average ROC-AUC by Feature Set"
        )
        fig_set.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_set.update_layout(height=360, yaxis_range=[0.70, 0.90])
        st.plotly_chart(fig_set, use_container_width=True)

elif page == "Prediction":
    st.title("🧠 Mortality Risk Prediction")
    st.markdown("Enter patient values on the left and review the prediction summary and patient profile on the right.")

    left_col, right_col = st.columns([0.68, 1.52], gap="large")

    defaults = {}
    for f in model_features:
        if f in df.columns and pd.api.types.is_numeric_dtype(df[f]):
            median_val = df[f].median()
            defaults[f] = float(median_val) if pd.notna(median_val) else 0.0
        else:
            defaults[f] = 0.0

    input_data = {}

    with left_col:
        st.subheader("Patient Inputs")
        st.caption("Compact numeric controls with 3 decimal precision.")

        with st.form("prediction_form", clear_on_submit=False):
            for feature in model_features:
                label = format_label(feature)
                label_col, input_col = st.columns([2.8, 0.8])

                with label_col:
                    st.markdown(f'<div class="compact-label">{label}</div>', unsafe_allow_html=True)

                with input_col:
                    if "ventilated" in feature.lower():
                        input_data[feature] = st.selectbox(
                            label,
                            options=[0, 1],
                            index=int(defaults.get(feature, 0)),
                            label_visibility="collapsed"
                        )
                    elif "diagnosis" in feature.lower():
                        input_data[feature] = st.number_input(
                            label,
                            value=float(defaults.get(feature, 0.0)),
                            step=1.0,
                            format="%.0f",
                            label_visibility="collapsed"
                        )
                    else:
                        input_data[feature] = st.number_input(
                            label,
                            value=float(defaults.get(feature, 0.0)),
                            step=0.001,
                            format="%.3f",
                            label_visibility="collapsed"
                        )

            submitted = st.form_submit_button("Predict Mortality Risk", use_container_width=True)

    with right_col:
        st.subheader("Prediction Summary")
        st.markdown('<div class="custom-card">Use the compact form on the left to generate a mortality risk estimate. The charts below help explain the model output and how the patient compares with the dataset.</div>', unsafe_allow_html=True)

        if submitted:
            try:
                input_df = pd.DataFrame([input_data])[model_features]
                input_imputed = pd.DataFrame(imputer.transform(input_df), columns=model_features)

                pred = int(model.predict(input_imputed)[0])
                prob = float(model.predict_proba(input_imputed)[0][1])
                survival_prob = 1 - prob

                st.markdown("---")
                r1, r2, r3 = st.columns(3)
                r1.metric("Mortality Probability", f"{prob * 100:.1f}%")
                r2.metric("Predicted Class", "High Risk" if pred == 1 else "Low Risk")
                r3.metric("Survival Probability", f"{survival_prob * 100:.1f}%")

                if pred == 1:
                    st.error("This patient is predicted to have a **high mortality risk**.")
                else:
                    st.success("This patient is predicted to have a **low mortality risk**.")

                tabs = st.tabs(["Risk Dashboard", "Patient vs Dataset", "Percentile Profile"])

                with tabs[0]:
                    top_g1, top_g2 = st.columns([1.2, 0.8])

                    with top_g1:
                        gauge = go.Figure(
                            go.Indicator(
                                mode="gauge+number",
                                value=prob * 100,
                                number={"suffix": "%"},
                                title={"text": "Mortality Risk"},
                                gauge={
                                    "axis": {"range": [0, 100]},
                                    "bar": {"thickness": 0.28},
                                    "steps": [
                                        {"range": [0, 35], "color": "rgba(0, 200, 83, 0.25)"},
                                        {"range": [35, 65], "color": "rgba(255, 193, 7, 0.28)"},
                                        {"range": [65, 100], "color": "rgba(244, 67, 54, 0.28)"},
                                    ],
                                    "threshold": {
                                        "line": {"width": 4},
                                        "thickness": 0.75,
                                        "value": prob * 100,
                                    },
                                },
                            )
                        )
                        gauge.update_layout(height=360, margin=dict(l=20, r=20, t=60, b=10))
                        st.plotly_chart(gauge, use_container_width=True)

                    with top_g2:
                        donut = go.Figure(
                            data=[
                                go.Pie(
                                    labels=["Survival Probability", "Mortality Probability"],
                                    values=[survival_prob, prob],
                                    hole=0.62,
                                    sort=False
                                )
                            ]
                        )
                        donut.update_layout(title="Risk Composition", height=360, margin=dict(l=20, r=20, t=60, b=10))
                        st.plotly_chart(donut, use_container_width=True)

                    risk_bar_df = pd.DataFrame({
                        "Category": ["Survival Probability", "Mortality Probability"],
                        "Probability": [survival_prob * 100, prob * 100]
                    })
                    fig_risk_bar = px.bar(
                        risk_bar_df,
                        x="Category",
                        y="Probability",
                        text="Probability",
                        title="Probability Breakdown"
                    )
                    fig_risk_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                    fig_risk_bar.update_layout(height=320, yaxis_title="Percent", yaxis_range=[0, 100])
                    st.plotly_chart(fig_risk_bar, use_container_width=True)

                with tabs[1]:
                    if hasattr(model, "feature_importances_"):
                        importance_df = pd.DataFrame({
                            "Feature": model_features,
                            "Importance": model.feature_importances_
                        }).sort_values("Importance", ascending=False)
                        compare_features = importance_df["Feature"].head(6).tolist()
                    else:
                        compare_features = model_features[:6]

                    compare_df = pd.DataFrame({
                        "Feature": [format_label(f) for f in compare_features],
                        "Patient Value": [input_data[f] for f in compare_features],
                        "Dataset Median": [float(df[f].median()) if pd.notna(df[f].median()) else 0.0 for f in compare_features]
                    })

                    compare_long = compare_df.melt(
                        id_vars="Feature",
                        value_vars=["Patient Value", "Dataset Median"],
                        var_name="Type",
                        value_name="Value"
                    )
                    fig_compare = px.bar(
                        compare_long,
                        x="Feature",
                        y="Value",
                        color="Type",
                        barmode="group",
                        title="Patient Values vs Dataset Median (Top Features)"
                    )
                    fig_compare.update_layout(height=430)
                    st.plotly_chart(fig_compare, use_container_width=True)

                    fig_patient_line = px.line(
                        compare_long,
                        x="Feature",
                        y="Value",
                        color="Type",
                        markers=True,
                        title="Profile Trend Across Key Features"
                    )
                    fig_patient_line.update_layout(height=380)
                    st.plotly_chart(fig_patient_line, use_container_width=True)

                with tabs[2]:
                    if hasattr(model, "feature_importances_"):
                        importance_df = pd.DataFrame({
                            "Feature": model_features,
                            "Importance": model.feature_importances_
                        }).sort_values("Importance", ascending=False)
                        pct_features = importance_df["Feature"].head(8).tolist()
                    else:
                        pct_features = model_features[:8]

                    percentile_rows = []
                    for feature in pct_features:
                        percentile_rows.append({
                            "Feature": format_label(feature),
                            "Percentile": compute_percentile(df[feature], input_data[feature])
                        })

                    pct_df = pd.DataFrame(percentile_rows).sort_values("Percentile", ascending=True)

                    fig_pct = px.bar(
                        pct_df,
                        x="Percentile",
                        y="Feature",
                        orientation="h",
                        text="Percentile",
                        title="Patient Percentile Rank Within Dataset"
                    )
                    fig_pct.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                    fig_pct.update_layout(height=430, xaxis_range=[0, 100])
                    st.plotly_chart(fig_pct, use_container_width=True)

                    st.dataframe(pct_df.sort_values("Percentile", ascending=False), use_container_width=True)

            except Exception as e:
                st.error("Prediction failed.")
                st.exception(e)
