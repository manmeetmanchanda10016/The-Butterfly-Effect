
import os
import html
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

MODEL_PATH = "butterfly_effect_model.pkl"
EVALUATION_PATH = "evaluation_results.csv"

MAE = 82.27740628573008
RMSE = 374.2801350831353
R2 = 0.9244857690274642

st.set_page_config(
    page_title="The Butterfly Effect | Sales Intelligence",
    page_icon="🦋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- DESIGN --------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');

:root {
    --ink: #203451;
    --heading: #1d3557;
    --muted: #526783;
    --blue: #377df6;
    --teal: #0eaa91;
}

.stApp {
    background:
        radial-gradient(ellipse at 78% -8%, rgba(160,195,245,.35), transparent 36%),
        radial-gradient(ellipse at 3% 25%, rgba(190,221,248,.45), transparent 32%),
        linear-gradient(180deg, #edf3fc, #e8f0fc 48%, #f0f5fd);
    font-family: 'DM Sans', sans-serif;
    color: #203451;
}

[data-testid="stHeader"] {
    background: rgba(237,243,252,.92);
}

/* SIDEBAR BACKGROUND */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #dceaff 0%,
        #e5efff 50%,
        #dceafa 100%
    ) !important;
    border-right: 1px solid #c5d7f2 !important;
}

[data-testid="stSidebar"] > div {
    background: transparent !important;
}

/* GLOBAL TEXT */

.stApp p,
.stApp li,
.stApp label,
.stApp small,
[data-testid="stMarkdownContainer"] p,
[data-testid="stCaptionContainer"] p {
    color: #344761;
}

h1, h2, h3, h4, h5, h6 {
    color: #1d3557 !important;
    font-family: 'Manrope', sans-serif;
}

/* SIDEBAR HEADINGS AND LABELS */

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] * {
    color: #203451 !important;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #526783 !important;
}

[data-testid="stSidebar"] h3 {
    font-weight: 800 !important;
}

/* CRITICAL: DROPDOWN SELECTED TEXT FIX */

[data-testid="stSidebar"] [data-baseweb="select"] {
    color: #203451 !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border: 1px solid #bfd1ec !important;
    border-radius: 10px !important;
}

/* Selected value inside Streamlit selectbox */
[data-testid="stSidebar"] [data-baseweb="select"] div,
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] input,
[data-testid="stSidebar"] [data-baseweb="select"] p,
[data-testid="stSidebar"] [data-baseweb="select"] [role="button"] {
    color: #203451 !important;
    -webkit-text-fill-color: #203451 !important;
    opacity: 1 !important;
}

/* Base Web selected-value container */
[data-testid="stSidebar"] [data-baseweb="select"] [class*="singleValue"],
[data-testid="stSidebar"] [data-baseweb="select"] [class*="ValueContainer"],
[data-testid="stSidebar"] [data-baseweb="select"] [class*="placeholder"],
[data-testid="stSidebar"] [data-baseweb="select"] [class*="control"] {
    color: #203451 !important;
    -webkit-text-fill-color: #203451 !important;
    opacity: 1 !important;
}

/* Dropdown menu and options */
[data-baseweb="popover"],
[data-baseweb="menu"],
[role="listbox"],
[role="option"] {
    background-color: #ffffff !important;
    color: #203451 !important;
}

[data-baseweb="popover"] *,
[role="listbox"] *,
[role="option"] * {
    color: #203451 !important;
    -webkit-text-fill-color: #203451 !important;
    opacity: 1 !important;
}

/* NUMBER INPUTS */

[data-testid="stSidebar"] [data-baseweb="input"] > div {
    background: #ffffff !important;
    border: 1px solid #bfd1ec !important;
    border-radius: 10px !important;
}

[data-testid="stSidebar"] [data-baseweb="input"] input {
    background: #ffffff !important;
    color: #203451 !important;
    -webkit-text-fill-color: #203451 !important;
    opacity: 1 !important;
}

[data-testid="stSidebar"] [data-testid="stNumberInput"] button {
    color: #ffffff !important;
}

/* SLIDER */

[data-testid="stSidebar"] [data-testid="stSlider"] label {
    color: #203451 !important;
}

[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: #203451 !important;
    font-weight: 700 !important;
}

/* BRAND */

.brand {
    text-align: center;
    padding: 12px 5px 22px;
    margin-bottom: 20px;
    border-bottom: 1px solid #bfd2ee;
}

.butterfly {
    font-size: 44px;
    filter: drop-shadow(0 5px 10px rgba(75,125,240,.24));
}

.brand h2 {
    color: #203451 !important;
    font-size: 18px;
    margin: 5px 0 2px;
    font-weight: 800;
}

.brand p {
    color: #536b89 !important;
    font-size: 10px;
    letter-spacing: 2px;
}

/* HERO */

.hero {
    padding: 28px 30px;
    border: 1px solid #c3d6f3;
    border-radius: 22px;
    margin-bottom: 18px;
    background:
        radial-gradient(circle at 88% 5%, rgba(255,255,255,.95), transparent 24%),
        radial-gradient(circle at 75% 100%, rgba(163,224,239,.42), transparent 34%),
        linear-gradient(112deg, #dce9fc, #edf1fc 52%, #dcf3fa);
    box-shadow: 0 12px 35px rgba(78,120,190,.09);
}

.eyebrow {
    color: #2865b5 !important;
    text-transform: uppercase;
    letter-spacing: 2.7px;
    font-size: 10px;
    font-weight: 800;
}

.hero h1 {
    color: #1e3555 !important;
    font-size: clamp(27px, 3vw, 42px);
    line-height: 1.15;
    margin: 9px 0;
    font-weight: 800;
}

.gradient-text {
    background: linear-gradient(90deg, #3679ed, #6875e8, #12a99a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #405873 !important;
    font-size: 13px;
}

.pill {
    display: inline-block;
    margin-top: 14px;
    border: 1px solid #c0d5f4;
    background: rgba(255,255,255,.8);
    color: #315e9b !important;
    border-radius: 99px;
    padding: 6px 11px;
    font-size: 10px;
    font-weight: 800;
}

/* METRIC CARDS */

.metric-card {
    min-height: 125px;
    padding: 18px;
    border-radius: 18px;
    border: 1px solid #d5e2f5;
    background: #ffffff;
    box-shadow: 0 8px 24px rgba(49,84,145,.055);
    transition: transform .18s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
}

.metric-top {
    display: flex;
    align-items: center;
    gap: 8px;
}

.metric-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
}

.metric-label {
    color: #526782 !important;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: .7px;
}

.metric-value {
    color: #203451 !important;
    font-size: 25px;
    font-weight: 800;
    margin-top: 10px;
    overflow-wrap: anywhere;
}

.metric-note {
    color: #647792 !important;
    font-size: 10px;
    margin-top: 4px;
}

/* PREDICTION RESULT */

.result {
    border: 1px solid #a9e2d6;
    background: linear-gradient(105deg, #e5f8f1, #eef8ff 75%, #eff0ff);
    border-radius: 17px;
    padding: 19px 22px;
    margin-top: 15px;
}

.result-label {
    color: #117e70 !important;
    font-size: 10px;
    letter-spacing: 1.2px;
    font-weight: 800;
}

.result-number {
    color: #078d7b !important;
    font-size: 34px;
    font-weight: 800;
    margin-top: 4px;
}

.result-description {
    color: #426d67 !important;
    font-size: 11px;
    margin-top: 3px;
}

/* PANELS */

.panel {
    background: #ffffff;
    border: 1px solid #d9e4f4;
    border-radius: 19px;
    padding: 19px;
    margin-top: 17px;
    box-shadow: 0 9px 28px rgba(49,84,145,.055);
}

.panel h3 {
    color: #233b5d !important;
    font-size: 16px;
    margin: 0 0 4px;
    font-weight: 800;
}

.panel-caption {
    color: #5f718b !important;
    font-size: 11px;
    margin-bottom: 12px;
}

/* BUTTON */

.stButton > button {
    width: 100%;
    border: 0;
    border-radius: 12px;
    color: #ffffff !important;
    font-weight: 800;
    padding: .72rem 1rem;
    background: linear-gradient(100deg, #367df4, #6288f1 60%, #26b8b0);
    box-shadow: 0 7px 17px rgba(55,125,246,.20);
}

.stButton > button p,
.stButton > button span {
    color: #ffffff !important;
}

.stButton > button:hover {
    color: #ffffff !important;
    box-shadow: 0 10px 23px rgba(55,125,246,.25);
}

/* MAIN AREA INPUTS */

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background: #ffffff !important;
    border-color: #bfd1ec !important;
    border-radius: 10px !important;
}

div[data-baseweb="select"] span,
div[data-baseweb="select"] input,
div[data-baseweb="input"] input {
    color: #203451 !important;
    -webkit-text-fill-color: #203451 !important;
}

/* STREAMLIT METRICS */

[data-testid="stMetric"] {
    background: #f7faff;
    border: 1px solid #d9e5f5;
    padding: 13px;
    border-radius: 13px;
}

[data-testid="stMetricLabel"] p {
    color: #526782 !important;
}

[data-testid="stMetricValue"] {
    color: #203451 !important;
}

/* CAPTIONS AND FOOTER */

.stCaption,
[data-testid="stCaptionContainer"] p {
    color: #526783 !important;
}

.footer {
    text-align: center;
    color: #526783 !important;
    font-size: 10px;
    padding: 25px 0 8px;
    line-height: 1.8;
}

hr {
    border-color: #cbdcf1;
}
</style>
""", unsafe_allow_html=True)


# -------------------- LOAD MODEL --------------------

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()


# -------------------- PRODUCT FAMILIES --------------------

FAMILIES = [
    "AUTOMOTIVE", "BABY CARE", "BEAUTY", "BEVERAGES", "BOOKS",
    "BREAD/BAKERY", "CELEBRATION", "CLEANING", "DAIRY", "DELI",
    "EGGS", "FROZEN FOODS", "GROCERY I", "GROCERY II", "HARDWARE",
    "HOME AND KITCHEN I", "HOME AND KITCHEN II", "HOME APPLIANCES",
    "HOME CARE", "LADIESWEAR", "LAWN AND GARDEN", "LINGERIE",
    "LIQUOR,WINE,BEER", "MAGAZINES", "MEATS", "PERSONAL CARE",
    "PET SUPPLIES", "PLAYERS AND ELECTRONICS", "POULTRY",
    "PREPARED FOODS", "PRODUCE", "SCHOOL AND OFFICE SUPPLIES",
    "SEAFOOD"
]


# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="butterfly">🦋</div>
        <h2>The Butterfly Effect</h2>
        <p>SALES INTELLIGENCE</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ✨ Forecast Inputs")
    st.caption("Adjust the details to estimate demand.")

    store_nbr = st.number_input(
        "Store Number",
        min_value=1,
        max_value=54,
        value=1,
        step=1
    )

    family = st.selectbox(
        "Product Family",
        FAMILIES,
        index=FAMILIES.index("GROCERY I")
    )

    onpromotion = st.number_input(
        "Items on Promotion",
        min_value=0,
        value=0,
        step=1
    )

    day_of_week = st.selectbox(
        "Day of Week",
        list(range(7)),
        format_func=lambda x: [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"
        ][x]
    )

    month = st.selectbox(
        "Month",
        list(range(1, 13)),
        format_func=lambda x: pd.Timestamp(
            2000, x, 1
        ).strftime("%B")
    )

    year = st.selectbox(
        "Year",
        list(range(2013, 2028)),
        index=4
    )

    day_of_month = st.slider(
        "Day of Month", 1, 31, 15
    )

    lag_1 = st.number_input(
        "Sales Yesterday (lag_1)",
        min_value=0.0,
        value=100.0,
        step=10.0
    )

    lag_7 = st.number_input(
        "Sales 7 Days Ago (lag_7)",
        min_value=0.0,
        value=100.0,
        step=10.0
    )

    predict_clicked = st.button("🦋 Predict Sales")

    st.markdown("---")
    st.caption("Forecasts are estimates, not guarantees.")


# -------------------- HEADER --------------------

st.markdown("""
<div class="hero">
    <div class="eyebrow">SMARTER RETAIL · CLEARER INSIGHTS</div>
    <h1>The <span class="gradient-text">Butterfly Effect</span> 🦋</h1>
    <p>Sales forecasting that turns small signals into smarter decisions.</p>
    <span class="pill">✦ LIGHTGBM POWERED FORECASTING</span>
</div>
""", unsafe_allow_html=True)


if model is None:
    st.error(
        "Model file not found. Place butterfly_effect_model.pkl "
        "in the same folder as app.py."
    )
    st.stop()


# -------------------- PREDICTION --------------------

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if predict_clicked:
    input_df = pd.DataFrame([{
        "store_nbr": int(store_nbr),
        "family": family,
        "onpromotion": int(onpromotion),
        "day_of_week": int(day_of_week),
        "month": int(month),
        "year": int(year),
        "day_of_month": int(day_of_month),
        "lag_1": float(lag_1),
        "lag_7": float(lag_7)
    }])

    try:
        prediction = float(model.predict(input_df)[0])
        st.session_state.prediction = max(0.0, prediction)
    except Exception as e:
        st.error(
            "Prediction failed. Check that the input columns and "
            f"model pipeline match. Details: {e}"
        )

prediction = st.session_state.prediction

predicted_value = (
    f"{prediction:,.2f}"
    if prediction is not None else "—"
)

predicted_note = (
    f"{family.title()} · Store {int(store_nbr)}"
    if prediction is not None else "Choose inputs and predict"
)


# -------------------- KPI CARDS --------------------

c1, c2, c3, c4 = st.columns(4)

kpis = [
    (c1, "PREDICTED SALES", predicted_value,
     predicted_note, "#377df6"),
    (c2, "MEAN ABSOLUTE ERROR", f"{MAE:,.2f}",
     "Lower is better", "#0eaa91"),
    (c3, "ROOT MEAN SQUARED ERROR", f"{RMSE:,.2f}",
     "Lower is better", "#8175e9"),
    (c4, "R² SCORE", f"{R2:.4f}",
     "Higher is better", "#0eaa91")
]

for col, label, value, note, color in kpis:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-top">
                <div class="metric-dot" style="background:{color}"></div>
                <div class="metric-label">{label}</div>
            </div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{html.escape(note)}</div>
        </div>
        """, unsafe_allow_html=True)


if prediction is not None:
    st.markdown(f"""
    <div class="result">
        <div class="result-label">
            ✦ YOUR SALES FORECAST · {html.escape(family.title())}
        </div>
        <div class="result-number">{prediction:,.2f}</div>
        <div class="result-description">
            Estimated sales for Store {int(store_nbr)} using the selected inputs.
        </div>
    </div>
    """, unsafe_allow_html=True)


# -------------------- SALES GRAPH --------------------

st.markdown("""
<div class="panel">
    <h3>📈 Sales Forecast Analysis</h3>
    <div class="panel-caption">
        Actual and predicted sales from your test observations
    </div>
</div>
""", unsafe_allow_html=True)

if os.path.exists(EVALUATION_PATH):
    try:
        eval_df = pd.read_csv(EVALUATION_PATH)
        required = {"Actual Sales", "Predicted Sales"}

        if required.issubset(eval_df.columns) and not eval_df.empty:
            n_options = [
                n for n in [50, 100, 200, 500]
                if n <= len(eval_df)
            ]

            if not n_options:
                n_options = [len(eval_df)]

            n = st.selectbox(
                "Test observations to display",
                n_options,
                index=min(2, len(n_options) - 1)
            )

            chart_df = eval_df[
                ["Actual Sales", "Predicted Sales"]
            ].head(n).copy()

            chart_df["Observation"] = np.arange(
                1, len(chart_df) + 1
            )

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=chart_df["Observation"],
                y=chart_df["Actual Sales"],
                name="Actual sales",
                mode="lines",
                line=dict(color="#377df6", width=2.7),
                fill="tozeroy",
                fillcolor="rgba(55,125,246,.07)"
            ))

            fig.add_trace(go.Scatter(
                x=chart_df["Observation"],
                y=chart_df["Predicted Sales"],
                name="Predicted sales",
                mode="lines",
                line=dict(color="#0eaa91", width=2.7)
            ))

            fig.update_layout(
                template="plotly_white",
                height=360,
                paper_bgcolor="rgba(255,255,255,0)",
                plot_bgcolor="#f9fbff",
                margin=dict(l=12, r=12, t=30, b=12),
                legend=dict(orientation="h", y=1.12, x=0),
                xaxis_title="Test observation",
                yaxis_title="Sales",
                font=dict(color="#344761"),
                hovermode="x unified"
            )

            fig.update_xaxes(gridcolor="#e2eaf5")
            fig.update_yaxes(gridcolor="#e2eaf5")

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning(
                "evaluation_results.csv must contain columns named "
                "'Actual Sales' and 'Predicted Sales'."
            )

    except Exception as e:
        st.warning(f"Could not read evaluation_results.csv: {e}")

else:
    st.caption(
        "Input-signal preview. Add evaluation_results.csv to show "
        "actual-versus-predicted test results."
    )

    input_signals = pd.DataFrame({
        "Input": [
            "Sales 7 days ago",
            "Sales yesterday",
            "Items on promotion"
        ],
        "Value": [
            float(lag_7),
            float(lag_1),
            float(onpromotion)
        ]
    })

    fig = px.bar(
        input_signals,
        x="Value",
        y="Input",
        orientation="h",
        color="Input",
        color_discrete_map={
            "Sales 7 days ago": "#8b83f6",
            "Sales yesterday": "#377df6",
            "Items on promotion": "#0eaa91"
        },
        text="Value"
    )

    fig.update_traces(
        texttemplate="%{text:,.1f}",
        textposition="outside",
        cliponaxis=False
    )

    fig.update_layout(
        template="plotly_white",
        height=300,
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#f9fbff",
        showlegend=False,
        margin=dict(l=12, r=45, t=12, b=10),
        font=dict(color="#344761"),
        xaxis_title="Input value",
        yaxis_title=""
    )

    fig.update_xaxes(gridcolor="#e2eaf5")
    fig.update_yaxes(gridcolor="#e2eaf5")

    st.plotly_chart(fig, use_container_width=True)


# -------------------- FEATURE IMPORTANCE --------------------

left, right = st.columns([1.12, 0.88])

with left:
    st.markdown("""
    <div class="panel">
        <h3>✦ What Drives Sales?</h3>
        <div class="panel-caption">
            Most influential features in the trained model
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        preprocessor = model.named_steps["preprocessor"]
        regressor = model.named_steps["regressor"]
        importances = getattr(
            regressor, "feature_importances_", None
        )

        if importances is not None:
            feature_names = preprocessor.get_feature_names_out()

            if len(feature_names) == len(importances):
                importance_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importances
                })

                importance_df["Feature"] = (
                    importance_df["Feature"]
                    .str.replace("num__", "", regex=False)
                    .str.replace("cat__", "", regex=False)
                )

                importance_df = importance_df.sort_values(
                    "Importance"
                ).tail(10)

                fig = px.bar(
                    importance_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale=[
                        "#9bc7ff", "#5d91f5", "#0eaa91"
                    ]
                )

                fig.update_layout(
                    template="plotly_white",
                    height=350,
                    paper_bgcolor="rgba(255,255,255,0)",
                    plot_bgcolor="#f9fbff",
                    coloraxis_showscale=False,
                    margin=dict(l=5, r=15, t=8, b=8),
                    font=dict(color="#344761"),
                    xaxis_title="Importance",
                    yaxis_title=""
                )

                fig.update_xaxes(gridcolor="#e2eaf5")
                fig.update_yaxes(gridcolor="#e2eaf5")

                st.plotly_chart(fig, use_container_width=True)

            else:
                st.info(
                    "Feature names do not match the model importance values."
                )
        else:
            st.info(
                "This model does not provide built-in feature importance."
            )

    except Exception as e:
        st.info(f"Feature importance unavailable: {e}")


# -------------------- MODEL PERFORMANCE --------------------

with right:
    st.markdown("""
    <div class="panel">
        <h3>🏆 Model Performance</h3>
        <div class="panel-caption">
            Recorded LightGBM evaluation metrics
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.metric("Mean Absolute Error (MAE)", f"{MAE:.2f}")
    st.metric("Root Mean Squared Error (RMSE)", f"{RMSE:.2f}")
    st.metric("R² Score", f"{R2:.4f}")

    st.markdown("""
    <div style="
        background:linear-gradient(120deg,#e5efff,#e4f7f1);
        border:1px solid #cfdef2;
        border-radius:15px;
        padding:17px;
        margin-top:14px;
    ">
        <div style="color:#526d8b;font-size:10px;letter-spacing:1px;font-weight:800">
            MODEL IN USE
        </div>
        <div style="color:#315c9c;font-size:23px;font-weight:800;margin-top:4px">
            LightGBM
        </div>
        <div style="color:#526d8b;font-size:11px">
            Gradient-boosted decision trees
        </div>
    </div>
    """, unsafe_allow_html=True)


# -------------------- FOOTER --------------------

st.markdown("""
<div class="footer">
    🦋 THE BUTTERFLY EFFECT · SALES INTELLIGENCE<br>
    Demand forecasting · Data-driven decisions · Smarter retail planning<br>
    Evaluation metrics reflect the recorded experiment.
</div>
""", unsafe_allow_html=True)
