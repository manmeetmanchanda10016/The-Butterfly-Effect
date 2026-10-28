
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

from pathlib import Path

st.set_page_config(
    page_title="The Butterfly Effect",
    page_icon="🦋",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = Path("butterfly_effect_model.pkl")
EVALUATION_PATH = Path("evaluation_results.csv")

MAE = 82.27740628573008
RMSE = 374.2801350831353
R2 = 0.9244857690274642

NAVY = "#142B4A"
DARK = "#263B53"
BLUE = "#3978D4"
TEAL = "#079F91"
BACKGROUND = "#EDF4FD"
WHITE = "#FFFFFF"
MUTED = "#526782"

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #E5F0FF 0%, #F5F8FF 100%);
        color: #142B4A;
    }

    [data-testid="stHeader"] {
        background: rgba(237, 244, 253, 0.95);
    }

    [data-testid="stSidebar"] {
        background: #DFECFF;
        border-right: 1px solid #C6D8F2;
    }

    [data-testid="stSidebar"] * {
        color: #142B4A !important;
    }

    [data-testid="stAppViewContainer"] *,
    [data-testid="stMain"] * {
        color-scheme: light;
    }

    h1, h2, h3, h4, p, label, span, div {
        --text-color: #142B4A;
    }

    h1, h2, h3, h4 {
        color: #142B4A !important;
        font-weight: 700 !important;
    }

    p, label, .stMarkdown, .stCaption {
        color: #263B53;
    }

    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label {
        color: #263B53 !important;
        font-weight: 600 !important;
    }

    [data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border-color: #B6CBE8 !important;
    }

    [data-baseweb="select"] *,
    [data-baseweb="input"] *,
    [data-baseweb="base-input"] * {
        color: #142B4A !important;
        -webkit-text-fill-color: #142B4A !important;
    }

    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [role="listbox"] {
        background-color: #FFFFFF !important;
    }

    [role="option"] {
        background-color: #FFFFFF !important;
        color: #142B4A !important;
    }

    [role="option"]:hover {
        background-color: #E5F0FF !important;
    }

    input, textarea {
        background-color: #FFFFFF !important;
        color: #142B4A !important;
        -webkit-text-fill-color: #142B4A !important;
        border-radius: 8px !important;
    }

    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #D5E2F3;
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 5px 18px rgba(31, 64, 110, 0.06);
    }

    [data-testid="stMetricLabel"] p {
        color: #526782 !important;
        font-size: 14px !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricValue"] {
        color: #142B4A !important;
        font-weight: 750 !important;
    }

    [data-testid="stMetricDelta"] {
        color: #079F91 !important;
    }

    div.stButton > button {
        background: #3978D4;
        color: #FFFFFF !important;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1.2rem;
        font-weight: 700;
        width: 100%;
    }

    div.stButton > button p,
    div.stButton > button span {
        color: #FFFFFF !important;
    }

    div.stButton > button:hover {
        background: #245DB1;
        border: none;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.94);
        border-radius: 16px;
        border-color: #D5E2F3 !important;
    }

    hr {
        border-color: #D5E2F3;
    }

    [data-testid="stAlert"] {
        color: #142B4A !important;
    }

    footer {
        visibility: hidden;
    }

    #MainMenu {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def style_chart(fig, height=400):
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color=NAVY
        ),
        title_font=dict(
            family="Arial, sans-serif",
            size=18,
            color=NAVY
        ),
        legend=dict(
            font=dict(color=DARK, size=12),
            bgcolor="rgba(255,255,255,0.85)"
        ),
        margin=dict(l=35, r=25, t=55, b=45),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            font=dict(color=NAVY, size=13),
            bordercolor="#B6CBE8"
        )
    )

    fig.update_xaxes(
        title_font=dict(color=DARK, size=12),
        tickfont=dict(color=DARK, size=11),
        gridcolor="#E3EAF4",
        linecolor="#B6CBE8",
        zerolinecolor="#D5E2F3"
    )

    fig.update_yaxes(
        title_font=dict(color=DARK, size=12),
        tickfont=dict(color=DARK, size=11),
        gridcolor="#E3EAF4",
        linecolor="#B6CBE8",
        zerolinecolor="#D5E2F3"
    )

    return fig


def metric_card_label():
    st.markdown(
        """
        <div style="
            color:#526782;
            font-size:13px;
            font-weight:600;
            margin-bottom:4px;">
        </div>
        """,
        unsafe_allow_html=True
    )


try:
    if not MODEL_PATH.exists():
        st.error(
            "Model file not found. Make sure "
            "'butterfly_effect_model.pkl' is in the project folder."
        )
        st.stop()

    model = load_model()

except Exception as e:
    st.error(f"Unable to load the trained model: {e}")
    st.stop()


st.sidebar.markdown(
    """
    <div style="text-align:center; padding:12px 0 20px;">
        <div style="font-size:42px;">🦋</div>
        <h2 style="margin:0; color:#142B4A !important;">
            The Butterfly Effect
        </h2>
        <p style="font-size:13px; color:#526782 !important;">
            Retail Sales Forecasting
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.divider()
st.sidebar.subheader("⚙️ Forecast Configuration")

store_nbr = st.sidebar.number_input(
    "Store Number",
    min_value=1,
    max_value=100,
    value=1,
    step=1
)

family = st.sidebar.selectbox(
    "Product Family",
    [
        "AUTOMOTIVE",
        "BABY CARE",
        "BEAUTY",
        "BEVERAGES",
        "BOOKS",
        "BREAD/BAKERY",
        "CELEBRATION",
        "CLEANING",
        "DAIRY",
        "DELI",
        "EGGS",
        "FROZEN FOODS",
        "GROCERY I",
        "GROCERY II",
        "HARDWARE",
        "HOME AND KITCHEN I",
        "HOME AND KITCHEN II",
        "HOME APPLIANCES",
        "HOME CARE",
        "LADIESWEAR",
        "LAWN AND GARDEN",
        "LINGERIE",
        "LIQUOR,WINE,BEER",
        "MAGAZINES",
        "MEATS",
        "PERSONAL CARE",
        "PET SUPPLIES",
        "PLAYERS AND ELECTRONICS",
        "POULTRY",
        "PREPARED FOODS",
        "PRODUCE",
        "SCHOOL AND OFFICE SUPPLIES",
        "SEAFOOD"
    ],
    index=12
)

onpromotion = st.sidebar.number_input(
    "Items on Promotion",
    min_value=0,
    value=25,
    step=1
)

day_of_week = st.sidebar.selectbox(
    "Day of Week",
    options=list(range(7)),
    index=1,
    format_func=lambda x: [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ][x]
)

month = st.sidebar.selectbox(
    "Month",
    options=list(range(1, 13)),
    index=10,
    format_func=lambda x: pd.Timestamp(
        year=2024, month=x, day=1
    ).strftime("%B")
)

year = st.sidebar.number_input(
    "Year",
    min_value=2013,
    max_value=2035,
    value=2017,
    step=1
)

day_of_month = st.sidebar.number_input(
    "Day of Month",
    min_value=1,
    max_value=31,
    value=15,
    step=1
)

lag_1 = st.sidebar.number_input(
    "Previous Day Sales (lag_1)",
    min_value=0.0,
    value=2500.0,
    step=100.0
)

lag_7 = st.sidebar.number_input(
    "Sales 7 Days Ago (lag_7)",
    min_value=0.0,
    value=2200.0,
    step=100.0
)

st.sidebar.divider()
st.sidebar.caption(
    "Adjust the input values to explore a sales forecast."
)


st.markdown(
    """
    <div style="padding: 8px 0 14px;">
        <h1 style="
            font-size:36px;
            margin-bottom:4px;
            color:#142B4A !important;">
            🦋 The Butterfly Effect
        </h1>
        <p style="
            color:#526782 !important;
            font-size:16px;">
            Intelligent Retail Sales Forecasting Dashboard
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown("### 📊 Model Performance")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Mean Absolute Error (MAE)", f"{MAE:.2f}")

with c2:
    st.metric("Root Mean Squared Error (RMSE)", f"{RMSE:.2f}")

with c3:
    st.metric("R² Score", f"{R2:.4f}")


st.markdown("### 🎯 Generate a Sales Forecast")

with st.container(border=True):
    st.markdown(
        "Review the selected inputs and generate a prediction."
    )

    predict_clicked = st.button(
        "🔮 Predict Retail Sales",
        use_container_width=True
    )

    if predict_clicked:
        input_data = pd.DataFrame(
            [{
                "store_nbr": int(store_nbr),
                "family": family,
                "onpromotion": int(onpromotion),
                "day_of_week": int(day_of_week),
                "month": int(month),
                "year": int(year),
                "day_of_month": int(day_of_month),
                "lag_1": float(lag_1),
                "lag_7": float(lag_7)
            }]
        )

        try:
            prediction = float(model.predict(input_data)[0])

            st.session_state["last_prediction"] = prediction
            st.session_state["last_input"] = input_data

        except Exception as e:
            st.error(
                "Prediction failed. Check that the input feature "
                f"types match the trained model. Details: {e}"
            )

    if "last_prediction" in st.session_state:
        prediction = st.session_state["last_prediction"]

        st.markdown(
            f"""
            <div style="
                background:linear-gradient(135deg,#FFFFFF,#E8F5FF);
                border:1px solid #BFD8F5;
                border-radius:16px;
                padding:24px;
                margin-top:12px;">
                <p style="
                    color:#526782;
                    font-size:14px;
                    margin-bottom:5px;">
                    Predicted Retail Sales
                </p>
                <h2 style="
                    color:#079F91 !important;
                    font-size:34px;
                    margin:0;">
                    {prediction:,.2f}
                </h2>
                <p style="
                    color:#526782;
                    font-size:12px;
                    margin-top:8px;">
                    Prediction generated by the trained LightGBM pipeline
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


st.markdown("### 📈 Actual vs Predicted Sales")

if EVALUATION_PATH.exists():
    try:
        evaluation_df = pd.read_csv(EVALUATION_PATH)

        required_columns = {"Actual Sales", "Predicted Sales"}

        if required_columns.issubset(evaluation_df.columns):
            evaluation_df = evaluation_df.copy()
            evaluation_df["Observation"] = np.arange(
                1, len(evaluation_df) + 1
            )

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=evaluation_df["Observation"],
                    y=evaluation_df["Actual Sales"],
                    mode="lines",
                    name="Actual Sales",
                    line=dict(color=BLUE, width=2)
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=evaluation_df["Observation"],
                    y=evaluation_df["Predicted Sales"],
                    mode="lines",
                    name="Predicted Sales",
                    line=dict(color=TEAL, width=2)
                )
            )

            fig.update_layout(
                title="Actual Sales vs Predicted Sales",
                xaxis_title="Observation",
                yaxis_title="Sales"
            )

            fig = style_chart(fig, height=380)
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info(
                "The evaluation CSV must contain 'Actual Sales' "
                "and 'Predicted Sales' columns."
            )

    except Exception as e:
        st.warning(f"Could not load evaluation results: {e}")

else:
    st.info(
        "Add evaluation_results.csv with 'Actual Sales' and "
        "'Predicted Sales' columns to display the evaluation graph."
    )


st.markdown("### ✦ What Drives Sales?")
st.caption(
    "Feature importance from the trained model. Longer bars indicate "
    "greater importance within this model."
)

try:
    regressor = model.named_steps["regressor"]
    preprocessor = model.named_steps.get("preprocessor")

    importances = regressor.feature_importances_

    if preprocessor is not None and hasattr(
        preprocessor, "get_feature_names_out"
    ):
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = [
                f"Feature {i + 1}"
                for i in range(len(importances))
            ]
    else:
        feature_names = [
            "store_nbr",
            "family",
            "onpromotion",
            "day_of_week",
            "month",
            "year",
            "day_of_month",
            "lag_1",
            "lag_7"
        ]

    if len(feature_names) != len(importances):
        feature_names = [
            f"Feature {i + 1}"
            for i in range(len(importances))
        ]

    importance_df = pd.DataFrame({
        "Feature": [
            str(name).replace("num__", "").replace("cat__", "")
            for name in feature_names
        ],
        "Importance": importances
    })

    importance_df = importance_df.sort_values(
        "Importance", ascending=True
    ).tail(15)

    fig_importance = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale=["#9CC7FF", "#3978D4", "#079F91"]
    )

    fig_importance.update_layout(
        title="Top Features Influencing Sales Predictions",
        xaxis_title="Importance",
        yaxis_title="Feature",
        coloraxis_showscale=False
    )

    fig_importance.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Importance: %{x:,.2f}"
            "<extra></extra>"
        )
    )

    fig_importance = style_chart(
        fig_importance,
        height=max(350, len(importance_df) * 32)
    )

    st.plotly_chart(
        fig_importance,
        use_container_width=True
    )

except Exception as e:
    st.warning(
        "Feature importance is unavailable. Check that your saved "
        f"pipeline contains a fitted regressor. Details: {e}"
    )


st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; padding:10px 0 18px;">
        <p style="
            color:#526782 !important;
            font-size:13px;">
            🦋 The Butterfly Effect · Retail Sales Forecasting
        </p>
        <p style="
            color:#526782 !important;
            font-size:12px;">
            Built with Python, LightGBM, Plotly and Streamlit
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
