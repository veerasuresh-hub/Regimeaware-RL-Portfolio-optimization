import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="RL Portfolio Decision Intelligence Dashboard",
    layout="wide"
)

st.markdown("""
<style>

div.stButton > button {
    font-size: 20px !important;
    font-weight: 800 !important;
    height: 52px !important;
    border-radius: 10px !important;
    border: 2px solid #334155 !important;
    background-color: #f8fafc !important;
    color: #111827 !important;
}

div.stButton > button:hover {
    background-color: #e0f2fe !important;
    color: #0f172a !important;
    border: 2px solid #0284c7 !important;
}

section[data-testid="stSidebar"] {
    width: 230px !important;
}

section[data-testid="stSidebar"] * {
    font-size: 17px !important;
    font-weight: 700 !important;
}

.decision-box {
    background-color: #fff3e6;
    color: #111827;
    padding: 32px 36px;
    border-radius: 18px;
    border-left: 12px solid #f97316;
    font-size: 32px;
    font-weight: 800;
    line-height: 1.7;
    box-shadow: 0px 5px 16px rgba(0,0,0,0.16);
    margin-bottom: 22px;
}

.decision-year {
    color: #c2410c;
    font-size: 42px;
    font-weight: 900;
}

</style>
""", unsafe_allow_html=True)

st.title("📊 Regime-Aware RL Portfolio Decision Intelligence Dashboard")

DATA_PATH = "outputs/final_dashboard_decision_table.csv"

if not os.path.exists(DATA_PATH):
    st.error(
        f"Data file not found: {DATA_PATH}. "
        "Run main.py first to generate the dashboard CSV."
    )
    st.stop()

df = pd.read_csv(DATA_PATH).sort_values("Year").reset_index(drop=True)

required_cols = [
    "Year",
    "Portfolio_Return",
    "Portfolio_Value",
    "Portfolio_Drawdown"
]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

df["Year"] = df["Year"].astype(int)

numeric_cols = [
    "Portfolio_Return",
    "Portfolio_Value",
    "Portfolio_Drawdown",
    "Bull",
    "Bear",
    "Crisis",
    "Cash"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

year_list = df["Year"].tolist()

if "idx" not in st.session_state:
    st.session_state.idx = 0

if "playing" not in st.session_state:
    st.session_state.playing = False

st.sidebar.header("🎮 Simulation Controls")

if st.sidebar.button("▶ Play", use_container_width=True):
    st.session_state.playing = True

if st.sidebar.button("⏸ Pause", use_container_width=True):
    st.session_state.playing = False

if st.sidebar.button("⏮ Reset", use_container_width=True):
    st.session_state.idx = 0
    st.session_state.playing = False

speed = st.sidebar.slider(
    "Animation Speed",
    min_value=0.5,
    max_value=3.0,
    value=1.0,
    step=0.5
)

selected_year_sidebar = st.sidebar.select_slider(
    "Select Year",
    options=year_list,
    value=year_list[st.session_state.idx],
    disabled=st.session_state.playing
)

if not st.session_state.playing:
    st.session_state.idx = year_list.index(selected_year_sidebar)

if st.session_state.playing:
    st_autorefresh(
        interval=int(speed * 1000),
        key="simulation_refresh"
    )

    if st.session_state.idx < len(year_list) - 1:
        st.session_state.idx += 1
    else:
        st.session_state.playing = False

selected_year = year_list[st.session_state.idx]
row = df.iloc[st.session_state.idx]

BIG_FONT = dict(size=24, family="Arial Black", color="black")
TICK_FONT = dict(size=20, family="Arial Black", color="black")
LEGEND_FONT = dict(size=24, family="Arial Black", color="black")
TITLE_FONT = dict(size=26, family="Arial Black", color="black")


def decision_explanation(r):
    regime = r["Dominant_Regime"] if "Dominant_Regime" in r.index else "N/A"
    mode = r["Decision_Mode"] if "Decision_Mode" in r.index else "Adaptive"
    cash = r["Cash"] if "Cash" in r.index else None

    cash_text = f"{cash:.2%}" if pd.notna(cash) else "N/A"

    return (
        f"In {int(r['Year'])}, the dominant regime is {regime}. "
        f"The RL policy selected {mode} behaviour. "
        f"Cash allocation is {cash_text}, realised portfolio return is "
        f"{r['Portfolio_Return']:.2%}, and drawdown is "
        f"{r['Portfolio_Drawdown']:.2%}."
    )


st.subheader(f"Portfolio Simulation Year: {int(selected_year)}")

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("Portfolio Value", f"{row['Portfolio_Value']:.2f}x")
k2.metric("Portfolio Return", f"{row['Portfolio_Return']:.2%}")
k3.metric("Drawdown", f"{row['Portfolio_Drawdown']:.2%}")

if "Cash" in df.columns:
    k4.metric("Cash Allocation", f"{row['Cash']:.2%}")
else:
    k4.metric("Cash Allocation", "N/A")

if "Dominant_Regime" in df.columns:
    k5.metric("Dominant Regime", str(row["Dominant_Regime"]))
else:
    k5.metric("Dominant Regime", "N/A")

st.markdown("### 🧠 Decision Intelligence Note")

st.markdown(
    f"""
    <div class="decision-box">
        <span class="decision-year">{int(selected_year)}</span><br>
        {decision_explanation(row)}
    </div>
    """,
    unsafe_allow_html=True
)

left, right = st.columns([1.3, 1])

with left:
    st.markdown("### 📈 Portfolio Growth Over Time")

    fig_value = go.Figure()

    fig_value.add_trace(
        go.Scatter(
            x=df["Year"],
            y=df["Portfolio_Value"],
            mode="lines+markers",
            name="Portfolio Value",
            line=dict(width=5),
            marker=dict(size=10)
        )
    )

    fig_value.add_trace(
        go.Scatter(
            x=[selected_year],
            y=[row["Portfolio_Value"]],
            mode="markers+text",
            name="Current Year",
            marker=dict(
                size=34,
                color="red",
                line=dict(width=5, color="white")
            ),
            text=[str(selected_year)],
            textposition="top center",
            textfont=dict(size=24, color="red")
        )
    )

    fig_value.add_vline(
        x=selected_year,
        line_dash="dash",
        line_color="red",
        line_width=3
    )

    fig_value.update_layout(
        height=620,
        xaxis_title="Year",
        yaxis_title="Portfolio Value",
        hovermode="x unified",
        font=dict(size=22),
        legend=dict(font=LEGEND_FONT),
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            tickmode="array",
            tickvals=year_list,
            ticktext=[str(y) for y in year_list],
            tickangle=45,
            tickfont=TICK_FONT,
            title_font=BIG_FONT,
            range=[min(year_list) - 0.5, max(year_list) + 0.5]
        ),
        yaxis=dict(
            tickfont=TICK_FONT,
            title_font=BIG_FONT,
            range=[
                df["Portfolio_Value"].min() * 0.95,
                df["Portfolio_Value"].max() * 1.08
            ]
        )
    )

    st.plotly_chart(fig_value, use_container_width=True)

with right:
    st.markdown("### 📉 Portfolio Drawdown")

    fig_dd = go.Figure()

    fig_dd.add_trace(
        go.Scatter(
            x=df["Year"],
            y=df["Portfolio_Drawdown"],
            mode="lines+markers",
            name="Drawdown",
            fill="tozeroy",
            line=dict(width=5),
            marker=dict(size=10)
        )
    )

    fig_dd.add_trace(
        go.Scatter(
            x=[selected_year],
            y=[row["Portfolio_Drawdown"]],
            mode="markers+text",
            name="Current Year",
            marker=dict(
                size=34,
                color="red",
                line=dict(width=5, color="white")
            ),
            text=[str(selected_year)],
            textposition="bottom center",
            textfont=dict(size=24, color="red")
        )
    )

    fig_dd.add_vline(
        x=selected_year,
        line_dash="dash",
        line_color="red",
        line_width=3
    )

    fig_dd.update_layout(
        height=620,
        xaxis_title="Year",
        yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
        hovermode="x unified",
        font=dict(size=22),
        legend=dict(font=LEGEND_FONT),
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            tickmode="array",
            tickvals=year_list,
            ticktext=[str(y) for y in year_list],
            tickangle=45,
            tickfont=TICK_FONT,
            title_font=BIG_FONT,
            range=[min(year_list) - 0.5, max(year_list) + 0.5]
        ),
        yaxis=dict(
            tickfont=TICK_FONT,
            title_font=BIG_FONT
        )
    )

    st.plotly_chart(fig_dd, use_container_width=True)

st.markdown("### 🧩 Asset Allocation")

non_asset_cols = [
    "Year",
    "Portfolio_Return",
    "Portfolio_Value",
    "Portfolio_Drawdown",
    "Bull",
    "Bear",
    "Crisis",
    "Dominant_Regime",
    "Decision_Mode",
    "Decision_Explanation"
]

possible_asset_cols = [c for c in df.columns if c not in non_asset_cols]

asset_cols = []

for c in possible_asset_cols:
    converted = pd.to_numeric(df[c], errors="coerce")

    if converted.notna().sum() > 0:
        asset_cols.append(c)

if asset_cols:
    allocation = row[asset_cols].apply(
        pd.to_numeric,
        errors="coerce"
    ).fillna(0)

    allocation = allocation.reset_index()
    allocation.columns = ["Asset", "Weight"]

    fig_alloc = px.bar(
        allocation,
        x="Asset",
        y="Weight",
        text=allocation["Weight"].apply(lambda x: f"{x:.1%}"),
        title=f"Portfolio Allocation in {int(selected_year)}"
    )

    fig_alloc.update_traces(
        textposition="outside",
        textfont=dict(size=22)
    )

    fig_alloc.update_layout(
        height=560,
        font=dict(size=22),
        title_font=TITLE_FONT,
        xaxis_title="Asset",
        yaxis_title="Weight",
        xaxis=dict(
            tickfont=TICK_FONT,
            title_font=BIG_FONT
        ),
        yaxis=dict(
            tickfont=TICK_FONT,
            title_font=BIG_FONT,
            tickformat=".0%",
            range=[
                0,
                max(0.05, allocation["Weight"].max() * 1.25)
            ]
        )
    )

    st.plotly_chart(fig_alloc, use_container_width=True)

regime_cols = [
    c for c in ["Bull", "Bear", "Crisis"]
    if c in df.columns
]

if regime_cols:
    st.markdown("### 🔄 HMM Regime Probabilities")

    fig_regime = go.Figure()

    for col in regime_cols:
        fig_regime.add_trace(
            go.Scatter(
                x=df["Year"],
                y=df[col],
                mode="lines+markers",
                name=col,
                line=dict(width=4),
                marker=dict(size=9)
            )
        )

    dominant_regime = row[regime_cols].astype(float).idxmax()
    dominant_prob = float(row[dominant_regime])

    fig_regime.add_trace(
        go.Scatter(
            x=[selected_year],
            y=[dominant_prob],
            mode="markers+text",
            name="Current Dominant Regime",
            marker=dict(
                size=34,
                color="red",
                line=dict(width=5, color="white")
            ),
            text=[f"{selected_year}: {dominant_regime}"],
            textposition="top center",
            textfont=dict(size=24, color="red")
        )
    )

    fig_regime.add_vline(
        x=selected_year,
        line_dash="dash",
        line_color="red",
        line_width=3
    )

    fig_regime.update_layout(
        height=620,
        xaxis_title="Year",
        yaxis_title="Regime Probability",
        yaxis_tickformat=".0%",
        hovermode="x unified",
        font=dict(size=22),
        legend=dict(font=LEGEND_FONT),
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            tickmode="array",
            tickvals=year_list,
            ticktext=[str(y) for y in year_list],
            tickangle=45,
            tickfont=TICK_FONT,
            title_font=BIG_FONT,
            range=[min(year_list) - 0.5, max(year_list) + 0.5]
        ),
        yaxis=dict(
            tickfont=TICK_FONT,
            title_font=BIG_FONT,
            range=[0, 1.05]
        )
    )

    st.plotly_chart(fig_regime, use_container_width=True)

st.markdown("### Full Decision Table")

st.dataframe(df, use_container_width=True)

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Decision Table CSV",
    data=csv,
    file_name="final_dashboard_decision_table.csv",
    mime="text/csv"
)
