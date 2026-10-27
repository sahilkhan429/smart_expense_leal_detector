import streamlit as st
import pandas as pd

from analyzer import calculate_total, category_summary, monthly_summary
from leak_detector import detect_unusual_expenses
from savings_advisor import suggest_savings
from forecaster import forecast_next_month
from ai_assistant import ask_expense_assistant
from ml_detector import detect_ml_anomalies


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Expense Lens | Smart Expense Leak Detector",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM DESIGN
# =========================================================

st.markdown(
    """
    <style>
    :root {
        --green: #315C4C;
        --green-dark: #244638;
        --green-light: #EAF2EC;
        --ink: #24332C;
        --muted: #738078;
        --border: #E0E8E2;
        --background: #F5F7F5;
    }

    .stApp {
        background: var(--background);
    }

    [data-testid="stHeader"] {
        background: rgba(245, 247, 245, 0.96);
    }

    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid var(--border);
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: -0.025em;
    }

    h1 {
        font-weight: 750 !important;
    }

    .hero {
        background: linear-gradient(120deg, #244638, #42735D);
        border-radius: 18px;
        padding: 30px;
        margin-bottom: 25px;
        color: white;
    }

    .hero .eyebrow {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
        color: #C7E3D1;
        margin-bottom: 12px;
    }

    .hero h1 {
        color: white !important;
        font-size: 30px;
        margin: 0 0 12px 0;
        line-height: 1.3;
    }

    .hero p {
        color: #E0ECE4 !important;
        font-size: 15px;
        margin: 0;
        line-height: 1.7;
    }

    .section-intro {
        color: var(--muted);
        margin-top: -8px;
        margin-bottom: 18px;
        font-size: 14px;
    }

    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 3px 12px rgba(25, 55, 38, 0.035);
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--ink) !important;
        font-weight: 700;
    }

    .feature-card {
        background: white;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 19px;
        min-height: 135px;
        margin-bottom: 12px;
    }

    .feature-card h3 {
        font-size: 16px;
        margin: 0 0 10px 0;
    }

    .feature-card p {
        font-size: 13px;
        color: var(--muted);
        line-height: 1.65;
        margin: 0;
    }

    .small-label {
        color: var(--muted);
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    [data-testid="stFileUploader"] {
        border-radius: 12px;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 9px;
        font-weight: 600;
        transition: 0.2s ease;
    }

    .stButton > button[kind="primary"] {
        background: var(--green);
        border-color: var(--green);
        color: white;
    }

    .stButton > button[kind="primary"]:hover {
        background: var(--green-dark);
        border-color: var(--green-dark);
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 10px;
        overflow: hidden;
    }

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    hr {
        border-color: var(--border);
    }

    @media (max-width: 768px) {
        .block-container {
            padding: 1rem;
        }

        .hero {
            padding: 22px;
        }

        .hero h1 {
            font-size: 24px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# CONSTANTS
# =========================================================

REQUIRED_COLUMNS = [
    "Date",
    "Description",
    "Category",
    "Amount",
]


# =========================================================
# DATA CLEANING AND VALIDATION
# =========================================================

def prepare_expenses(uploaded_file):
    df = pd.read_csv(uploaded_file)

    if df.empty:
        raise ValueError("The CSV file contains no expense records.")

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(missing)}"
        )

    df = df.copy()

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    df["Amount"] = pd.to_numeric(
        df["Amount"],
        errors="coerce",
    )

    invalid_dates = int(df["Date"].isna().sum())
    invalid_amounts = int(df["Amount"].isna().sum())

    df["Description"] = (
        df["Description"].fillna("").astype(str).str.strip()
    )

    df["Category"] = (
        df["Category"].fillna("").astype(str).str.strip()
    )

    df = df.dropna(
        subset=["Date", "Amount"]
    )

    df = df[
        (df["Description"] != "")
        & (df["Category"] != "")
        & (df["Amount"] >= 0)
    ]

    df = df.drop_duplicates().reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No valid expense records remain after cleaning."
        )

    return df, invalid_dates, invalid_amounts


# =========================================================
# AI SUMMARY
# =========================================================

def make_ai_summary(
    df,
    total,
    categories,
    monthly,
    unusual,
    ml_anomalies,
    savings,
    forecast,
):
    unusual_text = (
        unusual.to_string(index=False)
        if not unusual.empty
        else "No rule-based unusual expenses were detected."
    )

    ml_text = (
        ml_anomalies.to_string(index=False)
        if not ml_anomalies.empty
        else "No ML anomalies were returned."
    )

    savings_text = (
        savings.to_string(index=False)
        if not savings.empty
        else "No categories exceeded the configured spending limit."
    )

    if forecast is None:
        forecast_text = (
            "Unavailable: at least two months of valid data are needed."
        )
    else:
        forecast_text = (
            f"Forecast month: {forecast['Forecast month']}; "
            f"estimated amount: Rs. "
            f"{forecast['Forecast amount']:,.2f}; "
            f"observed months: {forecast['Months of data']}."
        )

    return f"""
EXPENSE ANALYSIS

Total expenses: Rs. {total:,.2f}
Valid transactions: {len(df)}
Date range: {df['Date'].min().date()} to {df['Date'].max().date()}

CATEGORY-WISE SPENDING
{categories.to_string()}

MONTHLY SPENDING
{monthly.to_string()}

RULE-BASED UNUSUAL EXPENSES
{unusual_text}

MACHINE LEARNING ANOMALIES
{ml_text}

POSSIBLE SAVINGS
{savings_text}

NEXT-MONTH FORECAST
{forecast_text}

Instructions:
Explain only the figures and results supplied above.
Do not invent amounts or claim that anomalies prove fraud.
Give practical, cautious suggestions based on the available data.
"""


# =========================================================
# HELPERS
# =========================================================

def money(value):
    return f"Rs. {value:,.2f}"


def show_section(title, description):
    st.subheader(title)
    st.markdown(
        f'<p class="section-intro">{description}</p>',
        unsafe_allow_html=True,
    )


def safe_ai_error(answer):
    if not answer:
        return "The local AI returned an empty response."

    known_errors = (
        "Ollama is not running.",
        "The local AI took too long",
        "Could not contact the local AI:",
    )

    return next(
        (message for message in known_errors if answer.startswith(message)),
        None,
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## 💸 Expense Lens")
    st.caption("SMART EXPENSE LEAK DETECTOR")

    st.divider()

    st.markdown("### Your data")

    st.write(
        "Upload an expense CSV to review spending patterns, "
        "potential anomalies and savings opportunities."
    )

    uploaded_file = st.file_uploader(
        "Choose expense CSV",
        type=["csv"],
        help="Required columns: Date, Description, Category, Amount",
    )

    st.divider()

    st.markdown("### Required CSV format")

    st.code(
        "Date,Description,Category,Amount\n"
        "2026-01-05,Groceries,Food,1200",
        language="csv",
    )

    st.caption(
        "Use fictional data for demonstrations. "
        "Keep private financial records off public repositories."
    )

    st.divider()

    st.caption("Local analysis · Python · Streamlit · scikit-learn")


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">PERSONAL FINANCE INTELLIGENCE</div>
        <h1>Understand your money. Find your leaks.</h1>
        <p>
            Explore spending, identify potentially unusual transactions,
            and make more informed decisions with local analysis and AI.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# EMPTY STATE
# =========================================================

if uploaded_file is None:
    st.markdown("## Your financial overview")

    st.write(
        "Start by uploading an expense CSV from the sidebar. "
        "Your dashboard will appear here after validation."
    )

    card1, card2, card3 = st.columns(3)

    with card1:
        st.markdown(
            """
            <div class="feature-card">
                <h3>📊 Understand spending</h3>
                <p>
                    Explore category totals and monthly trends
                    in one place.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with card2:
        st.markdown(
            """
            <div class="feature-card">
                <h3>🔎 Detect unusual expenses</h3>
                <p>
                    Review transactions flagged by rules
                    and machine learning.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with card3:
        st.markdown(
            """
            <div class="feature-card">
                <h3>🤖 Get local AI insights</h3>
                <p>
                    Ask your locally running assistant
                    to explain the analysis.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info(
        "Required columns: Date, Description, Category, Amount."
    )


# =========================================================
# MAIN DASHBOARD
# =========================================================

else:
    try:
        df, invalid_dates, invalid_amounts = prepare_expenses(
            uploaded_file
        )

        # Run existing analysis modules
        total = calculate_total(df)
        categories = category_summary(df)
        monthly = monthly_summary(df)

        unusual = detect_unusual_expenses(df)
        ml_anomalies = detect_ml_anomalies(df)

        savings = suggest_savings(
            df,
            spending_limit=5000,
        )

        forecast = forecast_next_month(df)

        st.success(
            f"CSV ready — {len(df)} valid transactions loaded."
        )

        if invalid_dates:
            st.warning(
                f"{invalid_dates} row(s) had invalid dates "
                "and were excluded."
            )

        if invalid_amounts:
            st.warning(
                f"{invalid_amounts} row(s) had invalid amounts "
                "and were excluded."
            )

        # ---------------------------------------------
        # KEY METRICS
        # ---------------------------------------------

        st.markdown("## Financial overview")

        st.caption(
            f"Reporting period: "
            f"{df['Date'].min():%d %b %Y} — "
            f"{df['Date'].max():%d %b %Y}"
        )

        metric1, metric2, metric3, metric4 = st.columns(4)

        with metric1:
            st.metric("Total expenses", money(total))

        with metric2:
            st.metric("Transactions", f"{len(df):,}")

        with metric3:
            st.metric("Rule-based flags", f"{len(unusual):,}")

        with metric4:
            st.metric("ML flags", f"{len(ml_anomalies):,}")

        # ---------------------------------------------
        # TABS
        # ---------------------------------------------

        overview_tab, transactions_tab, analytics_tab, alerts_tab, savings_tab, ai_tab = st.tabs(
            [
                "🏠 Overview",
                "📋 Transactions",
                "📊 Analytics",
                "🔎 Anomalies",
                "🌱 Savings & Forecast",
                "🤖 Local AI",
            ]
        )

        # ---------------------------------------------
        # OVERVIEW TAB
        # ---------------------------------------------

        with overview_tab:
            show_section(
                "Spending at a glance",
                "A quick view of where your money goes.",
            )

            left, right = st.columns(2)

            with left:
                st.markdown("#### Category-wise spending")
                st.bar_chart(categories)

            with right:
                st.markdown("#### Monthly spending")
                st.bar_chart(monthly)

            st.markdown("#### Recent transactions")

            recent = df.sort_values(
                "Date",
                ascending=False,
            ).head(8)

            st.dataframe(
                recent,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Anomaly flags are indicators for review, not proof of fraud."
            )

        # ---------------------------------------------
        # TRANSACTIONS TAB
        # ---------------------------------------------

        with transactions_tab:
            show_section(
                "Expense records",
                "Filter, search and export your transactions.",
            )

            filter_col1, filter_col2 = st.columns(2)

            all_categories = sorted(
                df["Category"].unique().tolist()
            )

            with filter_col1:
                selected_categories = st.multiselect(
                    "Filter categories",
                    options=all_categories,
                    default=all_categories,
                )

            with filter_col2:
                search_text = st.text_input(
                    "Search description",
                    placeholder="Search expenses...",
                )

            filtered = df[
                df["Category"].isin(selected_categories)
            ]

            if search_text.strip():
                filtered = filtered[
                    filtered["Description"].str.contains(
                        search_text.strip(),
                        case=False,
                        na=False,
                        regex=False,
                    )
                ]

            st.caption(
                f"{len(filtered):,} matching transaction(s)"
            )

            st.dataframe(
                filtered.sort_values(
                    "Date",
                    ascending=False,
                ),
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download filtered CSV",
                data=filtered.to_csv(index=False).encode("utf-8"),
                file_name="filtered_expenses.csv",
                mime="text/csv",
            )

        # ---------------------------------------------
        # ANALYTICS TAB
        # ---------------------------------------------

        with analytics_tab:
            show_section(
                "Spending analytics",
                "Explore category totals and monthly history.",
            )

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.markdown("#### Category totals")
                st.bar_chart(categories)

                st.markdown("#### Category summary")
                st.dataframe(
                    categories,
                    use_container_width=True,
                )

            with chart_col2:
                st.markdown("#### Monthly totals")
                st.bar_chart(monthly)

                st.markdown("#### Monthly summary")
                st.dataframe(
                    monthly,
                    use_container_width=True,
                )

        # ---------------------------------------------
        # ANOMALIES TAB
        # ---------------------------------------------

        with alerts_tab:
            show_section(
                "Potentially unusual expenses",
                "Review flagged transactions before making decisions.",
            )

            rules_tab, ml_tab = st.tabs(
                [
                    "Rule-based detection",
                    "Machine learning",
                ]
            )

            with rules_tab:
                st.markdown("#### Rule-based results")

                if unusual.empty:
                    st.info(
                        "No unusual expenses were flagged by the current rules."
                    )
                else:
                    st.warning(
                        f"{len(unusual)} transaction(s) flagged for review."
                    )

                    st.dataframe(
                        unusual,
                        use_container_width=True,
                        hide_index=True,
                    )

            with ml_tab:
                st.markdown("#### ML anomaly results")

                if ml_anomalies.empty:
                    st.info(
                        "No ML anomalies were returned. "
                        "A small dataset or low variation may affect detection."
                    )
                else:
                    st.warning(
                        f"{len(ml_anomalies)} transaction(s) flagged by ML."
                    )

                    st.dataframe(
                        ml_anomalies,
                        use_container_width=True,
                        hide_index=True,
                    )

            st.caption(
                "A flagged expense is a potential outlier, not confirmed fraud."
            )

        # ---------------------------------------------
        # SAVINGS AND FORECAST TAB
        # ---------------------------------------------

        with savings_tab:
            show_section(
                "Savings opportunities",
                "Review possible savings and the historical-average forecast.",
            )

            st.markdown("#### Possible savings")

            if savings.empty:
                st.info(
                    "No categories exceed the configured Rs. 5,000 limit."
                )
            else:
                st.dataframe(
                    savings,
                    use_container_width=True,
                    hide_index=True,
                )

            st.divider()

            st.markdown("#### Next-month forecast")

            if forecast is None:
                st.info(
                    "At least two months of valid data are needed "
                    "for this forecast."
                )
            else:
                forecast_col1, forecast_col2 = st.columns(2)

                with forecast_col1:
                    st.metric(
                        f"Estimated · {forecast['Forecast month']}",
                        money(forecast["Forecast amount"]),
                    )

                with forecast_col2:
                    st.metric(
                        "Months observed",
                        f"{forecast['Months of data']}",
                    )

                st.caption(
                    "This is a historical-average estimate, not a guarantee."
                )

        # ---------------------------------------------
        # LOCAL AI TAB
        # ---------------------------------------------

        with ai_tab:
            show_section(
                "Your local expense assistant",
                "Get an explanation using your locally running Ollama model.",
            )

            st.markdown(
                """
                <div class="feature-card">
                    <h3>Private, local AI insights</h3>
                    <p>
                        The assistant receives a summary of this analysis.
                        Make sure Ollama is running before requesting insights.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button(
                "✨ Explain my expenses",
                type="primary",
                key="explain_expenses",
            ):
                with st.spinner(
                    "The local AI is preparing your explanation..."
                ):
                    summary = make_ai_summary(
                        df=df,
                        total=total,
                        categories=categories,
                        monthly=monthly,
                        unusual=unusual,
                        ml_anomalies=ml_anomalies,
                        savings=savings,
                        forecast=forecast,
                    )

                    try:
                        answer = ask_expense_assistant(summary)
                    except Exception as error:
                        answer = f"Could not contact the local AI: {error}"

                error_message = safe_ai_error(answer)

                if error_message:
                    st.error(error_message)
                else:
                    st.markdown("#### AI explanation")
                    st.markdown(answer)

                st.caption(
                    "AI suggestions may be imperfect. "
                    "Verify important figures against the displayed analysis."
                )

        # ---------------------------------------------
        # FOOTER
        # ---------------------------------------------

        st.divider()

        st.caption(
            "Smart Expense Leak Detector · Local analysis · "
            "Forecasts and anomaly flags are estimates, not financial advice."
        )

    except (
        ValueError,
        pd.errors.EmptyDataError,
    ) as error:
        st.error(f"CSV validation error: {error}")

    except Exception as error:
        st.error(f"Could not process the CSV: {error}")