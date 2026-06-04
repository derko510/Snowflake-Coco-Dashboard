"""
PMC Biomedical Research Dashboard (Snowflake Edition)

Explores emerging drug-class literature (GLP-1, SGLT2, Checkpoint Inhibitors)
from the PMC_ANALYSIS.PUBLIC dataset.

Data sources:
- PMC_ANALYSIS.PUBLIC.PMC_ARTICLES      (title, abstract, journal, pub_date)
- PMC_ANALYSIS.PUBLIC.SEMANTIC_ANALYSIS (drug_class, study_type, sentiment, themes)

Connection:
- Locally:  SNOWFLAKE_DEFAULT_CONNECTION_NAME selects the connection.
- In SiS:   the active session is used (connection named "default").

All queries are parameterized. No user input is interpolated into SQL.
"""

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="PMC Research Dashboard",
    page_icon=":material/clinical_notes:",
    layout="wide",
)

CHART_HEIGHT = 300
DB_SCHEMA = "PMC_ANALYSIS.PUBLIC"

# Snowflake brand palette (matches .streamlit/config.toml)
BRAND_BLUE = "#29B5E8"
BRAND_TEXT = "#11567F"
CATEGORICAL = ["#29B5E8", "#FF8B00", "#36B37E", "#6554C0", "#DE350B", "#FFAB00", "#00A3BF"]

# Stable color per drug class so every chart agrees
DRUG_CLASS_COLORS = {
    "GLP-1 Agonists": "#29B5E8",
    "Checkpoint Inhibitors": "#6554C0",
    "SGLT2 Inhibitors": "#36B37E",
    "GLP-1 + SGLT2 Combination": "#FF8B00",
    "Unclassified": "#b8c7d1",
}


def _drug_class_scale() -> alt.Scale:
    return alt.Scale(
        domain=list(DRUG_CLASS_COLORS.keys()),
        range=list(DRUG_CLASS_COLORS.values()),
    )


# =============================================================================
# Snowflake connection & data loading
# =============================================================================


def get_connection():
    """Return a Snowflake connection.

    Connection selection is environment-driven, not hardcoded here:
    - Locally, set SNOWFLAKE_DEFAULT_CONNECTION_NAME to pick a connections.toml
      entry (the snowflake-python-connector reads this var).
    - In Streamlit-in-Snowflake, the active session is used automatically.

    Do NOT pass connection_name as a kwarg here: the positional "snowflake" is
    already forwarded as the connection name and the two collide.
    """
    try:
        return st.connection("snowflake")
    except Exception as e:  # noqa: BLE001 - surface config errors to the user
        st.error(f"Failed to connect to Snowflake: {e}")
        st.info(
            "Set SNOWFLAKE_DEFAULT_CONNECTION_NAME to a configured connection, "
            "or configure `.streamlit/secrets.toml`."
        )
        st.stop()


ARTICLES_QUERY = f"""
    SELECT
        a.PMCID          AS pmcid,
        a.TITLE          AS title,
        a.JOURNAL        AS journal,
        a.PUB_DATE       AS pub_date,
        a.ABSTRACT       AS abstract,
        -- Use the semantic drug class when available; otherwise backfill from the
        -- DRUG_CLASSIFICATIONS flags; anything with no flag set is Unclassified.
        COALESCE(
            s.DRUG_CLASS,
            CASE
                WHEN d.GLP1_FLAG AND d.SGLT2_FLAG THEN 'GLP-1 + SGLT2 Combination'
                WHEN d.GLP1_FLAG       THEN 'GLP-1 Agonists'
                WHEN d.SGLT2_FLAG      THEN 'SGLT2 Inhibitors'
                WHEN d.CHECKPOINT_FLAG THEN 'Checkpoint Inhibitors'
            END,
            'Unclassified'
        )                AS drug_class,
        s.STUDY_TYPE     AS study_type,
        s.SENTIMENT      AS sentiment,
        s.PRIMARY_THEME  AS primary_theme,
        s.CLINICAL_OUTLOOK AS clinical_outlook
    FROM {DB_SCHEMA}.PMC_ARTICLES a
    LEFT JOIN {DB_SCHEMA}.SEMANTIC_ANALYSIS s
      ON a.PMCID = s.PMCID
    LEFT JOIN {DB_SCHEMA}.DRUG_CLASSIFICATIONS d
      ON a.PMCID = d.PMCID
    WHERE a.PUB_DATE IS NOT NULL
    ORDER BY a.PUB_DATE
"""


@st.cache_data(ttl=3600, show_spinner="Loading research data from Snowflake...")
def load_articles() -> pd.DataFrame:
    """Load the enriched article-level dataset (one row per article)."""
    conn = get_connection()
    df = conn.query(ARTICLES_QUERY)
    df.columns = df.columns.str.lower()  # normalize Snowflake UPPER columns
    df["pub_date"] = pd.to_datetime(df["pub_date"])
    # Tidy display labels
    df["study_type"] = df["study_type"].fillna("unknown").str.replace("_", " ").str.title()
    df["sentiment"] = df["sentiment"].fillna("unknown").str.title()
    df["primary_theme"] = (
        df["primary_theme"].fillna("unknown").str.replace("_", " ").str.title()
    )
    return df


# =============================================================================
# Chart helpers
# =============================================================================


def _style(chart: alt.Chart) -> alt.Chart:
    """Apply consistent, clean Snowflake-brand styling to any chart."""
    return (
        chart.configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=BRAND_TEXT,
            titleColor=BRAND_TEXT,
            labelFont="Inter",
            titleFont="Inter",
            titleFontWeight=600,
            grid=False,
            domainColor="#d0e8f2",
            tickColor="#d0e8f2",
        )
        .configure_legend(
            labelColor=BRAND_TEXT,
            titleColor=BRAND_TEXT,
            labelFont="Inter",
            titleFont="Inter",
            titleFontWeight=600,
            symbolType="circle",
        )
    )


def publications_over_time(df: pd.DataFrame) -> alt.Chart:
    """Quarterly article counts, colored by drug class."""
    d = df.copy()
    d["quarter"] = d["pub_date"].dt.to_period("Q").dt.start_time
    agg = (
        d.groupby(["quarter", "drug_class"]).size().reset_index(name="articles")
    )
    base = alt.Chart(agg).encode(
        x=alt.X("quarter:T", title=None, axis=alt.Axis(format="%Y", tickCount="year")),
        y=alt.Y("articles:Q", title="Articles", scale=alt.Scale(zero=True)),
        color=alt.Color(
            "drug_class:N",
            title="Drug class",
            scale=_drug_class_scale(),
            legend=alt.Legend(orient="bottom"),
        ),
    )
    chart = (
        base.mark_area(opacity=0.12).encode(y=alt.Y("articles:Q", stack=None))
        + base.mark_line(strokeWidth=2.5, interpolate="monotone")
        + base.mark_point(size=45, filled=True, opacity=1).encode(
            tooltip=[
                alt.Tooltip("quarter:T", title="Quarter", format="%Y-Q%q"),
                alt.Tooltip("drug_class:N", title="Drug class"),
                alt.Tooltip("articles:Q", title="Articles"),
            ]
        )
    )
    return _style(chart.properties(height=CHART_HEIGHT))


def count_bar(df: pd.DataFrame, col: str, title: str, top_n: int | None = None) -> alt.Chart:
    """Horizontal bar of value counts, with value labels."""
    agg = df.groupby(col).size().reset_index(name="articles")
    agg = agg.sort_values("articles", ascending=False)
    if top_n:
        agg = agg.head(top_n)

    color = (
        alt.Color(f"{col}:N", scale=_drug_class_scale(), legend=None)
        if col == "drug_class"
        else alt.value(BRAND_BLUE)
    )
    base = alt.Chart(agg).encode(
        x=alt.X("articles:Q", title=None, axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y(f"{col}:N", title=None, sort="-x"),
        tooltip=[
            alt.Tooltip(f"{col}:N", title=title),
            alt.Tooltip("articles:Q", title="Articles"),
        ],
    )
    bars = base.mark_bar(cornerRadiusEnd=4).encode(color=color)
    labels = base.mark_text(
        align="left", dx=4, color=BRAND_TEXT, font="Inter", fontWeight=600
    ).encode(text="articles:Q")
    return _style((bars + labels).properties(height=CHART_HEIGHT))


def sentiment_by_class(df: pd.DataFrame) -> alt.Chart:
    """Stacked bar of sentiment share within each drug class."""
    agg = df.groupby(["drug_class", "sentiment"]).size().reset_index(name="articles")
    chart = (
        alt.Chart(agg)
        .mark_bar(cornerRadius=2)
        .encode(
            x=alt.X("drug_class:N", title=None, axis=alt.Axis(labelAngle=-15)),
            y=alt.Y("articles:Q", title="Share", stack="normalize",
                    axis=alt.Axis(format="%")),
            color=alt.Color(
                "sentiment:N",
                title="Sentiment",
                scale=alt.Scale(
                    domain=["Positive", "Neutral", "Negative", "Unknown"],
                    range=["#36B37E", "#8aa9bd", "#DE350B", "#c7c7c7"],
                ),
                legend=alt.Legend(orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("drug_class:N", title="Drug class"),
                alt.Tooltip("sentiment:N", title="Sentiment"),
                alt.Tooltip("articles:Q", title="Articles"),
            ],
        )
        .properties(height=CHART_HEIGHT)
    )
    return _style(chart)


def drug_class_donut(df: pd.DataFrame) -> alt.Chart:
    """Donut of article share by drug class, with a centered total."""
    agg = df.groupby("drug_class").size().reset_index(name="articles")
    total = int(agg["articles"].sum())
    base = alt.Chart(agg).encode(
        theta=alt.Theta("articles:Q", stack=True),
        color=alt.Color(
            "drug_class:N",
            title="Drug class",
            scale=_drug_class_scale(),
            legend=alt.Legend(orient="bottom", columns=2),
        ),
        tooltip=[
            alt.Tooltip("drug_class:N", title="Drug class"),
            alt.Tooltip("articles:Q", title="Articles"),
        ],
    )
    ring = base.mark_arc(innerRadius=70, outerRadius=110, cornerRadius=3, stroke="#fff",
                         strokeWidth=2)
    center = (
        alt.Chart(pd.DataFrame({"t": [f"{total}"], "l": ["articles"]}))
        .mark_text(font="Inter", fontWeight=700, fontSize=34, color=BRAND_TEXT, dy=-8)
        .encode(text="t:N")
    )
    center_lbl = (
        alt.Chart(pd.DataFrame({"l": ["articles"]}))
        .mark_text(font="Inter", fontSize=13, color="#7f9fb3", dy=18)
        .encode(text="l:N")
    )
    return _style((ring + center + center_lbl).properties(height=CHART_HEIGHT))


# =============================================================================
# Load data
# =============================================================================

articles = load_articles()

# =============================================================================
# Header
# =============================================================================

with st.container(
    horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"
):
    st.markdown("# :material/clinical_notes: PMC Research Dashboard")
    if st.button(":material/restart_alt: Reset filters", type="tertiary"):
        st.session_state.clear()
        st.rerun()
st.caption(
    ":material/cloud: Emerging drug-class literature (GLP-1 · SGLT2 · Checkpoint "
    "Inhibitors) · Powered by Snowflake"
)
st.divider()

# =============================================================================
# Filters (sidebar)
# =============================================================================

with st.sidebar:
    st.markdown(
        "<div style='display:flex;align-items:center;gap:.5rem;"
        "padding:.25rem 0 .75rem;font-size:1.15rem;font-weight:700;color:#fff'>"
        "<span style='font-size:1.5rem'>&#10052;&#65039;</span> PMC Research</div>",
        unsafe_allow_html=True,
    )
    st.caption("Emerging drug-class literature")
    st.divider()
    st.header(":material/filter_alt: Filters")

    all_classes = sorted(articles["drug_class"].dropna().unique().tolist())
    selected_classes = st.multiselect(
        "Drug class", options=all_classes, default=all_classes
    )

    min_d = articles["pub_date"].min().date()
    max_d = articles["pub_date"].max().date()
    date_range = st.slider(
        "Publication date range",
        min_value=min_d,
        max_value=max_d,
        value=(min_d, max_d),
    )

# Apply filters
mask = (
    articles["drug_class"].isin(selected_classes or all_classes)
    & (articles["pub_date"].dt.date >= date_range[0])
    & (articles["pub_date"].dt.date <= date_range[1])
)
fdf = articles[mask]

if fdf.empty:
    st.warning("No articles match the current filters.")
    st.stop()

# =============================================================================
# KPI cards
# =============================================================================

# Year-over-year delta on article volume: latest year present vs. the year before.
_year_counts = fdf.groupby(fdf["pub_date"].dt.year)["pmcid"].nunique()
_articles_delta = None
if len(_year_counts) >= 2:
    years_sorted = sorted(_year_counts.index)
    latest, prior = years_sorted[-1], years_sorted[-2]
    _articles_delta = f"{int(_year_counts[latest] - _year_counts[prior]):+d} vs {prior}"

k = st.columns(4)
k[0].metric(
    ":material/article: Articles",
    f"{fdf['pmcid'].nunique():,}",
    delta=_articles_delta,
    border=True,
)
k[1].metric(":material/menu_book: Journals", f"{fdf['journal'].nunique():,}", border=True)
k[2].metric(
    ":material/medication: Drug classes", f"{fdf['drug_class'].nunique():,}", border=True
)
k[3].metric(
    ":material/calendar_month: Date range",
    f"{fdf['pub_date'].min():%Y-%m} → {fdf['pub_date'].max():%Y-%m}",
    border=True,
)

st.divider()

# =============================================================================
# Charts
# =============================================================================

with st.container(border=True):
    st.markdown("**Publications over time** (quarterly, by drug class)")
    st.altair_chart(publications_over_time(fdf), width="stretch")

row = st.columns(2)
with row[0]:
    with st.container(border=True):
        st.markdown("**Article share by drug class**")
        st.altair_chart(drug_class_donut(fdf), width="stretch")
with row[1]:
    with st.container(border=True):
        st.markdown("**Sentiment mix by drug class** (share)")
        st.altair_chart(sentiment_by_class(fdf), width="stretch")

row2 = st.columns(2)
with row2[0]:
    with st.container(border=True):
        st.markdown("**Study types**")
        st.altair_chart(count_bar(fdf, "study_type", "Study type"),
                        width="stretch")
with row2[1]:
    with st.container(border=True):
        st.markdown("**Top research themes**")
        st.altair_chart(count_bar(fdf, "primary_theme", "Theme", top_n=10),
                        width="stretch")

st.divider()

# =============================================================================
# Article browser
# =============================================================================

st.markdown("### :material/menu_book: Article browser")
st.caption(f"{len(fdf):,} articles match the current filters.")

st.dataframe(
    fdf[["pub_date", "title", "journal", "drug_class", "study_type", "sentiment"]]
    .sort_values("pub_date", ascending=False)
    .rename(
        columns={
            "pub_date": "Published",
            "title": "Title",
            "journal": "Journal",
            "drug_class": "Drug class",
            "study_type": "Study type",
            "sentiment": "Sentiment",
        }
    ),
        width="stretch",
        hide_index=True,
    column_config={"Published": st.column_config.DateColumn(format="YYYY-MM-DD")},
)

with st.expander("Read abstracts & clinical outlook"):
    for _, r in fdf.sort_values("pub_date", ascending=False).iterrows():
        st.markdown(
            f"**{r['title']}**  \n"
            f":gray[{r['journal']} · {r['pub_date']:%Y-%m-%d} · "
            f"{r['drug_class']} · {r['study_type']} · {r['sentiment']}]"
        )
        if pd.notna(r["clinical_outlook"]):
            st.markdown(f"> {r['clinical_outlook']}")
        if pd.notna(r["abstract"]):
            with st.popover("Abstract"):
                st.write(r["abstract"])
        st.divider()
