import streamlit as st
import pandas as pd
import plotly.express as px

# PAGE SETUP
# This must be the first streamlit command always
# layout="wide" uses the full browser width
st.set_page_config(
    page_title="London Sentiment Dashboard",
    page_icon="🇬🇧",
    layout="wide"
)

# TITLE SECTION
st.title("🇬🇧 London YouTube Sentiment Dashboard")
st.markdown("Analysing what people say about **London restaurants, attractions & tourism** on YouTube")
st.markdown("---")

# LOAD DATA
# @st.cache_data means streamlit remembers the data
# so it doesnt reload the file every time someone clicks anything
@st.cache_data
def load_data():
    # pd.read_csv reads our CSV file into a DataFrame (table)
    df = pd.read_csv("london_sentiment_data.csv")
    return df

df = load_data()

# SIDEBAR FILTERS
# The sidebar is the left panel — good place for filters
st.sidebar.header("🔍 Filters")

# Radio button — user picks one option at a time
sentiment_filter = st.sidebar.radio(
    "Filter by Sentiment",
    options=["All", "POSITIVE", "NEGATIVE"]
)

# APPLY FILTER
# If user picked All — show everything
# Otherwise filter the table to just that sentiment
if sentiment_filter == "All":
    filtered_df = df
else:
    # == checks if the value matches
    # this keeps only rows where label equals what user picked
    filtered_df = df[df["label"] == sentiment_filter]

# METRIC CARDS
# st.columns splits the page into side by side sections
# 3 equal columns
col1, col2, col3 = st.columns(3)

# len() counts rows in the dataframe
total     = len(df)
positive  = len(df[df["label"] == "POSITIVE"])
negative  = len(df[df["label"] == "NEGATIVE"])

# st.metric shows a big number with a label
# great for dashboards
with col1:
    st.metric("💬 Total Comments", total)

with col2:
    st.metric("😊 Positive", positive)

with col3:
    st.metric("😞 Negative", negative)

st.markdown("---")

# CHARTS — side by side
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("📊 Sentiment Split")

    # Count how many positive vs negative
    counts = df["label"].value_counts().reset_index()
    # reset_index() turns the count into a proper table
    counts.columns = ["sentiment", "count"]

    # Donut chart — hole=0.4 makes the middle empty
    fig1 = px.pie(
        counts,
        values="count",
        names="sentiment",
        hole=0.4,
        color="sentiment",
        color_discrete_map={
            "POSITIVE": "#2ecc71",
            "NEGATIVE": "#e74c3c"
        }
    )
    st.plotly_chart(fig1, use_container_width=True)

with chart_col2:
    st.subheader("📈 Confidence Scores")

    # Bar chart showing each comment's confidence score
    # score tells us how sure the model was about its prediction
    fig2 = px.bar(
        filtered_df,
        x=filtered_df.index,
        y="score",
        color="label",
        color_discrete_map={
            "POSITIVE": "#2ecc71",
            "NEGATIVE": "#e74c3c"
        },
        labels={"x": "Comment #", "score": "Confidence Score"}
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# COMMENTS TABLE
st.subheader("💬 Comments")

# Loop through each row and display it as a card
# iterrows() goes through each row one at a time
# i = row number, row = the actual data
for i, row in filtered_df.iterrows():

    # Pick emoji based on sentiment
    emoji = "😊" if row["label"] == "POSITIVE" else "😞"

    # st.expander creates a collapsible box
    # shows the first line, hides the rest until clicked
    with st.expander(f"{emoji} {row['text'][:80]}..."):
        st.write(row["text"])
        st.caption(f"Sentiment: **{row['label']}** | Confidence: **{round(row['score'], 2)}**")
