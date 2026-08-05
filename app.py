import streamlit as st
import pandas as pd
import plotly.express as px
from googleapiclient.discovery import build

# Must be the very first streamlit command always
st.set_page_config(
    page_title="London Sentiment Dashboard",
    page_icon="🇬🇧",
    layout="wide"
)

# Custom CSS to make it look nicer
# unsafe_allow_html=True lets us inject raw CSS
st.markdown("""
    <style>
    .metric-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        border: 1px solid #2d3250;
        margin-bottom: 1rem;
    }
    .metric-number { font-size: 2rem; font-weight: 700; margin: 0; }
    .metric-label  { font-size: 0.85rem; color: #8892b0; margin: 0; }
    .positive { color: #2ecc71; }
    .negative { color: #e74c3c; }
    .neutral  { color: #95a5a6; }
    </style>
""", unsafe_allow_html=True)

# TITLE
st.title("🇬🇧 London YouTube Sentiment Dashboard")
st.markdown("Analysing **live** YouTube comments about London restaurants, attractions & tourism")
st.markdown("---")

# READ API KEY FROM SECRETS
# st.secrets reads from Streamlit's secure settings
# Never put the actual key in your code
API_KEY = st.secrets["AIzaSyC0WRQPsDmpUdy4R1PH5vWrN9qZDKxWnKk"]

# Connect to YouTube API
# "youtube" = which Google service
# "v3" = version 3
youtube = build("youtube", "v3", developerKey="AIzaSyC0WRQPsDmpUdy4R1PH5vWrN9qZDKxWnKk")

# SIDEBAR CONTROLS
st.sidebar.header("🔍 Controls")

# Dropdown — user picks which London topic to analyse
topic = st.sidebar.selectbox(
    "Choose a topic",
    options=[
        "London restaurant review",
        "London tourist attractions vlog",
        "London food markets",
        "London travel guide",
        "London museums review"
    ]
)

# Slider — user picks how many videos to search
# min_value = minimum they can pick
# max_value = maximum they can pick
# value = default starting value
num_videos = st.sidebar.slider(
    "Number of videos",
    min_value=2,
    max_value=8,
    value=3
)

# Radio button — filter results by sentiment
sentiment_filter = st.sidebar.radio(
    "Filter by sentiment",
    options=["All", "POSITIVE", "NEGATIVE"]
)

# Button — only fetch data when user clicks this
# Saves API quota — don't fetch on every page load
fetch_btn = st.sidebar.button("🔄 Fetch Live Data")

# FETCH FUNCTION
# @st.cache_data remembers results so clicking filters
# doesn't refetch from YouTube every time
# ttl=3600 means cache expires after 1 hour
@st.cache_data(ttl=3600)
def fetch_comments(topic, num_videos):

    all_rows = []

    # Search YouTube for videos on this topic
    search_response = youtube.search().list(
        q=topic,
        type="video",
        part="snippet",
        maxResults=num_videos
    ).execute()

    # Loop through each video
    for item in search_response["items"]:
        video_id    = item["id"]["videoId"]
        video_title = item["snippet"]["title"]

        try:
            # Fetch comments for this video
            comments_response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=25,
                textFormat="plainText"
            ).execute()

            for comment in comments_response["items"]:
                text = comment["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                all_rows.append({
                    "video_title": video_title,
                    "comment":     text,
                    "video_id":    video_id
                })

        except Exception:
            continue

    return pd.DataFrame(all_rows)

# SENTIMENT FUNCTION
# @st.cache_data means sentiment only runs once per dataset
# not every time user clicks a filter
@st.cache_data(ttl=3600)
def run_sentiment(_df):

    # Load model inside function
    # @st.cache_data means this only runs once
    from transformers import pipeline
    analyser = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=512
    )

    labels = []
    scores = []

    for comment in _df["comment"]:
        try:
            result = analyser(comment[:512])
            labels.append(result[0]["label"])
            scores.append(round(result[0]["score"], 2))
        except Exception:
            labels.append("NEUTRAL")
            scores.append(0.5)

    _df = _df.copy()
    _df["sentiment"] = labels
    _df["score"]     = scores
    return _df

# MAIN LOGIC
# Only runs when user clicks the fetch button
if fetch_btn:

    with st.spinner("🔍 Fetching live YouTube comments..."):
        df_raw = fetch_comments(topic, num_videos)

    if len(df_raw) == 0:
        st.warning("No comments found — try a different topic.")

    else:
        with st.spinner("🧠 Running AI sentiment analysis..."):
            df = run_sentiment(df_raw)

        # APPLY FILTER
        # If user picked All show everything
        # Otherwise filter to just that sentiment
        if sentiment_filter != "All":
            filtered = df[df["sentiment"] == sentiment_filter]
        else:
            filtered = df

        # COUNT METRICS
        total    = len(df)
        positive = len(df[df["sentiment"] == "POSITIVE"])
        negative = len(df[df["sentiment"] == "NEGATIVE"])
        pct      = round((positive / total) * 100) if total > 0 else 0

        # METRIC CARDS
        # st.columns splits page into side by side sections
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-number">{total}</p>
                    <p class="metric-label">💬 Comments</p>
                </div>""", unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-number positive">{positive}</p>
                    <p class="metric-label">😊 Positive</p>
                </div>""", unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-number negative">{negative}</p>
                    <p class="metric-label">😞 Negative</p>
                </div>""", unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-number">{pct}%</p>
                    <p class="metric-label">⭐ Positive Rate</p>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # CHARTS
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Sentiment Split")

            # Count sentiments for pie chart
            counts = df["sentiment"].value_counts().reset_index()
            counts.columns = ["sentiment", "count"]

            # hole=0.4 makes it a donut chart
            fig1 = px.pie(
                counts,
                values="count",
                names="sentiment",
                hole=0.4,
                color="sentiment",
                color_discrete_map={
                    "POSITIVE": "#2ecc71",
                    "NEGATIVE": "#e74c3c",
                    "NEUTRAL":  "#95a5a6"
                }
            )
            # Transparent background so it fits dark theme
            fig1.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white"
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.subheader("📈 Sentiment by Video")

            # groupby groups rows by two columns together
            # size() counts how many rows in each group
            # reset_index converts back to flat table
            grouped = df.groupby(
                ["video_title", "sentiment"]
            ).size().reset_index(name="count")

            # Shorten long titles so they fit on chart
            # lambda = tiny one-line function
            # x[:35] takes first 35 characters
            grouped["video_title"] = grouped["video_title"].apply(
                lambda x: x[:35] + "..." if len(x) > 35 else x
            )

            fig2 = px.bar(
                grouped,
                x="video_title",
                y="count",
                color="sentiment",
                barmode="group",
                color_discrete_map={
                    "POSITIVE": "#2ecc71",
                    "NEGATIVE": "#e74c3c",
                    "NEUTRAL":  "#95a5a6"
                }
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                xaxis_tickangle=-30
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")

        # COMMENTS TABLE
        st.subheader("💬 Comments")

        # iterrows() loops through each row one at a time
        # i = row number, row = the actual data
        for i, row in filtered.iterrows():
            emoji = "😊" if row["sentiment"] == "POSITIVE" else "😞"

            # expander = collapsible box
            # shows first line, hides rest until clicked
            with st.expander(f"{emoji} {str(row['comment'])[:80]}..."):
                st.write(row["comment"])
                st.caption(f"📹 {row['video_title']}")
                st.caption(f"Sentiment: **{row['sentiment']}** | Confidence: **{row['score']}**")

else:
    # Shows when app first loads before button clicked
    st.info("👈 Pick a topic in the sidebar and click **Fetch Live Data** to begin.")
