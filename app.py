import streamlit as st
import pandas as pd
import plotly.express as px
from googleapiclient.discovery import build

# PAGE SETUP
# Must always be the very first streamlit command
st.set_page_config(
    page_title="London Sentiment Dashboard",
    page_icon="🇬🇧",
    layout="wide"
)

# TITLE
st.title("🇬🇧 London YouTube Sentiment Dashboard")
st.markdown("Analysing **live** YouTube comments about London restaurants, attractions & tourism")
st.markdown("---")

# LOAD API KEY FROM STREAMLIT SECRETS
# st.secrets reads from the Secrets section we set up
# This keeps our key safe — never appears in code
API_KEY = st.secrets["YOUTUBE_API_KEY"]

# CONNECT TO YOUTUBE
# build() creates our connection to YouTube API
# "youtube" = which Google service
# "v3" = version 3 of the API
youtube = build("youtube", "v3", developerKey=API_KEY)

# SIDEBAR — user controls
st.sidebar.header("🔍 Controls")

# Selectbox = dropdown menu
# User picks which topic to analyse
topic = st.sidebar.selectbox(
    "Choose a topic",
    options=[
        "London restaurant review",
        "London tourist attractions",
        "London food markets",
        "London travel guide",
        "London street food"
    ]
)

# Slider = draggable number picker
# User picks how many videos to search
num_videos = st.sidebar.slider(
    "Number of videos to analyse",
    min_value=3,    # minimum they can pick
    max_value=10,   # maximum they can pick
    value=5         # default value
)

# Button — app only fetches data when user clicks this
# This saves API quota — don't fetch on every page load
fetch_button = st.sidebar.button("🔄 Fetch Live Data")

# SENTIMENT FILTER
sentiment_filter = st.sidebar.radio(
    "Filter by Sentiment",
    options=["All", "POSITIVE", "NEGATIVE"]
)

# FETCH DATA FUNCTION
# @st.cache_data means streamlit remembers results
# so if same topic is searched again it doesn't refetch
# ttl=3600 means cache expires after 1 hour (3600 seconds)
@st.cache_data(ttl=3600)
def fetch_youtube_comments(topic, num_videos):

    # Tell the user something is happening
    # st.spinner shows a loading animation
    comments_list = []

    # Step 1 — Search YouTube for videos on this topic
    search_response = youtube.search().list(
        q=topic,
        type="video",
        part="snippet",
        maxResults=num_videos
    ).execute()

    # Step 2 — For each video fetch its comments
    for item in search_response["items"]:

        # Get video ID and title from search results
        video_id    = item["id"]["videoId"]
        video_title = item["snippet"]["title"]

        try:
            # Fetch comments for this video
            comments_response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=30,          # 30 comments per video
                textFormat="plainText"  # no HTML tags in text
            ).execute()

            # Loop through each comment
            for comment in comments_response["items"]:

                # Navigate into the nested dictionary to get comment text
                # snippet > topLevelComment > snippet > textDisplay
                text = comment["snippet"]["topLevelComment"]["snippet"]["textDisplay"]

                comments_list.append({
                    "video_title": video_title,
                    "comment": text,
                    "video_id": video_id
                })

        except Exception:
            # Some videos have comments disabled — skip them silently
            continue

    return pd.DataFrame(comments_list)

# SENTIMENT FUNCTION
@st.cache_data(ttl=3600)
def run_sentiment(df):

    # We use a lightweight model here
    # it runs on Streamlit's free servers so needs to be small
    from transformers import pipeline
    analyser = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=512
    )

    labels = []
    scores = []

    for comment in df["comment"]:
        try:
            result  = analyser(comment[:512])
            labels.append(result[0]["label"])
            scores.append(round(result[0]["score"], 2))
        except Exception:
            labels.append("NEUTRAL")
            scores.append(0.5)

    df["sentiment"] = labels
    df["score"]     = scores
    return df

# MAIN APP LOGIC
# Only runs when user clicks the Fetch button
if fetch_button:

    with st.spinner("Fetching live YouTube comments..."):
        df_raw = fetch_youtube_comments(topic, num_videos)

    if len(df_raw) == 0:
        # st.warning shows a yellow warning box
        st.warning("No comments found — try a different topic.")

    else:
        with st.spinner("Running sentiment analysis..."):
            df = run_sentiment(df_raw)

        # APPLY SENTIMENT FILTER
        if sentiment_filter != "All":
            filtered_df = df[df["sentiment"] == sentiment_filter]
        else:
            filtered_df = df

        # METRIC CARDS
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("💬 Comments Analysed", len(df))
        with col2:
            st.metric("📹 Videos Searched", num_videos)
        with col3:
            positive = len(df[df["sentiment"] == "POSITIVE"])
            st.metric("😊 Positive", positive)
        with col4:
            negative = len(df[df["sentiment"] == "NEGATIVE"])
            st.metric("😞 Negative", negative)

        st.markdown("---")

        # CHARTS
        chart1, chart2 = st.columns(2)

        with chart1:
            st.subheader("📊 Sentiment Split")

            counts = df["sentiment"].value_counts().reset_index()
            counts.columns = ["sentiment", "count"]

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
            st.plotly_chart(fig1, use_container_width=True)

        with chart2:
            st.subheader("📈 Sentiment by Video")

            # Group by video title and sentiment
            # Count how many of each per video
            grouped = df.groupby(
                ["video_title", "sentiment"]
            ).size().reset_index(name="count")

            # Shorten long video titles so they fit on chart
            # lambda is a tiny one-line function
            # x[:40] takes first 40 characters
            grouped["video_title"] = grouped["video_title"].apply(
                lambda x: x[:40] + "..." if len(x) > 40 else x
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
            fig2.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")

        # COMMENTS TABLE
        st.subheader("💬 Live Comments")

        for i, row in filtered_df.iterrows():
            emoji = "😊" if row["sentiment"] == "POSITIVE" else "😞"
            with st.expander(f"{emoji} {row['comment'][:80]}..."):
                st.write(row["comment"])
                st.caption(f"Video: {row['video_title']}")
                st.caption(f"Sentiment: **{row['sentiment']}** | Confidence: **{row['score']}**")

else:
    # This shows when app first loads before button is clicked
    st.info("👈 Choose a topic in the sidebar and click **Fetch Live Data** to begin.")
