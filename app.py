import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import time
from googleapiclient.discovery import build

# Must be first streamlit command always
st.set_page_config(
    page_title="London Sentiment Dashboard",
    page_icon="🇬🇧",
    layout="wide"
)

# Custom styling
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
    </style>
""", unsafe_allow_html=True)

# TITLE
st.title("🇬🇧 London YouTube Sentiment Dashboard")
st.markdown("Analysing **live** YouTube comments about London restaurants, attractions & tourism")
st.markdown("---")

# READ SECRETS
# st.secrets reads safely from Streamlit's secure settings
# Never put actual keys in code
YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
HF_API_TOKEN    = st.secrets["HF_API_TOKEN"]

# Connect to YouTube
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# HuggingFace API settings
# Instead of running model locally we send text to HuggingFace servers
# They run the model and send back the result
# This means no torch needed on our server
HF_API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
HF_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# SENTIMENT FUNCTION
def analyse_sentiment(texts):
    # We send texts in batches of 10
    # Batch processing is faster than one at a time
    results = []
# Wake up the model first with a test request
    requests.post(HF_API_URL, headers=HF_HEADERS, 
    json={"inputs": ["test"]}, timeout=60)
    time.sleep(5)

    # range(0, len, 10) creates [0, 10, 20, 30...]
    # this loops through texts in groups of 10
    for i in range(0, len(texts), 10):

        # texts[i:i+10] gets the next 10 texts
        batch = texts[i:i+10]

        try:
            # requests.post sends data to HuggingFace API
            # json= sends our texts as JSON format
            # headers= sends our API token for authentication
            response = requests.post(
                HF_API_URL,
                headers=HF_HEADERS,
                json={"inputs": batch},
                timeout=70  # give up after 30 seconds
            )

            # If model is loading HuggingFace returns 503
            # We wait 10 seconds and try again
            if response.status_code == 503:
                time.sleep(30)
                response = requests.post(
                    HF_API_URL,
                    headers=HF_HEADERS,
                    json={"inputs": batch},
                    timeout=70
                    
                )

            # .json() converts response to Python list
            batch_results = response.json()

            # Each result is a list of dicts with label and score
            # We pick the one with highest score
            for result in batch_results:
                if isinstance(result, list):
                    # max() finds dict with highest score value
                    best = max(result, key=lambda x: x["score"])
                    results.append({
                        "sentiment": best["label"],
                        "score":     round(best["score"], 2)
                    })
                else:
                    results.append({
                        "sentiment": "NEUTRAL",
                        "score":     0.5
                    })

        except Exception:
            # If anything fails add neutral for that batch
            for _ in batch:
                results.append({
                    "sentiment": "NEUTRAL",
                    "score":     0.5
                })

    return results

# FETCH YOUTUBE COMMENTS
# @st.cache_data remembers results for 1 hour
# so clicking filters doesn't refetch from YouTube
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

# SIDEBAR
st.sidebar.header("🔍 Controls")

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

num_videos = st.sidebar.slider(
    "Number of videos",
    min_value=2,
    max_value=8,
    value=3
)

sentiment_filter = st.sidebar.radio(
    "Filter by sentiment",
    options=["All", "POSITIVE", "NEGATIVE"]
)

fetch_btn = st.sidebar.button("🔄 Fetch Live Data")

# MAIN LOGIC
if fetch_btn:

    with st.spinner("🔍 Fetching YouTube comments..."):
        df_raw = fetch_comments(topic, num_videos)

    if len(df_raw) == 0:
        st.warning("No comments found — try a different topic.")

    else:
        with st.spinner("🧠 Running sentiment analysis via HuggingFace..."):
            # Get list of comment texts
            texts = df_raw["comment"].tolist()

            # Run sentiment on all of them
            sentiment_results = analyse_sentiment(texts)

            # Add results back to our DataFrame
            df_raw["sentiment"] = [r["sentiment"] for r in sentiment_results]
            df_raw["score"]     = [r["score"]     for r in sentiment_results]
            df = df_raw.copy()

        # APPLY FILTER
        if sentiment_filter != "All":
            filtered = df[df["sentiment"] == sentiment_filter]
        else:
            filtered = df

        # METRICS
        total    = len(df)
        positive = len(df[df["sentiment"] == "POSITIVE"])
        negative = len(df[df["sentiment"] == "NEGATIVE"])
        pct      = round((positive / total) * 100) if total > 0 else 0

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
            fig1.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white"
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.subheader("📈 Sentiment by Video")
            grouped = df.groupby(
                ["video_title", "sentiment"]
            ).size().reset_index(name="count")
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

        # COMMENTS
        st.subheader("💬 Comments")
        for i, row in filtered.iterrows():
            emoji = "😊" if row["sentiment"] == "POSITIVE" else "😞"
            with st.expander(f"{emoji} {str(row['comment'])[:80]}..."):
                st.write(row["comment"])
                st.caption(f"📹 {row['video_title']}")
                st.caption(f"Sentiment: **{row['sentiment']}** | Confidence: **{row['score']}**")

else:
    st.info("👈 Pick a topic in the sidebar and click **Fetch Live Data** to begin.")
