'''
IMPORTANT:
Make sure you're using Python 3.8-3.11 or Streamlit/BERTopic won't work
'''

# =====================================
# Setup & Imports
# =====================================
import os
os.environ["NUMBA_THREADING_LAYER"] = "tbb"

import numpy as np
import pandas as pd
import streamlit as st
from loreal.utils import build_video_topics, calc_soe_table, truncate_text
from loreal.data_loader import DataLoader
from loreal.analysis import Analyzer
from loreal.preprocessing import Preprocessor
from loreal.visualization import Visualizer


# =====================================
# Page Layout
# =====================================

# ---------- Streamlit Page Title----------
st.set_page_config(page_title="L'Oréal CommentSense", layout="wide")
st.title("L'Oréal CommentSense: Topic & Engagement Analysis")

# ----- Sidebar: File Uploads -----
st.sidebar.header("Upload CSV Files")
comments_files = st.sidebar.file_uploader("Upload comments CSV files", type=["csv"], accept_multiple_files=True)
videos_file = st.sidebar.file_uploader("Upload a videos CSV file", type=["csv"], key="videos")

# ----- Sidebar: Options -----
st.sidebar.header("Options")
min_comment_len = st.sidebar.number_input("Minimum comment length (chars)", 5, 500, value=5, step=5)
top_n_comments  = st.sidebar.slider("Top N comments to show", 10, 200, 50, 5)
truncate_n      = st.sidebar.slider("Truncate comment label length", 40, 160, 90, 5)
quality_threshold = st.sidebar.slider("High-quality threshold (normalized score)", 0, 100, 70, 1)


# =====================================
# Main Analysis
# =====================================
if comments_files and videos_file:

    # ----- Load & standardise -----
    comments_df = DataLoader.load_comments(comments_files)
    videos_df   = DataLoader.load_videos(videos_file)
    comments_df = DataLoader.standardize_comments(comments_df)
    videos_df   = DataLoader.standardize_videos(videos_df)

    # Show initial count
    st.write(f"Original comments: {len(comments_df):,}")

    # ----- Clean comments (BATCHED version as we have 5 million rows) -----
    # Neutral anti-spam rules; but also avoid biased keyword lists as it might lead to unintended exclusion of valid comments
    st.write("Filtering comments...")
    progress_bar = st.progress(0)
    
    # Pre-compile regex patterns for faster matching
    import re
    spam_pattern = re.compile(r"http|www\.|bit\.ly|free money|subscribe|click here", re.IGNORECASE)
    emoji_only_pattern = re.compile(r"^[\W_]+$")
    tag_pattern = re.compile(r"^[@#].*")
    numbers_only_pattern = re.compile(r"^[0-9]+$")
    
    # Process in batches
    batch_size = 100000  # Adjust based on memory constraints
    filtered_batches = []
    total_batches = (len(comments_df) // batch_size) + 1
    
    for batch_num, i in enumerate(range(0, len(comments_df), batch_size)):
        batch = comments_df.iloc[i:i+batch_size].copy()
        text_batch = batch["textOriginal"].astype(str)
        
        # Create mask for this batch using pre-compiled patterns
        mask = (
            (text_batch.str.len() >= min_comment_len) &
            (~text_batch.apply(lambda x: bool(spam_pattern.search(x)) if pd.notna(x) else False)) &
            (~text_batch.apply(lambda x: bool(emoji_only_pattern.match(x)) if pd.notna(x) else False)) &
            (~text_batch.apply(lambda x: bool(tag_pattern.match(x)) if pd.notna(x) else False)) &
            (~text_batch.apply(lambda x: bool(numbers_only_pattern.match(x)) if pd.notna(x) else False))
        )
        
        filtered_batches.append(batch[mask])
        progress_bar.progress(min((i + batch_size) / len(comments_df), 1.0))
        # st.write(f"Processed batch {batch_num + 1}/{total_batches}")

    # Combine filtered batches
    if filtered_batches:
        comments_df = pd.concat(filtered_batches, ignore_index=True)
    else:
        comments_df = comments_df.iloc[0:0]  # Empty DataFrame
    
    st.write(f"After spam filtering: {len(comments_df):,} comments")
    st.write("Filtered data:", comments_df.head())
    
    # Early exit if no data remains
    if len(comments_df) == 0:
        st.warning("No comments remaining after filtering. Please adjust filter criteria.")
        st.stop()

    # ----- Prune columns early for speed -----
    # (analysis is reaaaally slow otherwise)
    comments_df = comments_df[[
        "videoId","commentId","textOriginal","comment_likeCount"
    ]].copy()
    
    # Optimize data types
    comments_df["comment_likeCount"] = comments_df["comment_likeCount"].fillna(0).astype(int)
    comments_df["videoId"] = comments_df["videoId"].astype("category")
    comments_df["commentId"] = comments_df["commentId"].astype("category")

    videos_df = videos_df[[
        "videoId","title","description","tags","topicCategories",
        "video_likeCount","video_commentCount","video_favCount","video_viewCount"
    ]].copy()
    
    # Only keep videos that have comments
    videos_df = videos_df[videos_df["videoId"].isin(comments_df["videoId"].unique())]

    # ----- Merge comments with videos (on videoId) -----
    df = comments_df.merge(videos_df, on="videoId", how="left")
    st.write("Sample merged data:", df.head())

    # ----- Text preprocessing -----
    st.write("Cleaning text...")
    progress_bar = st.progress(0)
    
    cleaned_texts = []
    for i in range(0, len(df), batch_size):
        batch = df["textOriginal"].iloc[i:i+batch_size]
        cleaned_batch = [Preprocessor.clean_comment(text) for text in batch]
        cleaned_texts.extend(cleaned_batch)
        progress_bar.progress(min((i + batch_size) / len(df), 1.0))
    
    df["cleaned_text"] = cleaned_texts
    print("Comments after cleaning:\n", df["cleaned_text"].head())

    # ----- Topics: compute ONCE per video, then merge -----
    video_topics = build_video_topics(videos_df)
    df = df.merge(video_topics, on="videoId", how="left")
    st.write("Topic samples:", video_topics.head())
    print("Video topics sample:", video_topics.head())

    # ----- Sentiment Analysis WITH PROGRESS BAR -----
    st.write("Analyzing sentiment...")
    progress_bar = st.progress(0)

    batch_size = 5000  # Smaller batch for sentiment analysis
    sentiments = []
    for i in range(0, len(df), batch_size):
        batch = df["textOriginal"].iloc[i:i+batch_size]
        batch_sentiments = [Analyzer.get_sentiment(text) for text in batch]
        sentiments.extend(batch_sentiments)
        progress_bar.progress(min((i + batch_size) / len(df), 1.0))
        # st.write(f"Sentiment analysis: {min(i + batch_size, len(df)):,}/{len(df):,} comments")
    
    df["sentiment"] = sentiments

    # ----- SoE: compute ONCE per video, then merge (EXAMPLE ONLY) -----
    # TODO: REFORMULATE THIS
    per_video = (videos_df.groupby("videoId", as_index=False)
                 .agg(video_likeCount=("video_likeCount","max"),
                      video_commentCount=("video_commentCount","max"),
                      video_favCount=("video_favCount","max"),
                      video_viewCount=("video_viewCount","max")))
    per_video_soe = calc_soe_table(per_video)
    df = df.merge(per_video_soe, on="videoId", how="left")

    # ----- Relevance scoring (EXAMPLE ONLY) -----
    # TODO: REFORMULATE THIS
    # Individual factors
    df["comment_len"] = df["textOriginal"].astype(str).str.len()
    df["word_count"]  = df["textOriginal"].astype(str).str.split().str.len()
    df["likes"]    = df["comment_likeCount"].fillna(0).astype(float) + 1.0
    # Weights
    w_like, w_len, w_words = 1.0, 0.01, 0.5
    # Sum of weighted factors
    df["relevance_raw"] = w_like*df["likes"] + w_len*df["comment_len"] + w_words*df["word_count"]

    # ----- Topic-aware ranking -----
    st.write("Processing topic-aware ranking...")

    # Keep only relevant columns for topic-level scoring
    topic_df = df[[
        "videoId","textOriginal","comment_likeCount","SoE","relevance_raw","video_topics","business_category_primary"
    ]].copy()

    # Each comment may have multiple ontology topics (list)
    # Thus, need to explode so each (comment × topic) becomes its own row
    # Explode in batches to avoid memory issues
    exploded_batches = []
    for i in range(0, len(topic_df), batch_size):
        batch = topic_df.iloc[i:i+batch_size].copy()
        exploded_batch = batch.explode("video_topics", ignore_index=True)
        exploded_batches.append(exploded_batch)
    
    topic_df = pd.concat(exploded_batches, ignore_index=True)

    # Fill missing topic labels with "Other" to avoid NaNs
    topic_df["video_topics"] = topic_df["video_topics"].fillna("Other")

    # ----- Relevance normalization within each topic -----
    # Scale raw relevance scores (calculated in relation to topic group), range 0-100
    # Process normalization in smaller groups to avoid memory issues
    unique_topics = topic_df["video_topics"].unique()
    progress_bar = st.progress(0)
    
    for i, topic in enumerate(unique_topics):
        topic_mask = topic_df["video_topics"] == topic
        topic_subset = topic_df[topic_mask]
        
        if len(topic_subset) > 0:
            # Normalize relevance within topic
            topic_df.loc[topic_mask, "relevance_norm_topic"] = Analyzer.minmax_0_100(
                topic_subset["relevance_raw"]
            )
            
            # Calculate SoE percentile
            topic_df.loc[topic_mask, "soe_pct_topic"] = topic_subset["SoE"].rank(pct=True).fillna(0.5)
        
        progress_bar.progress((i + 1) / len(unique_topics))

    # ----- SoE percentile within each topic -----
    # Compute the percentile rank of each video's SoE inside that topic group, range 0–1
    # Fill NaN with 0.5 if group is empty so that it is neutral
    soe_weight = 0.2
    topic_df["soe_pct_topic"] = topic_df.groupby("video_topics")["SoE"] \
                                        .transform(lambda s: s.rank(pct=True)).fillna(0.5)

    # ----- Final topic-aware relevance score -----
    # Combine normalized relevance and SoE percentile into one score
    # Formula: 80% comment relevance + 20% SoE influence (as suggested by ChatGPT)
    # TODO: REFORMULATE THIS
    topic_df["relevance_topicaware"] = (1 - soe_weight) * topic_df["relevance_norm_topic"] \
                                       + soe_weight * (topic_df["soe_pct_topic"] * 100)

    # ----- Ranking comments within each topic -----
    # Assign rank (1 = best) by topic-aware score for each topic group
    topic_df["rank_within_topic"] = topic_df.groupby("video_topics")["relevance_topicaware"] \
                                            .rank(ascending=False, method="dense")
    
    # ----- High-quality flag -----
    # Mark comments as high-quality if their topic-aware score passes the sidebar-defined threshold (quality_threshold)
    HQ_THRESH = quality_threshold 
    topic_df["is_high_quality_by_topic"] = topic_df["relevance_topicaware"] >= HQ_THRESH


    # =====================================
    # Dashboard
    # =====================================
    st.header("CommentSense Dashboard")
    
    # Sample data for visualization 
    sample_size = min(100000, len(topic_df)) # Apparently charts with >100k data points can freeze the browser
    if len(topic_df) > sample_size:
        st.warning(f"Sampling {sample_size:,} records from {len(topic_df):,} for visualization")
        viz_topic_df = topic_df.sample(sample_size, random_state=42)
    else:
        viz_topic_df = topic_df

    # ----- Metrics -----
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Comments (post-clean)", f"{len(df):,}")
    with col2:
        hq_count = int(topic_df["is_high_quality_by_topic"].sum())
        hq_ratio = 100.0 * hq_count / max(1, len(topic_df))
        st.metric("High-Quality (topic-aware)", f"{hq_count:,}", f"{hq_ratio:.1f}%")
    with col3:
        avg_soe = 100.0 * per_video_soe["SoE"].mean() if len(per_video_soe) else 0.0
        st.metric("Avg SoE (videos)", f"{avg_soe:.2f}%")
    with col4:
        pos = int((df["sentiment"] == "positive").sum())
        pos_ratio = 100.0 * pos / max(1, len(df))
        st.metric("Positive Sentiment", f"{pos:,}", f"{pos_ratio:.1f}%")

    # SENTIMENT DISTRIBUTION
    st.subheader("Sentiment Distribution")
    Visualizer.plot_sentiment_pie(df["sentiment"].value_counts())

    # QUALITY BY BUSINESS CATEGORY
    # Something wrong with this. Not picking up any categories except Science/Ingredients
    st.subheader("High-Quality % by Business Category")
    cat_quality = (viz_topic_df.groupby("business_category_primary")["is_high_quality_by_topic"]
                   .mean().mul(100).sort_values(ascending=False))
    Visualizer.plot_bar(
        cat_quality.index, 
        cat_quality.values,
        "Business Category", 
        "% High-Quality (topic-aware)",
        "High-Quality Comments by Category"
    )

    # SoE BY CATEGORY
    st.subheader("Average SoE by Category (Videos)")
    # Join category to per_video_soe via video_topics table to get business_category_primary
    per_video_with_cat = per_video_soe.merge(
        video_topics[["videoId","business_category_primary"]].drop_duplicates("videoId"),
        on="videoId", how="left"
    )
    soe_cat = per_video_with_cat.groupby("business_category_primary")["SoE"] \
                                .mean().mul(100).sort_values(ascending=False)
    Visualizer.plot_bar(
        soe_cat.index, 
        soe_cat.values,
        "Business Category", "SoE (%)",
        "Average Share of Engagement by Category")

    # TOP COMMENTS (topic-aware)
    # i.e. Relevant to the video's topic
    st.subheader("Top Comments (Topic-Aware Relevance)")
    viz_topic_df["topic_for_video"] = viz_topic_df["video_topics"]
    topic_choice = st.selectbox("Filter by topic",
                                ["All"] + sorted(viz_topic_df["topic_for_video"].dropna().unique().tolist()))
    subset = viz_topic_df if topic_choice == "All" else viz_topic_df[viz_topic_df["topic_for_video"] == topic_choice]
    topc = subset.sort_values(["relevance_topicaware","comment_likeCount"],
                              ascending=[False, False]).head(top_n_comments).copy()
    topc["short_text"] = truncate_text(topc["textOriginal"], truncate_n)
    # Table
    st.dataframe(
        topc[["topic_for_video","videoId","short_text","comment_likeCount","SoE","relevance_topicaware","rank_within_topic"]],
        use_container_width=True
    )
    # Bar chart
    Visualizer.plot_bar(
        topc.sort_values("relevance_topicaware")["relevance_topicaware"],
        topc.sort_values("relevance_topicaware")["short_text"],
        "Topic-Aware Score (0–100)", "Comment",
        "Top Comments by Topic-Aware Relevance"
    )

    # Tool to export data
    st.subheader("Export Analyzed Data")
    if st.button("Download Analyzed Comments CSV"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV", 
            data=csv,
            file_name="loreal_analyzed_comments.csv", 
            mime="text/csv")
else:
    st.info("Please upload comments and videos CSV files in the sidebar to begin analysis.")



# Problem: Some videos come back as Other, e.g.:
# youtube#video,85806,2024-01-15 00:59:29+00:00,33807,Unlocking the Benefits of Face Masks for Skin Health,,,en-US,en-US,PT9S,72.0,0.0,0.0,0.0,"['https://en.wikipedia.org/wiki/Health', 'https://en.wikipedia.org/wiki/Lifestyle_(sociology)']"
# youtube#video,30556,2023-10-27 19:32:16+00:00,46650,Get ready for the Magic💚💜🤍💝✨ #hydration #glowingskin #nomakeuplook #skincare,,,,,PT45S,257.0,7.0,0.0,0.0,"['https://en.wikipedia.org/wiki/Lifestyle_(sociology)', 'https://en.wikipedia.org/wiki/Physical_attractiveness']"
# youtube#video,43611,2023-04-29 18:47:37+00:00,8143,Full Face of Merit Beauty 🤎 featuring new Flush Balm Shades! #merit #sephora #makeuptutorial,,,,en,PT56S,8647.0,268.0,0.0,7.0,"['https://en.wikipedia.org/wiki/Lifestyle_(sociology)', 'https://en.wikipedia.org/wiki/Physical_attractiveness']"




#     # =====================================
#     # Topic Extraction from videos.topicCategories
#     #     - parse Wikipedia URLs -> readable titles
#     #     - choose primary topic per video
#     #     - bucket to business categories (Makeup/Skincare/Haircare/Fragrance/Fashion/Other)
#     # =====================================
#     def extract_topic_names(topic_categories):
#         if not isinstance(topic_categories, str) or not topic_categories.strip():
#             return ["Other"]
#         try:
#             urls = ast.literal_eval(topic_categories)
#         except (SyntaxError, ValueError):
#             return ["Other"]
#         return [url.split("/wiki/")[-1].replace("_", " ") for url in urls]
    
#     print("Unique topicCategories values:", merged_df["topicCategories"].unique())

#     # Add readable topic names to merged_df
#     merged_df["video_topics"] = merged_df["topicCategories"].apply(lambda x: extract_topic_names(x)[0] if extract_topic_names(x) else "Other")

#     # DEBUG: Show extracted topics
#     st.write("Extracted video topics:", merged_df[["videoId", "video_topics"]].drop_duplicates().head())

#     # Aggregate SoE by topic 
#     if "SoE" in merged_df.columns:
#         topic_soe = merged_df.groupby("video_topics")["SoE"].mean().sort_values(ascending=False)
#         st.subheader("Average Share of Engagement (SoE) by Topic")
#         st.write(topic_soe)
#         fig_topic_soe = px.bar(topic_soe, x=topic_soe.index, y=topic_soe.values, labels={"x": "Topic", "y": "Avg SoE"}, title="Avg SoE by Topic")
#         st.plotly_chart(fig_topic_soe)

#     # =====================================
#     # Per-Comment Features
#     #    - relevance_score 
#     #    - normalized relevance
#     # =====================================

#     # Calculate: 

#     # Calculate: Final relevance score 
#     # TODO: REFINE SCORING METRIC
#     df["relevance_score"] = (
#         df["comment_likeCount"].fillna(0) * 2 +  # EXAMPLE ONLY: Likes are weighted more
#         df["textOriginal"].str.len().fillna(0) * 0.01  # EXAMPLE ONLY: Longer comments get a small boost
#     )

#     # Normalize relevance score to 0-100 for easier comparison
#     min_score = df["relevance_score"].min()
#     max_score = df["relevance_score"].max()
#     df["relevance_score_normalized"] = 100 * (df["relevance_score"] - min_score) / (max_score - min_score)

#     # =====================================
#     # Top Comments View
#     #    - table + bar chart of most relevant comments
#     # =====================================

#     # Show top comments by normalized relevance score
#     st.subheader("Top Comments by Normalized Relevance Score")
#     top_comments = df.sort_values("relevance_score_normalized", ascending=False).head(50)
#     # Table: Top comments by relevance score
#     st.write(top_comments[["videoId","textOriginal", "comment_likeCount", "relevance_score_normalized"]]) 
#     # Bar chart: Top comments by relevance score (ascending order)
#     top_comments = top_comments.sort_values("relevance_score_normalized", ascending=True)
#     fig = px.bar(
#         top_comments,
#         x="relevance_score_normalized",
#         y="textOriginal",
#         orientation="h",
#         title="Top Comments by Normalized Relevance Score"
#     )
#     st.plotly_chart(fig)

#     # =====================================
#     # Aggregate Comment-Quality Metrics per Video
#     #    - n_comments
#     #    - median comment length
#     #    - mean comment likes
#     #    - % high-quality comments
#     # =====================================
#     st.subheader("Comment Metrics")

#     # =====================================
#     # Compute SoE (Share of Engagement)
#     #    - E.g. (video_likeCount + video_commentCount + video_favCount) / video_viewCount
#     # ================================
#     st.subheader("Share of Engagement (SoE)")

#     # =====================================
#     # Topic-aware SoE Table
#     #     - ensure topic columns carried into the per-video SoE table
#     # =====================================

#     # =====================================
#     # Topic-Level Aggregates & Visuals
#     #     - roll up SoE and Comment Quality by topic buckets
#     #     - charts: Avg SoE by topic; Avg % High-Quality by topic
#     # =====================================

#     # =====================================
#     # Join SoE × Comment Quality
#     #     - final per-video dataset
#     #     - scatter (SoE vs quality)
#     #     - bar (top videos by % quality)
#     # =====================================
#     st.subheader("Engagement vs Comment Quality")


#     # =====================================
#     # Sentiment Analysis
#     #     - label comments (positive/neutral/negative)
#     #     - aggregate % sentiment per video
#     # =====================================
#     st.subheader("Sentiment Analysis")


#     # =====================================
#     # Topic Modeling (BERTopic)
#     #     - run on top comments only
#     #     - show clusters of themes
#     # =====================================
#     st.subheader("Topic Modeling with BERTopic (Top Comments Only)")
#     texts = top_comments["textOriginal"].dropna().astype(str)

#     # DEBUG
#     st.write("Number of non-empty top comments for topic modeling:", len(texts))
#     st.write("Top comments for topic modeling:", texts.tolist())

#     if len(texts) < 5:
#         st.warning("Not enough comments for meaningful topic modeling. Please upload more data.") # Doesn't seem to work with too few comments  
#     else:
#         with st.spinner("Extracting topics..."):
#             topic_model = BERTopic()
#             topics, probs = topic_model.fit_transform(texts.tolist())
#             top_comments.loc[texts.index, "topic"] = topics
#             topic_info = topic_model.get_topic_info()
#             st.write("Topic summary (Top Comments Only):", topic_info)


# else:
#     st.info("Please upload comments and videos CSV files in the sidebar.")

