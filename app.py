'''
Make sure you're using Python 3.8-3.11 or Streamlit/BERTopic won't work
'''

import streamlit as st
import pandas as pd
from bertopic import BERTopic
import matplotlib.pyplot as plt
import plotly.express as px
import os
os.environ["NUMBA_THREADING_LAYER"] = "tbb" # Should fix Numba threading error???

st.title("L'Oréal CommentSense: Topic & Engagement Analysis")

# Sidebar for multiple file uploads
st.sidebar.header("Upload CSV Files")
comments_files = st.sidebar.file_uploader("Upload comments CSV files", type=["csv"], accept_multiple_files=True)
videos_file = st.sidebar.file_uploader("Upload a videos CSV file", type=["csv"], key="videos")

if comments_files and videos_file:
    # Concatenate all comments CSVs
    comments_dfs = [pd.read_csv(f) for f in comments_files]
    comments_df = pd.concat(comments_dfs, ignore_index=True)
    videos_df = pd.read_csv(videos_file)
    # Merge comments with videos on videoId
    merged_df = comments_df.merge(videos_df, on="videoId", how="left")
    st.write("Sample merged data:", merged_df.head())
    # Use merged_df for all further filtering, scoring, and analysis
    df = merged_df

    # ----- Filter data: Remove spam/irrelevant comments -----
    # Comments with links or spammy phrases
    df = df[~df["textOriginal"].str.contains("http|www\\.|bit\\.ly|free money|subscribe|click here", case=False, na=False)] 
    # Comments that consist only of non-word characters (e.g., all emojis, punctuation)
    df = df[~df["textOriginal"].str.match("^[\W_]+$", na=False)]
    # Comments with only hashtags/mentions
    df = df[~df["textOriginal"].str.match(r"^[@#].*", na=False)]
    # Comments with only numbers
    df = df[~df["textOriginal"].str.match("^[0-9]+$", na=False)]
    # Comments with bad words
    df = df[~df["textOriginal"].str.contains("lesbian|morebadwords|morebadwords|morebadwords", case=False, na=False)] 
    # Others?

#     # ----- Get most relevant comments -----

#     # Calculate relevance score using likes and comment length
#     df["relevance_score"] = (
#         df["likeCount"].fillna(0) * 2 +  # Likes are weighted more
#         df["textOriginal"].str.len().fillna(0) * 0.01  # Longer comments get a small boost
#     )

#     # Normalize relevance score to 0-100 for easier comparison
#     min_score = df["relevance_score"].min()
#     max_score = df["relevance_score"].max()
#     df["relevance_score_normalized"] = 100 * (df["relevance_score"] - min_score) / (max_score - min_score)

#     # Show top comments by normalized relevance score
#     st.subheader("Top Comments by Normalized Relevance Score")
#     top_comments = df.sort_values("relevance_score_normalized", ascending=False).head(50)
#     # Table: Top comments by relevance score
#     st.write(top_comments[["textOriginal", "likeCount", "relevance_score_normalized"]]) 
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

#     # ----- Topic Modeling & Engagement Analysis -----

#     # Topic modeling with BERTopic (only on top comments)
#     st.subheader("Topic Modeling with BERTopic (Top Comments Only)")
#     texts = top_comments["textOriginal"].dropna().astype(str)

#     # DEBUG
#     # st.write("Number of non-empty top comments for topic modeling:", len(texts))
#     # st.write("Top comments for topic modeling:", texts.tolist())

#     if len(texts) < 5:
#         st.warning("Not enough comments for meaningful topic modeling. Please upload more data.")
#     else:
#         with st.spinner("Extracting topics..."):
#             topic_model = BERTopic()
#             topics, probs = topic_model.fit_transform(texts.tolist())
#             top_comments.loc[texts.index, "topic"] = topics
#             topic_info = topic_model.get_topic_info()
#             st.write("Topic summary (Top Comments Only):", topic_info)

#     # Merge comments with videos on videoId
#     merged_df = df.merge(videos_df, on="videoId", how="left")
#     st.write("Merged comments and videos data:", merged_df.head())
#     # You can now use merged_df for further analysis, scoring, and topic modeling

else:
    st.info("Please upload comments and videos CSV files in the sidebar.")

