'''
Upload CSV of comments (with likes and replies), then run the app which uses BERTopic to extract topics from the comments.
App aggregates engagement metrics (likes, replies, comment count) by topic. 
We can modify the code and run the provided data when we get it. 

*** Make sure you're using Python 3.8-3.11 or Streamlit/BERTopic won't work ***
'''

import streamlit as st
import pandas as pd
from bertopic import BERTopic
import matplotlib.pyplot as plt

st.title("L'Oréal CommentSense: Topic & Engagement Analysis")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file with comments", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Sample data:", df.head())

    # Topic modeling
    st.subheader("Topic Modeling with BERTopic")
    with st.spinner("Extracting topics..."):
        topic_model = BERTopic()
        topics, probs = topic_model.fit_transform(df["comment_text"].astype(str).tolist())
        df["topic"] = topics

    # Show topic info
    topic_info = topic_model.get_topic_info()
    st.write("Topic summary:", topic_info)

    # Aggregate engagement metrics by topic
    agg = df.groupby("topic").agg({"num_replies": "sum", "num_likes": "sum", "comment_text": "count"}).rename(columns={"comment_text": "num_comments"})
    agg = agg.join(topic_info.set_index("Topic"), how="left")
    st.subheader("Engagement by Topic")
    st.write(agg)

    # Plot most liked topics
    st.subheader("Top Topics by Likes")
    top_likes = agg.sort_values("num_likes", ascending=False).head(5)
    fig, ax = plt.subplots()
    ax.barh(top_likes["Name"], top_likes["num_likes"])
    ax.set_xlabel("Total Likes")
    ax.set_ylabel("Topic")
    st.pyplot(fig)

    # Plot most replied topics
    st.subheader("Top Topics by Replies")
    top_replies = agg.sort_values("num_replies", ascending=False).head(5)
    fig2, ax2 = plt.subplots()
    ax2.barh(top_replies["Name"], top_replies["num_replies"])
    ax2.set_xlabel("Total Replies")
    ax2.set_ylabel("Topic")
    st.pyplot(fig2)

    st.info("Try uploading the provided comments_sample.csv to see a demo.")
else:
    st.info("Please upload a CSV file with columns: comment_text, num_replies, num_likes.")

