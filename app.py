'''
PROCESS:

1. Setup & Imports
- Loads all required libraries for data analysis, NLP, topic modeling, and visualization.
- Downloads necessary NLTK resources for text processing.

2. File Uploads
- Upload multiple comments CSVs and a single videos CSV.
- All comments files are concatenated; videos are loaded into a DataFrame.

3. Data Preparation
- Standardizes column names for consistency.
- Merges comments and videos on videoId to create a unified DataFrame.

4. Data Cleaning
- Filters out spam, links, emojis, hashtags, numbers, and inappropriate words from comments.

5. Text Preprocessing
- Cleans and tokenizes comment text, removes stopwords, and lemmatizes words.

6. Sentiment Analysis
- Uses TextBlob to label each comment as positive, neutral, or negative.

7. Relevance Scoring
- Calculates a custom relevance score for each comment based on likes, length, and word count.
- Normalizes scores and flags high-quality comments.

8. Engagement Metrics
- Computes Share of Engagement (SoE): TO BE REFORMULATED
- Merges SoE back into the main DataFrame.

9. Topic Extraction
- Parses Wikipedia URLs in topicCategories to extract readable topic names.
- Categorizes each video into business categories (makeup, skincare, haircare, fragrance, fashion, other).

10. Dashboard Visualizations
- Displays key metrics
- Shows sentiment distribution, comment quality by category, SoE by category, and top comments.
- Generates a word cloud of frequent terms.

11. Topic Modeling
- Runs BERTopic on high-quality comments to discover and visualize comment themes.

12. Data Export
- Offers the option to download the analyzed comments as a CSV file.

IMPORTANT:
Make sure you're using Python 3.8-3.11 or Streamlit/BERTopic won't work
'''
# =====================================
# Setup & Imports
# =====================================
import streamlit as st
import pandas as pd
from bertopic import BERTopic
import matplotlib.pyplot as plt
import plotly.express as px
import os
import ast
import ssl
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import download
# SSL workaround for macOS certificate issues :(
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from sentence_transformers import SentenceTransformer
from wordcloud import WordCloud
from textblob import TextBlob
os.environ["NUMBA_THREADING_LAYER"] = "tbb" # Fixes Numba threading error that I was getting

st.title("L'Oréal CommentSense: Topic & Engagement Analysis")

# =====================================
# File Uploads (comments + videos)
# =====================================

# Sidebar for multiple file uploads
st.sidebar.header("Upload CSV Files")
comments_files = st.sidebar.file_uploader("Upload comments CSV files", type=["csv"], accept_multiple_files=True)
videos_file = st.sidebar.file_uploader("Upload a videos CSV file", type=["csv"], key="videos")

if comments_files and videos_file:
    print(f"Uploaded {len(comments_files)} comments files and 1 videos file")

    # Concatenate all comments CSVs
    comments_dfs = [pd.read_csv(f) for f in comments_files]
    comments_df = pd.concat(comments_dfs, ignore_index=True)
    videos_df = pd.read_csv(videos_file)

    # =====================================
    # Standardize Column Names
    # =====================================

    # Rename comments columns 
    comments_df = comments_df.rename(columns={
        "likeCount": "comment_likeCount"
    })
    # Rename videos columns 
    videos_df = videos_df.rename(columns={
        "likeCount": "video_likeCount",
        "viewCount": "video_viewCount",
        "commentCount": "video_commentCount",
        "favouriteCount": "video_favCount"
    })

    print("Columns standardized")
    # DEBUG: Show raw data
    # st.write("Raw comments data:", comments_df.head())
    # print(comments_df.columns.tolist())
    # st.write("Raw videos data:", videos_df.head())
    # print(videos_df.columns.tolist())

    # =====================================
    # Merge Comments with Videos on videoId
    # =====================================
    merged_df = comments_df.merge(videos_df, on="videoId", how="left")
    # st.write("Sample merged data:", merged_df.head())
    df = merged_df
    print("Merged comments and videos data")

    # =====================================
    # Clean and Preprocess Data
    # =====================================
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

    # Remove (what else?)

    # DEBUG: Show filtered data
    st.write("Filtered data:", df.head())
    print("Data cleaned and preprocessed")

    # =====================================
    # Text Preprocessing
    # =====================================

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    def preprocess_text(text):
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    df['cleaned_text'] = df['textOriginal'].apply(preprocess_text)
    print("Text preprocessed")

    # =====================================
    # Sentiment Analysis
    # =====================================
    def get_sentiment(text):
        if not text or not isinstance(text, str):
            return "neutral"
        
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        
        if polarity > 0.1:
            return "positive"
        elif polarity < -0.1:
            return "negative"
        else:
            return "neutral"
    
    df['sentiment'] = df['textOriginal'].apply(get_sentiment)
    print("Sentiment analysis completed")
    
    # =====================================
    # Relevance Score Calculation
    # =====================================
    # Calculate engagement metrics
    df['comment_engagement'] = df['comment_likeCount'].fillna(0) + 1  # Add 1 to avoid division by zero
    
    # Calculate text quality metrics
    df['text_length'] = df['textOriginal'].str.len().fillna(0)
    df['word_count'] = df['textOriginal'].apply(lambda x: len(str(x).split()))
    
    # Calculate relevance score (customizable weights)
    like_weight = 0.5
    length_weight = 0.3
    word_weight = 0.2
    
    df['relevance_score'] = (
        like_weight * df['comment_engagement'] + 
        length_weight * df['text_length'] + 
        word_weight * df['word_count']
    )
    
    # Normalize to 0-100 scale
    min_score = df['relevance_score'].min()
    max_score = df['relevance_score'].max()
    if max_score > min_score:
        df['relevance_score_normalized'] = 100 * (df['relevance_score'] - min_score) / (max_score - min_score)
    else:
        df['relevance_score_normalized'] = 50  # Default value if all scores are same
    
    # Classify comments as high quality based on threshold
    quality_threshold = 70  # Adjustable threshold
    df['is_high_quality'] = df['relevance_score_normalized'] >= quality_threshold
    print("Relevance scores calculated")
    
    # =====================================
    # Share of Engagement (SoE) Calculation
    # TODO: DISCUSS AND REFORMULATE SoE CALCULATION
    # =====================================
    def calculate_soe(row):
        try:
            # Calculate total engagement
            total_engagement = (
                row['video_likeCount'] + 
                row['video_commentCount'] + 
                row['video_favCount']
            )
            
            # Avoid division by zero
            if row['video_viewCount'] > 0:
                return total_engagement / row['video_viewCount']
            else:
                return 0
        except:
            return 0
    
    videos_df['SoE'] = videos_df.apply(calculate_soe, axis=1)
    
    # Merge SoE back to main dataframe
    df = df.merge(videos_df[['videoId', 'SoE']], on='videoId', how='left')
    print("Share of Engagement calculated")
    
    # =====================================
    # Topic Extraction from Video Categories
    # =====================================
    def extract_topic_names(topic_categories):
        if not isinstance(topic_categories, str) or not topic_categories.strip():
            return ["Other"]
        try:
            urls = ast.literal_eval(topic_categories)
        except (SyntaxError, ValueError):
            return ["Other"]
        
        # Extract topic names from URLs
        topics = []
        for url in urls:
            if '/wiki/' in url:
                topic = url.split('/wiki/')[-1].replace('_', ' ')
                # Clean up topic names
                topic = re.sub(r'\([^)]*\)', '', topic)  # Remove parentheses content
                topic = topic.strip()
                topics.append(topic)
        
        return topics if topics else ["Other"]
    
    # Apply topic extraction
    df['video_topics'] = df['topicCategories'].apply(extract_topic_names)
    
    # Flatten topics for analysis
    topic_df = df.explode('video_topics')
    
    # Categorize topics into business categories
    # TODO: REFINE MAPPING
    category_mapping = {
        'makeup': ['cosmetics', 'makeup', 'beauty', 'cosmetic', 'make up'],
        'skincare': ['skincare', 'skin care', 'skin', 'facial', 'moisturizer'],
        'haircare': ['hair', 'haircare', 'hair care', 'shampoo', 'conditioner'],
        'fragrance': ['fragrance', 'perfume', 'scent', 'cologne', 'aroma'],
        'fashion': ['fashion', 'clothing', 'apparel', 'outfit', 'style']
    }
    
    def categorize_topic(topic):
        topic_lower = topic.lower()
        for category, keywords in category_mapping.items():
            for keyword in keywords:
                if keyword in topic_lower:
                    return category
        return 'other'
    
    df['business_category'] = df['video_topics'].apply(
        lambda topics: categorize_topic(topics[0]) if topics else 'other'
    )
    print("Topics extracted and categorized")

    # =====================================
    # Dashboard Layout
    # TODO: IMPROVE DASHBOARD COMPONENTS
    # =====================================
    st.header("CommentSense Dashboard")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_comments = len(df)
        st.metric("Total Comments", f"{total_comments:,}")
    
    with col2:
        high_quality_count = df['is_high_quality'].sum()
        quality_ratio = (high_quality_count / total_comments * 100) if total_comments > 0 else 0
        st.metric("High Quality Comments", f"{high_quality_count:,}", f"{quality_ratio:.1f}%")
    
    with col3:
        avg_soe = df['SoE'].mean() * 100 if 'SoE' in df.columns else 0
        st.metric("Avg Share of Engagement", f"{avg_soe:.2f}%")
    
    with col4:
        positive_sentiment = len(df[df['sentiment'] == 'positive'])
        positive_ratio = (positive_sentiment / total_comments * 100) if total_comments > 0 else 0
        st.metric("Positive Sentiment", f"{positive_sentiment:,}", f"{positive_ratio:.1f}%")
    
    # Sentiment Distribution
    st.subheader("Sentiment Analysis")
    sentiment_counts = df['sentiment'].value_counts()
    fig_sentiment = px.pie(
        values=sentiment_counts.values, 
        names=sentiment_counts.index,
        title="Comment Sentiment Distribution"
    )
    st.plotly_chart(fig_sentiment, use_container_width=True)
    
    # Quality Comments by Category
    st.subheader("Comment Quality by Business Category")
    if 'business_category' in df.columns and 'is_high_quality' in df.columns:
        quality_by_category = df.groupby('business_category')['is_high_quality'].mean() * 100
        fig_quality = px.bar(
            x=quality_by_category.index, 
            y=quality_by_category.values,
            labels={'x': 'Business Category', 'y': 'Percentage of High Quality Comments'},
            title="Percentage of High Quality Comments by Category"
        )
        st.plotly_chart(fig_quality, use_container_width=True)
    
    # SoE by Category
    st.subheader("Share of Engagement by Business Category")
    if 'business_category' in df.columns and 'SoE' in df.columns:
        soe_by_category = df.groupby('business_category')['SoE'].mean() * 100
        fig_soe = px.bar(
            x=soe_by_category.index, 
            y=soe_by_category.values,
            labels={'x': 'Business Category', 'y': 'Share of Engagement (%)'},
            title="Average Share of Engagement by Category"
        )
        st.plotly_chart(fig_soe, use_container_width=True)
    
    # Top Comments
    st.subheader("Top Comments by Relevance Score")
    top_comments = df.sort_values('relevance_score_normalized', ascending=False).head(10)
    for _, row in top_comments.iterrows():
        with st.expander(f"Comment (Score: {row['relevance_score_normalized']:.1f}, Likes: {row['comment_likeCount']})"):
            st.write(row['textOriginal'])
            st.caption(f"Sentiment: {row['sentiment']}, Category: {row.get('business_category', 'N/A')}")
    
    # =====================================
    # Topic Modeling with BERTopic
    # =====================================
    st.subheader("Topic Modeling of Comments")
    
    # Use only high-quality comments for topic modeling
    hq_comments = df[df['is_high_quality']]['cleaned_text'].dropna()
    
    if len(hq_comments) > 10:
        with st.spinner("Analyzing topics in high-quality comments..."):
            # Initialize and fit BERTopic model
            sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
            topic_model = BERTopic(embedding_model=sentence_model, verbose=True)
            
            # Fit the model
            topics, probabilities = topic_model.fit_transform(hq_comments.tolist())
            
            # Get topic info
            topic_info = topic_model.get_topic_info()
            
            # Display topic information
            st.write("Discovered Topics:")
            for index, row in topic_info.iterrows():
                if row['Topic'] != -1:  # Skip outlier topic
                    with st.expander(f"Topic {row['Topic']}: {row['Count']} comments"):
                        topic_words = topic_model.get_topic(row['Topic'])
                        words = [word for word, score in topic_words[:10]]
                        st.write("Top words: " + ", ".join(words))
            
            # Visualize topics
            fig = topic_model.visualize_topics()
            st.plotly_chart(fig, use_container_width=True)
            
            # Visualize barchart for the first few topics
            try:
                fig_barchart = topic_model.visualize_barchart(top_n_topics=10)
                st.plotly_chart(fig_barchart, use_container_width=True)
            except:
                st.info("Could not generate barchart visualization")
    else:
        st.warning("Not enough high-quality comments for topic modeling")
    
    # =====================================
    # Word Cloud for Most Frequent Terms
    # =====================================
    st.subheader("Most Frequent Terms in Comments")
    
    # Combine all text
    all_text = ' '.join(df['cleaned_text'].dropna().tolist())
    
    if all_text.strip():
        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
        
        # Display word cloud
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.info("No text available for word cloud generation")
    
    # =====================================
    # Data Export
    # =====================================
    st.subheader("Export Analyzed Data")
    
    if st.button("Download Analyzed Comments CSV"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="loreal_analyzed_comments.csv",
            mime="text/csv"
        )

else:
    st.info("Please upload comments and videos CSV files in the sidebar to begin analysis.")










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

