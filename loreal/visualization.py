import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st

class Visualizer:
    @staticmethod
    def plot_sentiment_pie(sent_counts):
        '''Generates a pie chart visualization of comment sentiment distribution.'''
        if len(sent_counts):
            fig = px.pie(values=sent_counts.values, names=sent_counts.index, title="Comment Sentiment")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No comments available for sentiment plot.")

    @staticmethod
    def plot_bar(x, y, x_label, y_label, title):
        '''Generates a bar chart visualization.'''
        fig = px.bar(x=x, y=y, labels={"x": x_label, "y": y_label}, title=title)
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def plot_wordcloud(text):
        '''Generates and displays a word cloud from the provided text.'''
        if text.strip():
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info("No text available for word cloud generation")
