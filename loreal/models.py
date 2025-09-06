from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

class TopicModeler:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.model = BERTopic(embedding_model=self.embedding_model, verbose=False)

    def fit_transform(self, texts):
        '''Fits the BERTopic model to the provided texts and transforms them into topic representations.'''
        return self.model.fit_transform(texts)

    def get_topic_info(self):
        '''Retrieves information about the topics discovered by the BERTopic model.'''
        return self.model.get_topic_info()

    def get_topic(self, topic_id):
        '''Retrieves the most relevant words for a specific topic.'''
        return self.model.get_topic(topic_id)

    def visualize_topics(self):
        '''Generates a visualization of the topics discovered by the BERTopic model.'''
        return self.model.visualize_topics()

    def visualize_barchart(self, top_n_topics=10):
        '''Generates a bar chart visualization of the top N topics.'''
        return self.model.visualize_barchart(top_n_topics=top_n_topics)
