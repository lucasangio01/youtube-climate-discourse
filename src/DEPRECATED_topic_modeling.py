import pandas as pd
import numpy as np
import ast
from bertopic import BERTopic 
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# THIS MODEL HAS NOT BEEN USED


class TopicModeling:
    def __init__(self):
        self.df = pd.read_csv("../data/videos_embedded.csv")
        self.text = self.df["chunks_punctuation"].tolist()
        self.embeddings = np.vstack(self.df["embedding"].apply(ast.literal_eval).values)
        self.wl = WordNetLemmatizer()
        self.basic_stopwords = set(stopwords.words('english'))
        self.custom_stopwords = set(["uh", "um", "like", "you", "know", "i", "mean", "so", "just", "gonna", "yeah", "right", "really", "talking", "going", "get"])
        self.full_stopwords = self.basic_stopwords.union(self.custom_stopwords)

    def remove_stopwords(self):
        corpus = []
        text_data = ""
        for i in range(self.df.shape[0]):
          text_data = re.sub('[^a-zA-Z]', ' ', self.text[i])
          text_data = text_data.lower()
          text_data = text_data.split()
          text_data = [self.wl.lemmatize(word) for word in text_data if not word in self.full_stopwords]
          text_data = ' '.join(text_data)
          corpus.append(text_data)  
        return corpus

    def model_topics(self):
        docs_clean = self.remove_stopwords()
        topic_model = BERTopic()
        topics, probs = topic_model.fit_transform(documents = docs_clean, embeddings = self.embeddings)
        topic_info = topic_model.get_topic_info()
        
        return topic_info, topic_model, topics, probs