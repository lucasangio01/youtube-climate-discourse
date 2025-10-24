from deepmultilingualpunctuation import PunctuationModel
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np


class PreprocessText:
    def __init__(self):
        self.punctuation_model = PunctuationModel()
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.videos_original = pd.read_csv("../data/videos_original.csv")
        self.chunk_size = 250
        self.climate_concepts = ["climate change", "global warming", "carbon emissions", "green energy policies", "fossil fuel industry", "climate activism", "environmental regulation", "climate policy debate", "renewable energy transition", "carbon tax"]
        self.climate_columns = ["clim_change", "glob_warm", "carb_emis", "green_en_policies", "fos_fu_ind", "clim_activ", "env_regul", "clim_polic_debate", "renew_en_trans", "carb_tax"]
        self.rogressive_speakers = ["Secular Talk", "David Pakman", "Democrat politicians"]
        self.conservative_speakers = ["Candace Owens", "Ben Shapiro", "Charlie Kirk", "Vivek Ramaswamy", "John Coleman", "Joe Rogan", "Dan Pena", "Republican politicians", "Donald Trump", "Fox News"]


    def split_into_chunks(self, text):
        words = text.split()
        return [" ".join(words[i:i + self.chunk_size]) for i in range(0, len(words), self.chunk_size)]

    def create_punctuation(self, text):
        return self.punctuation_model.restore_punctuation(text)

    def apply_chunking(self):
        videos_chunked = (self.videos_original.assign(chunks=self.videos_original["transcript"].apply(lambda text: self.split_into_chunks(text))).explode("chunks", ignore_index=True)).drop(columns = ["transcript"])
        videos_chunked = videos_chunked.to_csv("../data/videos_chunked.csv", index = False)
        return videos_chunked

    def apply_punctuation(self, videos_chunked):
        videos_punctuation = videos_chunked.copy()
        videos_punctuation["chunks_punctuation"] = videos_punctuation["chunks"].apply(self.create_punctuation)
        videos_punctuation = videos_punctuation.drop(columns=["chunks"])
        videos_punctuation.to_csv("../data/videos_punctuation.csv", index = False)
        return videos_punctuation

    def compute_similarity(self, videos_punctuation):
        embedded_concepts = self.embedding_model.encode(self.climate_concepts)
        embedded_chunks = self.embedding_model.encode(videos_punctuation["chunks_punctuation"])
        similarities = self.embedding_model.similarity(embedded_concepts, embedded_chunks).tolist()
        similarities_T = list(map(list, zip(*similarities)))
        similarities_T_rounded = [np.round(x, 3) for x in similarities_T]

        videos_punctuation_copy = videos_punctuation.copy()
        videos_similarity_raw = pd.DataFrame(similarities_T_rounded, columns=self.climate_columns, index=videos_punctuation_copy.index)

        videos_similarity = pd.concat([videos_punctuation_copy, videos_similarity_raw], axis=1).drop(["chunks_punctuation"], axis = 1)
        videos_similarity.to_csv("../data/videos_similarity.csv", index=False)

        return videos_similarity

    def filter_similarity(self, videos_similarity):
        videos_filtered = videos_similarity.copy()
        videos_filtered["counter_>_0.4"] = (videos_filtered.iloc[:, 3:] > 0.4).sum(axis=1)
        videos_filtered.to_csv("../data/videos_filtered.csv")
        return videos_filtered

    def run_pipeline(self):
        first_chunk = self.apply_chunking()
        second_addpunctuation = self.apply_punctuation(first_chunk)
        third_computesimilarity = self.compute_similarity(second_addpunctuation)
        fourth_filter = self.filter_similarity(third_computesimilarity)
