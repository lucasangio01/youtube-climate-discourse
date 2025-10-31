from deepmultilingualpunctuation import PunctuationModel
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from detoxify import Detoxify


class PreprocessText:
    def __init__(self):
        self.punctuation_model = PunctuationModel()
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.detox_model = Detoxify("unbiased")
        self.videos_original = pd.read_csv("../data/videos_original.csv")
        self.chunk_size = 250
        self.climate_concepts = ["climate change", "global warming", "carbon emissions", "green energy policies", "fossil fuel industry", "climate activism", "environmental regulation", "climate policy debate", "renewable energy transition", "carbon tax"]
        self.climate_columns = ["clim_change", "glob_warm", "carb_emis", "green_en_policies", "fos_fu_ind", "clim_activ", "env_regul", "clim_polic_debate", "renew_en_trans", "carb_tax"]
        self.progressive_speakers = ["Secular Talk", "David Pakman", "Democrat politicians"]
        self.conservative_speakers = ["Candace Owens", "Ben Shapiro", "Charlie Kirk", "Vivek Ramaswamy", "John Coleman", "Joe Rogan", "Dan Pena", "Republican politicians", "Donald Trump", "Fox News"]

    def split_into_chunks(self, text):
        words = text.split()
        return [" ".join(words[i:i + self.chunk_size]) for i in range(0, len(words), self.chunk_size)]

    def create_punctuation(self, text):
        return self.punctuation_model.restore_punctuation(text)

    def apply_chunking(self):
        self.videos_chunked = (self.videos_original.assign(chunks=self.videos_original["transcript"].apply(lambda text: self.split_into_chunks(text))).explode("chunks", ignore_index=True)).drop(columns = ["transcript"])
        self.videos_chunked = self.videos_chunked.assign(ideology=["progressive" if x in self.progressive_speakers else "conservative" for x in self.videos_chunked["channel"]])

    def apply_punctuation(self):
        self.videos_punctuation = self.videos_chunked.copy()
        self.videos_punctuation["chunks_punctuation"] = self.videos_punctuation["chunks"].apply(self.create_punctuation)
        self.videos_punctuation = self.videos_punctuation.drop(columns=["chunks"])

    def compute_similarity(self):
        embedded_concepts = self.embedding_model.encode(self.climate_concepts)
        embedded_chunks = self.embedding_model.encode(self.videos_punctuation["chunks_punctuation"])
        similarities = self.embedding_model.similarity(embedded_concepts, embedded_chunks).tolist()
        similarities_T = list(map(list, zip(*similarities)))
        similarities_T_rounded = [np.round(x, 3) for x in similarities_T]

        videos_punctuation_copy = self.videos_punctuation.copy()
        videos_similarity_raw = pd.DataFrame(similarities_T_rounded, columns=self.climate_columns, index=videos_punctuation_copy.index)

        self.videos_similarity = pd.concat([videos_punctuation_copy, videos_similarity_raw], axis=1)

    def filter_similarity(self):
        self.videos_filtered = self.videos_similarity.copy()
        self.videos_filtered["counter_>_0.4"] = (self.videos_filtered.iloc[:, 5:] > 0.4).sum(axis=1)
        self.videos_filtered = self.videos_filtered[self.videos_filtered["counter_>_0.4"] > 0]
        self.videos_filtered = self.videos_filtered.drop(columns = ["counter_>_0.4"])

    def classify_toxicity(self):
        videos_final = self.videos_filtered.copy()
        toxicity_dict =  self.detox_model.predict(videos_final["chunks_punctuation"].tolist())
        toxicity_scores = [round(score, 3) for score in toxicity_dict["toxicity"]]
        videos_final["text_toxicity"] = toxicity_scores
        videos_final.to_csv("../data/videos_final.csv", index = False)

    def run_pipeline(self):
        first_chunk = self.apply_chunking()
        second_addpunctuation = self.apply_punctuation(first_chunk)
        third_computesimilarity = self.compute_similarity(second_addpunctuation)
        fourth_filter = self.filter_similarity(third_computesimilarity)
        fifth_toxicity = self.classify_toxicity(fourth_filter)