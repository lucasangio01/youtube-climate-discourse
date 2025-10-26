import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModel

class ZeroShotStance:
    def __init__(self):
        self.stance_targets = ["believes climate change is an urgent crisis", "believes climate change is exaggerated"]
        self.zeroshot_pipeline = pipeline("zero-shot-classification", model = "mlburnham/Political_DEBATE_base_v1.0")
        self.videos_filtered = pd.read_csv("../data/videos_filtered.csv")

    def classify_stance(self, text):
        zeroshot_classifier = self.zeroshot_pipeline(text, self.stance_targets)
        return zeroshot_classifier["labels"][0], zeroshot_classifier["scores"][0]
    
    def append_zeroshot_label(self):
        self.videos_filtered[["zeroshot_label", "zeroshot_conf"]] = self.videos_filtered["chunks_punctuation"].apply(lambda x: pd.Series(self.classify_stance(x)))
        self.videos_filtered.to_csv("../data/videos_filtered_zeroshot.csv", index = False)
