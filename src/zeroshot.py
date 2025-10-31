import pandas as pd
from transformers import pipeline

class ZeroShotStance:
    def __init__(self):
        self.stance_targets = ["believes climate change is an urgent crisis", "believes climate change is exaggerated"]
        self.zeroshot_pipeline = pipeline("zero-shot-classification", model = "mlburnham/Political_DEBATE_base_v1.0")
        self.videos_final = pd.read_csv("../data/videos_final.csv")

    def classify_stance(self, text):
        zeroshot_classifier = self.zeroshot_pipeline(text, self.stance_targets)
        return zeroshot_classifier["labels"][0], round(zeroshot_classifier["scores"][0], 3)

    def append_zeroshot_label(self):
        self.videos_final[["zeroshot_label", "zeroshot_conf"]] = self.videos_final["chunks_punctuation"].apply(lambda x: pd.Series(self.classify_stance(x)))
        self.videos_final.to_csv("../data/videos_final_zeroshot.csv", index = False)