import spacy
import pandas as pd

def useful_ner():
    videos_final = pd.read_csv("videos_final.csv")
    nlp = spacy.load("en_core_web_sm")
    chosen_ner_labels = ["LOC", "GPE", "ORG", "PERSON", "WORK_OF_ART", "EVENT", "PRODUCT", "FAC", "NORP"]
    total_entities = []
    entities_df = pd.DataFrame({"entity": [], "label": []})
    text_list = videos_final["chunks_punctuation"].tolist()
    for text in text_list:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in chosen_ner_labels:
                new_rows = pd.DataFrame({"entity": [ent.text.lower()], "label": [ent.label_]})
                entities_df = pd.concat([entities_df, new_rows])
    entities_df.to_csv("../data/entities_df.csv", index = False)
