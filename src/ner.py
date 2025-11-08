import spacy
import pandas as pd
from tqdm import tqdm


def useful_ner():
    videos_classified = pd.read_csv("../data/videos_classified.csv")
    nlp = spacy.load("en_core_web_sm")
    chosen_ner_labels = ["LOC", "GPE", "ORG", "PERSON", "WORK_OF_ART", "NORP"]
    total_entities = []
    entities_df = pd.DataFrame({"entity": [], "label": [], "stance": []})
    text_list = videos_classified["chunks_punctuation"].tolist()
    stance_list = videos_classified["lr_believes_exaggeration"].tolist()
    for text, stance in tqdm(zip(text_list, stance_list), total=len(text_list)):
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in chosen_ner_labels:
                total_entities.append({"entity": ent.text.lower(), "label": ent.label_, "lr_believes_exaggeration": stance})

    entities_df = pd.DataFrame(total_entities)
    entities_df.to_csv("../data/entities_df.csv", index=False)
