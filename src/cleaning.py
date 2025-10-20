from deepmultilingualpunctuation import PunctuationModel
from sentence_transformers import SentenceTransformer
import pandas as pd


punctuation_model = PunctuationModel()
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def split_into_chunks(text):
    chunk_size = 250
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def create_punctuation(text):
    return punctuation_model.restore_punctuation(text)


def apply_chunking(videos_original):
    videos_chunked = (videos_original.assign(chunks=videos_original["transcript"].apply(lambda text: split_into_chunks(text))).explode("chunks", ignore_index=True)).drop(columns = ["transcript"])
    videos_chunked = videos_chunked.to_csv("../data/videos_chunked.csv", index = False)
    return videos_chunked


def apply_punctuation(videos_chunked):
    videos_punctuation = videos_chunked.copy()
    videos_punctuation["chunks_punctuation"] = videos_punctuation["chunks"].apply(create_punctuation)
    videos_punctuation = videos_punctuation.drop(columns=["chunks"])
    videos_punctuation = videos_punctuation.to_csv("../data/videos_punctuation.csv", index = False)
    return videos_punctuation


def compute_similarity(videos_punctuation):
    climate_concepts = ["climate change", "global warming", "carbon emissions", "green energy policies", "fossil fuel industry", "climate activism", "environmental regulation", "climate policy debate", "renewable energy transition", "carbon tax"]
    climate_columns = ["clim_change", "glob_warm", "carb_emis", "green_en_policies", "fos_fu_ind", "clim_activ", "env_regul", "clim_polic_debate", "renew_en_trans", "carb_tax"]

    embedded_concepts = embedding_model.encode(climate_concepts)
    embedded_chunks = embedding_model.encode(videos_punctuation["chunks_punctuation"])
    similarities = embedding_model.similarity(embedded_concepts, embedded_chunks).tolist()
    similarities_T = list(map(list, zip(*similarities)))

    videos_similarity = videos_punctuation.copy()
    videos_similarity_full = pd.DataFrame(similarities_T, columns=climate_columns, index=videos_similarity.index)

    concat_df = pd.concat([videos_similarity, videos_similarity_full], axis=1)
    concat_df.to_csv("../data/videos_similarity.csv", index=False)

    return concat_df
