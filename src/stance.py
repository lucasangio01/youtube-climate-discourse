import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from lime.lime_text import LimeTextExplainer


class CreateEmbeddings:
    def __init__(self):
        self.videos_zeroshot = pd.read_csv("../data/videos_final_zeroshot.csv").drop(columns = ["video_id", "clim_change" ,"glob_warm","carb_emis","green_en_policies","fos_fu_ind","clim_activ","env_regul","clim_polic_debate","renew_en_trans","carb_tax", "text_toxicity"])
        self.text = self.videos_zeroshot["chunks_punctuation"].astype(str).tolist()
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def append_embeddings(self):
        embeddings = self.embedding_model.encode(self.text, show_progress_bar = True, convert_to_numpy = True).astype(float)
        self.videos_zeroshot["embedding"] = [json.dumps(emb.tolist()) for emb in embeddings]
        self.videos_zeroshot = pd.get_dummies(self.videos_zeroshot, columns = ["zeroshot_label"], drop_first = True)
        self.videos_zeroshot = self.videos_zeroshot.rename(columns = {"zeroshot_label_believes climate change is exaggerated": "zeroshot_believes_exaggeration"})
        self.videos_zeroshot.to_csv("../data/videos_embedded.csv", index = False)


class StanceClassifier:
    def __init__(self):
        self.videos_embedded = pd.read_csv("../data/videos_embedded.csv")
        self.videos_embedded["embedding"] = self.videos_embedded["embedding"].apply(lambda x: np.array(json.loads(x), dtype=float))
        self.lr = LogisticRegression(solver="liblinear", penalty="l2")
        self.X = np.vstack(self.videos_embedded["embedding"].values)
        self.y = self.videos_embedded["zeroshot_believes_exaggeration"]
        self.pca = PCA(n_components = 50)

    def prepare_data(self):
        self.X = self.pca.fit_transform(self.X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.2, stratify = self.y, random_state = 42)

    def logistic_regression(self):
        param_grid = {'C': np.logspace(-4, 4, 20), 'max_iter': [50, 100, 500]}
        grid_search = GridSearchCV(estimator = self.lr, param_grid = param_grid, cv = 5, scoring = 'accuracy')
        grid_search.fit(self.X_train, self.y_train)
        y_pred = grid_search.predict(self.X_test)
        best_params = grid_search.best_params_
        print("Test set report:\n")
        print(classification_report(self.y_test, y_pred))

        self.final_model = LogisticRegression(solver = "liblinear", penalty = "l2", C = best_params["C"], max_iter = best_params["max_iter"])
        self.final_model.fit(self.X, self.y)
        y_pred_full = self.final_model.predict(self.X)
        self.videos_embedded["lr_believes_exaggeration"] = y_pred_full
        print("Full-dataset report:\n")
        print(classification_report(self.y, y_pred_full))

        self.videos_embedded["embedding"] = self.videos_embedded["embedding"].apply(lambda x: json.dumps(x.tolist()))
        self.videos_embedded.to_csv("../data/videos_classified.csv", index = False)

    def lime_explanations_to_csv(self):
        lime_embedder = SentenceTransformer("all-MiniLM-L6-v2")
        videos_classified = pd.read_csv("../data/videos_classified.csv")

        def predict_proba(texts):
            embeddings = lime_embedder.encode(texts, show_progress_bar=False)
            reduced = self.pca.transform(embeddings)
            return self.final_model.predict_proba(reduced)

        explainer = LimeTextExplainer(class_names = ["believes climate change is exaggerated", "believes climate change is an urgent crisis"])
        rows = []
        for _, row in tqdm(videos_classified.iterrows(), total = len(videos_classified)):
            text = str(row["chunks_punctuation"])
            predicted_label = bool(row["lr_believes_exaggeration"])
            exp = explainer.explain_instance(text, predict_proba, num_features = 10, num_samples = 400)
            for word, weight in exp.as_list():
                if (predicted_label and weight > 0) or (not predicted_label and weight < 0):
                    rows.append({"lr_believes_exaggeration": predicted_label, "word": word, "magnitude": round(abs(weight), 4)})
        lime_words = pd.DataFrame(rows)
        lime_words.to_csv("../data/lime_words.csv", index = False)
