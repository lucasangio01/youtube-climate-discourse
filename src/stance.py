import pandas as pd
import numpy as np
import ast
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA


class CreateEmbeddings:
    def __init__(self):
        self.df = pd.read_csv("../data/videos_filtered_zeroshot.csv").drop(columns = ["video_id", "clim_change" ,"glob_warm","carb_emis","green_en_policies","fos_fu_ind","clim_activ","env_regul","clim_polic_debate","renew_en_trans","carb_tax"])
        self.text = self.df["chunks_punctuation"].astype(str).tolist()
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def embed_chunks(self):
        return self.model.encode(self.text, show_progress_bar = True, convert_to_numpy = True).astype(float)
    
    def append_embeddings(self):
        embeddings = self.embed_chunks()
        self.df["embedding"] = [emb.tolist() for emb in embeddings]
        self.df = pd.get_dummies(self.df, columns = ["zeroshot_label"], drop_first = True)
        self.df = self.df.rename(columns = {"zeroshot_label_believes climate change is exaggerated": "believes_exaggeration"})
        self.df.to_csv("../data/videos_embedded.csv", index = False)

class StanceClassifier:
    def __init__(self):
        self.df = pd.read_csv("../data/videos_embedded.csv")
        self.df["embedding"] = self.df["embedding"].apply(lambda x: np.array(ast.literal_eval(x), dtype=float))
        self.lr = LogisticRegression()
        self.X = np.vstack(self.df["embedding"].values)
        self.y = self.df["believes_exaggeration"]
        self.pca = PCA(n_components = 50)
        self.X = self.pca.fit_transform(self.X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.2, stratify = self.y, random_state = 42)

    def logistic_regression(self):
        param_grid = {'C': np.logspace(-4, 4, 20), 'max_iter': [50, 100, 500]}
        grid_search = GridSearchCV(estimator=self.lr, param_grid=param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)
        y_pred = grid_search.predict(self.X_test)
        print("Best Hyperparameters:", grid_search.best_params_)
        print("Best Cross-Validation Score:", grid_search.best_score_)
        print(classification_report(self.y_test, y_pred))
        return y_pred
