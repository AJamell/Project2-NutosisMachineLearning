import numpy as np
import pandas as pd
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

nlp = spacy.load("en_core_web_lg")

behavior_columns = [
    "Running", "Chasing", "Climbing", "Eating", "Foraging", "Kuks", "Quaas", "Moans",
    "Tail flags", "Tail twitches", "Approaches", "Indifferent", "Runs from",
]
highlight_colors = ["White", "Cinnamon", "Gray", "Black"]
behavior_text_columns = ["Specific Location", "Other Activities", "Other Interactions"]
aggression_concept = "aggressive behavior, attacking, biting"
play_concept = "playful behavior, playing"
injury_pattern = r"injur|wound|no tail|missing tail|\.5 of a tail"
num_columns = ["X", "Y", "day", "log_height", "aggression_sim", "play_sim"]
weather_columns = ["prev_tmin_c", "prev_tmax_c", "log_precip_mm"]
event_columns = ["park_event_count", "nearby_event_count", "log_nearest_event_m"]
cat_columns = ["Primary Fur Color", "Age", "Location", "Shift"]
flag_columns = behavior_columns + [
    "hl_white", "hl_cinnamon", "hl_gray", "hl_black",
    "has_behavior_text", "tail_or_injury", "mentions_white",
]



def texts_to_vectors(texts):
    cleaned = texts.fillna("").astype(str).tolist()
    vectors = []
    for doc in nlp.pipe(cleaned):
        tokens = [t for t in doc if t.has_vector and not t.is_punct]
        if tokens:
            vectors.append(np.mean([t.vector for t in tokens], axis=0))
        else:
            vectors.append(np.zeros(nlp.vocab.vectors_length))
    return np.array(vectors)


class CleanColumns(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()

        data["day"] = data["Date"].dt.day
        data["Age"] = data["Age"].replace("?", np.nan)

        highlight = data["Highlight Fur Color"].fillna("")
        for color in highlight_colors:
            data["hl_" + color.lower()] = highlight.str.contains(color).astype(int)

        height = data["Above Ground Sighter Measurement"].replace("FALSE", "0")
        data["log_height"] = np.log1p(pd.to_numeric(height, errors="coerce"))

        data["log_precip_mm"] = np.log1p(data["prev_precip_mm"])
        data["log_nearest_event_m"] = np.log1p(data["nearest_event_meters"])

        data[behavior_columns] = data[behavior_columns].astype(int)
        return data


class NoteFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()

        behavior_notes = data[behavior_text_columns].fillna("").agg(" ".join, axis=1).str.strip()
        has_note = behavior_notes != ""
        data["has_behavior_text"] = has_note.astype(int)

        data["aggression_sim"] = 0.0
        data["play_sim"] = 0.0
        if has_note.any():
            note_vectors = texts_to_vectors(behavior_notes[has_note])
            concept_vectors = texts_to_vectors(pd.Series([aggression_concept, play_concept]))
            similarities = cosine_similarity(note_vectors, concept_vectors)
            data.loc[has_note, "aggression_sim"] = similarities[:, 0]
            data.loc[has_note, "play_sim"] = similarities[:, 1]

        all_notes = data[["Color notes"] + behavior_text_columns].fillna("").agg(" ".join, axis=1)
        data["tail_or_injury"] = all_notes.str.contains(injury_pattern, case=False).astype(int)

        color_notes = data["Color notes"].fillna("").str.lower()
        data["mentions_white"] = color_notes.str.contains(r"\bwhite\b").astype(int)
        return data

def numeric_steps():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])


def category_steps():
    return Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])


def make_preprocessor():
    column_steps = ColumnTransformer(
        [
            ("num", numeric_steps(), num_columns),
            ("weather", numeric_steps(), weather_columns),
            ("events", numeric_steps(), event_columns),
            ("cat", category_steps(), cat_columns),
            ("flags", "passthrough", flag_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    column_steps.set_output(transform="pandas")

    preprocessor = Pipeline([
        ("clean", CleanColumns()),
        ("notes", NoteFeatures()),
        ("columns", column_steps),
    ])
    return preprocessor
