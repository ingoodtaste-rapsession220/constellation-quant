"""NLP feature pipeline: embeddings, sentiment, language drift."""

from data_pipeline.nlp.embeddings import EmbeddingsModel
from data_pipeline.nlp.sentiment import SentimentModel
from data_pipeline.nlp.drift import compute_drift_features

__all__ = ["EmbeddingsModel", "SentimentModel", "compute_drift_features"]
