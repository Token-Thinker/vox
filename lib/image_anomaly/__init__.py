"""
Image anomaly detection package for detecting object or scene anomalies in camera frames.
Add a TFLite autoencoder or object detector model under models/ and implement the predictor.
"""
from .predictor import ImageAnomalyDetector