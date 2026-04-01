import torch
from transformers import pipeline

class SentimentModel:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.classifier = pipeline("sentiment-analysis", model=model_name)

    def predict_sentiment(self, text):
        result = self.classifier(text)[0]
        return result["label"], result["score"]

if __name__ == "__main__":
    model = SentimentModel()
    text = "This is an amazing product!"
    label, score = model.predict_sentiment(text)
    print(f"Text: {text}, Label: {label}, Score: {score}")
