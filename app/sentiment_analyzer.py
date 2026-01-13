import numpy as np
from transformers import pipeline
import torch

class SentimentAnalyzer:
    """Singleton class for Hugging Face sentiment analysis"""
    
    _instance = None
    _classifier = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SentimentAnalyzer, cls).__new__(cls)
        return cls._instance
    
    def get_classifier(self):
        """Lazy load the classifier"""
        if self._classifier is None:
            print("Loading Hugging Face sentiment model...")
            # Use CPU for compatibility
            self._classifier = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1  # Use CPU
            )
            print("Model loaded successfully!")
        return self._classifier

# Global analyzer instance
_analyzer = SentimentAnalyzer()

def analyze_sentiment(text):
    """
    Main function to analyze sentiment
    This is the function your main.py is calling
    """
    try:
        if not text or not text.strip():
            return {
                "success": False,
                "result": None,
                "error": "Text cannot be empty"
            }
        
        # Clean text
        text = text.strip()
        
        # Truncate if too long for the model
        if len(text) > 500:
            text = text[:500]
        
        # Get classifier and analyze
        classifier = _analyzer.get_classifier()
        result = classifier(text)[0]
        
        # Process result
        label = result['label'].lower()
        confidence = result['score']
        
        # Convert to numerical score (-1 to 1)
        if label == 'positive':
            score = 0.5 + (confidence * 0.5)  # 0.5 to 1.0
        elif label == 'negative':
            score = -0.5 - (confidence * 0.5)  # -0.5 to -1.0
        else:
            score = 0.0
        
        return {
            "success": True,
            "result": {
                "sentiment": {
                    "document": {
                        "score": round(score, 3),
                        "label": label,
                        "confidence": round(confidence, 3),
                        "raw_label": result['label']
                    }
                },
                "metadata": {
                    "model": "distilbert-base-uncased-finetuned-sst-2-english",
                    "is_free": True
                }
            },
            "error": None
        }
        
    except Exception as e:
        return {
            "success": False,
            "result": None,
            "error": str(e)
        }

def format_sentiment_result(result):
    """
    Format the result for display
    This is the function your main.py is calling
    """
    if not result.get("success"):
        return result
    
    sentiment_data = result["result"]["sentiment"]["document"]
    score = sentiment_data["score"]
    label = sentiment_data["label"]
    
    # Determine emoji and color
    if score > 0.3:
        emoji = "😊"
        color = "positive"
        sentiment_desc = "Positive"
    elif score < -0.3:
        emoji = "😞"
        color = "negative"
        sentiment_desc = "Negative"
    else:
        emoji = "😐"
        color = "neutral"
        sentiment_desc = "Neutral"
    
    # Format confidence as percentage
    confidence_pct = sentiment_data.get("confidence", 0.5) * 100
    
    return {
        "success": True,
        "sentiment": sentiment_desc,
        "score": f"{score:.3f}",
        "confidence": f"{confidence_pct:.1f}%",
        "raw_score": score,
        "raw_label": label,
        "emoji": emoji,
        "color": color,
        "details": result["result"],
        "is_free": True
    }