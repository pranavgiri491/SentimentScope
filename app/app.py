import os
import traceback
from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Global model variable
sentiment_model = None

def load_model():
    """Load the sentiment analysis model"""
    global sentiment_model
    if sentiment_model is None:
        try:
            print("🔄 Loading Hugging Face sentiment model...")
            sentiment_model = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print(traceback.format_exc())
            sentiment_model = None
    return sentiment_model

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze sentiment from text"""
    try:
        # Get text from form
        text = request.form.get('text', '').strip()
        print(f"📝 Received text: {text[:50]}...")
        
        if not text:
            return render_template('result.html', 
                                 error="Please enter some text to analyze.")
        
        # Load model
        model = load_model()
        if model is None:
            return render_template('result.html',
                                 error="AI model failed to load. Please check console.")
        
        # Analyze sentiment
        print("🤖 Analyzing sentiment...")
        try:
            # Limit text length for the model
            text_to_analyze = text[:500] if len(text) > 500 else text
            results = model(text_to_analyze)
            print(f"✅ Analysis results: {results}")
            
            if not results:
                return render_template('result.html',
                                     error="No results returned from model.")
            
            result = results[0]  # Get first result
            label = result['label'].lower()
            confidence = result['score']
            
            # Convert to our format
            if label == 'positive':
                score = 0.5 + (confidence * 0.5)  # 0.5 to 1.0
                emoji = "😊"
                color = "positive"
                sentiment = "Positive"
            elif label == 'negative':
                score = -0.5 - (confidence * 0.5)  # -0.5 to -1.0
                emoji = "😞"
                color = "negative"
                sentiment = "Negative"
            else:
                score = 0.0
                emoji = "😐"
                color = "neutral"
                sentiment = "Neutral"
            
            # Create result dictionary
            formatted_result = {
                'success': True,
                'sentiment': sentiment,
                'label': label,
                'score': round(score, 3),
                'confidence': round(confidence * 100, 2),
                'emoji': emoji,
                'color': color,
                'raw_score': confidence
            }
            
            print(f"📊 Formatted result: {formatted_result}")
            
            return render_template('result.html',
                                 text=text[:200],
                                 result=formatted_result)
            
        except Exception as model_error:
            print(f"❌ Model error: {model_error}")
            print(traceback.format_exc())
            return render_template('result.html',
                                 error=f"Model analysis failed: {str(model_error)}")
        
    except Exception as e:
        print(f"❌ General error: {e}")
        print(traceback.format_exc())
        return render_template('result.html',
                             error=f"An error occurred: {str(e)}")

@app.route('/health')
def health():
    """Health check endpoint"""
    model = load_model()
    status = "healthy" if model else "unhealthy"
    return jsonify({
        "status": status,
        "model_loaded": model is not None,
        "service": "sentiment-analysis"
    })

if __name__ == '__main__':
    print("🚀 Starting Sentiment Analysis Web App")
    print(f"📁 Current directory: {os.getcwd()}")
    
    # Create templates folder if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("📁 Created templates folder")
    
    # Pre-load model on startup (optional)
    print("⏳ Pre-loading model...")
    load_model()
    
    print("🌐 Server starting on http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)