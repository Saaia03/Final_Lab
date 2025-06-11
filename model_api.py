from fastapi import FastAPI
import gradio as gr
import joblib
import os

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
MODEL_PATH = 'sentiment_model/sentiment_model.pkl'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

def predict_sentiment(text):
    try:
        # Make prediction
        prediction = model.predict([text])[0]
        probability = model.predict_proba([text])[0]
        
        # Convert prediction to label
        label = 'positive' if prediction == 1 else 'negative'
        confidence = probability[prediction]
        
        # Translate to Russian
        label_ru = {
            'positive': 'позитивное',
            'negative': 'негативное'
        }.get(label, label)
        
        return f"Настроение: {label_ru}, Уверенность: {confidence:.2f}"
    
    except Exception as e:
        return f"Ошибка анализа: {str(e)}"

# Create Gradio interface
iface = gr.Interface(
    fn=predict_sentiment,
    inputs="text",
    outputs="text",
    title="Анализатор настроения"
)

# Mount Gradio app
app = gr.mount_gradio_app(app, iface, path="/gradio")

# API endpoint for sentiment analysis
@app.get("/api/get_analysis")
async def get_analysis(text: str):
    return {"result": predict_sentiment(text)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=7860)
