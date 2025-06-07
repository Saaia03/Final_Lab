from fastapi import FastAPI
import gradio as gr
from transformers import pipeline

# Инициализация модели
classifier = pipeline(
    "text-classification", 
    model="distilbert-base-uncased-finetuned-sst-2-english",
    framework="pt"
)

app = FastAPI()

def predict_sentiment(text):
    try:
        result = classifier(text)[0]
        label = result['label']
        confidence = result['score']
        
        label_ru = {
            'POSITIVE': 'позитивное',
            'NEGATIVE': 'негативное',
            'NEUTRAL': 'нейтральное'
        }.get(label, label)
        
        return f"Настроение: {label_ru}, Уверенность: {confidence:.2f}"
    
    except Exception as e:
        return f"Ошибка анализа: {str(e)}"

# Gradio интерфейс
iface = gr.Interface(
    fn=predict_sentiment,
    inputs="text",
    outputs="text",
    title="Анализатор настроения"
)

# Монтируем Gradio
app = gr.mount_gradio_app(app, iface, path="/gradio")

# GET-эндпоинт
@app.get("/api/get_analysis")
async def get_analysis(text: str):
    return {"result": predict_sentiment(text)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=7860)