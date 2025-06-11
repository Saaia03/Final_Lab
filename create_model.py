import json
import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def load_data(json_path):
    """Загрузка и предварительная обработка данных"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Проверка структуры данных
    if not {'text', 'sentiment'}.issubset(df.columns):
        raise ValueError("JSON должен содержать поля 'text' и 'sentiment'")
    
    # Удаление строк с пропущенными значениями
    df = df.dropna(subset=['text', 'sentiment'])
    
    # Преобразование меток и фильтрация некорректных значений
    valid_sentiments = {'negative', 'positive'}
    df = df[df['sentiment'].isin(valid_sentiments)]  # Оставляем только корректные метки
    
    df['label'] = df['sentiment'].map({'negative': 0, 'positive': 1})
    
    # Дополнительная проверка на NaN после преобразования
    if df['label'].isna().any():
        raise ValueError("Обнаружены некорректные значения в метках после преобразования")
    
    return df['text'].values, df['label'].values

def train_model(X, y):
    """Обучение модели"""
    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            token_pattern=r'\b\w+\b'
        )),
        ('clf', LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            class_weight='balanced'  # Для несбалансированных данных
        ))
    ])
    
    model.fit(X, y)
    return model

def save_model(model, model_dir='saved_model'):
    """Сохранение модели и компонентов"""
    os.makedirs(model_dir, exist_ok=True)
    
    # Сохраняем всю модель (пайплайн)
    joblib.dump(model, f'{model_dir}/sentiment_model.pkl')
    
    # Сохраняем информацию о данных для воспроизводимости
    with open(f'{model_dir}/metadata.json', 'w') as f:
        json.dump({
            'model_type': 'LogisticRegression',
            'features': 'text',
            'target': 'sentiment',
            'classes': ['negative', 'positive']
        }, f)
    
    print(f"Модель и метаданные сохранены в {model_dir}/")

if __name__ == "__main__":
    # Параметры
    JSON_PATH = 'train.json'
    MODEL_DIR = 'sentiment_model'
    
    try:
        # Загрузка данных с проверкой
        print("Загрузка и проверка данных...")
        X, y = load_data(JSON_PATH)
        
        # Проверка размера данных
        if len(X) == 0:
            raise ValueError("Нет данных для обучения после предварительной обработки")
        
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Обучение модели
        print("Обучение модели...")
        model = train_model(X_train, y_train)
        
        # Оценка
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        print(f"Точность модели: train={train_score:.3f}, test={test_score:.3f}")
        
        # Сохранение
        save_model(model, MODEL_DIR)
        
    except Exception as e:
        print(f"Ошибка: {str(e)}")
        print("Проверьте:")
        print("1. Структуру JSON-файла (должны быть поля 'text' и 'sentiment')")
        print("2. Значения в поле 'sentiment' (должны быть 'negative' или 'positive')")
        print("3. Отсутствие пропущенных значений (NaN) в данных")