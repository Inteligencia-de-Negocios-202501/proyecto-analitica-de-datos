# ==================== Importaciones ====================
import pandas as pd
import joblib
import unicodedata
import re
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk import word_tokenize
import nltk

# ==================== Descargar recursos NLTK ====================
nltk.download('stopwords')
nltk.download('punkt')

# ==================== Configuración de stopwords y stemmer ====================
stop_words = stopwords.words('spanish')
stemmer = SnowballStemmer("spanish")

# ==================== Funciones de Preprocesamiento ====================
def remove_non_ascii(words):
    return [unicodedata.normalize('NFKD', w).encode('ascii', 'ignore').decode('utf-8', 'ignore') for w in words if w]

def normalize_text(text):
    text = text.replace('á', 'a').replace('é', 'e').replace('í', 'i')
    text = text.replace('ó', 'o').replace('ú', 'u').replace('ü', 'u')
    text = text.replace('ñ', 'n')
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

def to_lowercase(words):
    return [w.lower() for w in words]

def remove_punctuation(words):
    return [re.sub(r'[^\w\s]', '', w) for w in words if w]

def replace_numbers(words):
    from num2words import num2words
    return [num2words(w, lang='es') if w.isdigit() else w for w in words]

def remove_stopwords(words):
    return [w for w in words if w not in stop_words]

def stem_words(words):
    return [stemmer.stem(w) for w in words]

def preprocessing(text):
    words = word_tokenize(text)
    words = remove_punctuation(words)
    words = to_lowercase(words)
    words = replace_numbers(words)
    words = remove_non_ascii(words)
    words = remove_stopwords(words)
    words = stem_words(words)
    return ' '.join(words)

def transfor_data_local(df):
    """Aplicar preprocesamiento a columnas 'Titulo' y 'Descripcion'."""
    columnas = ['Titulo', 'Descripcion', ]

    # Verificar que las columnas requeridas existan
    for col in columnas:
        if col not in df.columns:
            raise ValueError(f"La columna {col} no existe en el DataFrame.")
    
    df[columnas] = df[columnas].fillna('')
    
    # Aplicar preprocesamiento
    for columna in columnas:
        df[columna] = df[columna].apply(preprocessing)
    
    # Concatenar ambas columnas
    df['Texto_Procesado'] = df['Titulo'] + ' ' + df['Descripcion']
    
    return df['Texto_Procesado']

# ==================== Función para Crear y Entrenar el Modelo ====================
def createModel(text_transformer, df):
    """Crear y entrenar el modelo usando un pipeline."""
    # Asegúrate de que las columnas necesarias existan
    for col in ['Titulo', 'Descripcion', 'Label']:
        if col not in df.columns:
            raise ValueError(f"La columna {col} es requerida en el DataFrame.")
    
    # Eliminar columnas irrelevantes antes del entrenamiento
    df = df.drop(columns=['ID', 'Fecha'], errors='ignore')
    
    # Eliminar filas con valores nulos
    df = df.dropna(subset=['Titulo', 'Descripcion', 'Label'])
    
    # Crear conjuntos de entrenamiento y prueba
    X = df[['Titulo', 'Descripcion']]
    y = df['Label']
    
    # Transformador para aplicar preprocesamiento sobre el DataFrame
    data_transformer = FunctionTransformer(transfor_data_local)
    
    # Dividir datos para entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Modelo base: RandomForest
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=10,
        random_state=42
    )
    
    # Definición del pipeline
    pipeline = Pipeline([
        ("data_transform", data_transformer),
        ("vectorizer", text_transformer),
        ("classifier", model)
    ])
    
    # Entrenar el modelo
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Métricas de evaluación
    results = classification_report(y_test, y_pred, output_dict=True)
    print("Reporte de Clasificación:\n", classification_report(y_test, y_pred))
    
    return results, pipeline


# ==================== Función para Cargar Modelo ====================
def loadModel(MODEL_PATH):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo no encontrado en {MODEL_PATH}")
    return joblib.load(MODEL_PATH)
