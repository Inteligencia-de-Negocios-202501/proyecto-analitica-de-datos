# ==================== Importaciones ====================
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import os
from app.api.utils import transfor_data_local, createModel, loadModel

# ==================== Inicializar FastAPI ====================
app = FastAPI()

# Servir archivos estáticos desde la carpeta /static
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Ruta del modelo entrenado
MODEL_PATH = "app/models/Predictor.joblib"

# Cargar modelo entrenado
try:
    modelo = loadModel(MODEL_PATH)
except Exception as e:
    modelo = None

# ==================== Esquemas Pydantic ====================
class NewsInput(BaseModel):
    Titulo: str
    Descripcion: str

class RetrainInput(BaseModel):
    Titulo: str
    Descripcion: str
    Label: int

# ==================== Función para Obtener el Modelo ====================
def get_model():
    """
    Verifica si el modelo está cargado. Si no, lo vuelve a cargar.
    """
    global modelo
    if modelo is None:
        try:
            modelo = loadModel(MODEL_PATH)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al cargar modelo: {str(e)}")
    return modelo

# ==================== Endpoint para Servir Página ====================
@app.get("/")
def serve_index():
    """Sirve el archivo index.html en la raíz."""
    return FileResponse("app/static/pagina.html")

# ==================== Endpoint para Predicción ====================
@app.post("/predict/")
def predict_news(news_list: List[NewsInput]):
    """
    Recibe una lista de noticias y devuelve predicciones.
    """
    modelo = get_model()
    
    # Convertir datos a DataFrame
    if isinstance(news_list, list):
        df = pd.DataFrame([item.dict() for item in news_list])
    else:
        df = pd.DataFrame([news_list.dict()])

    predicciones = modelo.predict(df)
    probabilidades = modelo.predict_proba(df).tolist()

    resultados = [{
        "Titulo": row.Titulo,
        "Descripcion": row.Descripcion,
        "Prediccion": "Real" if pred == 1 else "Falsa",
        "Probabilidad": prob
    } for row, pred, prob in zip(news_list, predicciones, probabilidades)]

    return resultados

# ==================== Endpoint para Predicción desde CSV ====================
@app.post("/predict_csv/")
async def predict_csv(file: UploadFile = File(...)):
    """
    Recibe un archivo CSV sin la columna 'Label' y devuelve predicciones
    con probabilidades para cada noticia.
    """
    # Verificar si el archivo es un CSV
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV.")
    
    # Guardar archivo temporal
    temp_path = "app/data/temp_predict.csv"
    with open(temp_path, "wb") as f:
        f.write(file.file.read())

    # Cargar el archivo CSV
    try:
        df = pd.read_csv(temp_path, sep=";")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer el archivo CSV: {str(e)}")

    # Validar columnas requeridas
    required_cols = {"Titulo", "Descripcion"}
    if not required_cols.issubset(df.columns):
        raise HTTPException(status_code=400, detail="El archivo debe contener las columnas: Titulo y Descripcion.")


    # Verificar si el modelo está disponible
    modelo = get_model()

    # Realizar predicción
    try:
        predicciones = modelo.predict(df)
        probabilidades = modelo.predict_proba(df).tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")

    # Construir resultados
    resultados = [{
        "Titulo": row.Titulo,
        "Descripcion": row.Descripcion,
        "Prediccion": "Real" if pred == 1 else "Falsa",
        "Probabilidad": {
            "Real": round(prob[1], 4),
            "Falsa": round(prob[0], 4)
        }
    } for row, pred, prob in zip(df.itertuples(), predicciones, probabilidades)]
    
    return {"predicciones": resultados}

# ==================== Endpoint para Re-entrenamiento ====================
@app.post("/retrain/")
async def retrain_model(file: UploadFile = File(...)):
    """
    Recibe un archivo CSV y reentrena el modelo.
    Devuelve métricas de evaluación.
    """
    # Verificar si el archivo es CSV
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV.")
    
    # Guardar archivo temporal
    temp_path = "app/data/temp_retrain.csv"
    with open(temp_path, "wb") as f:
        f.write(file.file.read())

    # Cargar el archivo CSV
    df = pd.read_csv(temp_path, sep=";")

    # Validar columnas requeridas
    required_cols = {"Titulo", "Descripcion", "Label"}
    if not required_cols.issubset(df.columns):
        raise HTTPException(status_code=400, detail="El archivo CSV debe contener las columnas: Titulo, Descripcion y Label.")

    # Procesar los datos para reentrenamiento
    df['Texto_Procesado'] = transfor_data_local(df)

    # Cargar vectorizador y crear nuevo modelo
    from sklearn.feature_extraction.text import TfidfVectorizer
    text_transformer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

    # Entrenar nuevo modelo
    results, nuevo_modelo = createModel(text_transformer, df)

    # Guardar nuevo modelo
    joblib.dump(nuevo_modelo, MODEL_PATH)
    global modelo
    modelo = nuevo_modelo

    return {"message": "Modelo actualizado correctamente", "metrics": results}



