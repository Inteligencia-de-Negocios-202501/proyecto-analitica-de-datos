# BI-proyecto-noticias-falsas
En este repositorio se encuentra la implementación de la segunda etapa del proyecto relacionado con la clasificación de noticias falsas. 
¡Perfecto! Aquí tienes una guía paso a paso y clara para ejecutar una API hecha con **FastAPI** en local, **desde cero**, incluyendo la instalación del entorno, las dependencias y su ejecución:


## 📄 Documentación para ejecutar la API FastAPI en local

### 🧰 Requisitos previos
- Tener instalado **Python 3.8 o superior**.
- Tener instalado **pip** (el gestor de paquetes de Python).
- (Opcional) Tener instalado **`conda`** o **`venv`** para entornos virtuales.


### 🔹 1. Clonar el proyecto o descargar el código

Si el proyecto está en GitHub:

```bash
git clone https://github.com/Inteligencia-de-Negocios-202501/proyecto-analitica-de-datos.git
cd proyecto-analitica-de-datos/
```

### 🔹 2. Crear entorno virtual (recomendado)

Usando **`venv`**:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

O si usas **Conda**:

```bash
conda create -n miapi
conda activate miapi
```

### 🔹 3. Instalar dependencias

Si tienes el archivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 🔹 5. Ejecutar la API con Uvicorn

```bash
uvicorn app.api.main:app --reload
```

🔗 Accede en tu navegador a:  
**http://127.0.0.1:8000**

### 🔹 6. Documentación automática

FastAPI genera automáticamente documentación:

- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

### 🧼 Limpieza

Para desactivar el entorno virtual:

```bash
deactivate        # si usaste venv
conda deactivate  # si usaste conda
```