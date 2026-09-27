from contextlib import asynccontextmanager
from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ============================================================
# CONFIGURACIÓN MLFLOW
# ============================================================

mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)

MODELO_URI = (
    "models:/modelo-enfermedad-cardiaca@champion"
)


# ============================================================
# ESTADO DEL MODELO
# ============================================================

modelo = None


# ============================================================
# CARGAR MODELO AL INICIAR FASTAPI
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global modelo

    modelo = mlflow.sklearn.load_model(
        MODELO_URI
    )

    print("Modelo champion cargado correctamente")

    yield

    modelo = None


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="API Enfermedad Cardíaca",
    version="2.0.0",
    lifespan=lifespan
)


# ============================================================
# ENTRADA
# ============================================================

class PacienteInput(BaseModel):

    sbp: int
    Tabaco: float
    ldl: float
    Adiposidad: float
    Familia: Literal["Presente", "Ausente"]
    Tipo: int
    Obesidad: float
    Alcohol: float
    Edad: int


# ============================================================
# SALIDA
# ============================================================

class PacienteOutput(BaseModel):

    chd_predicho: int
    probabilidad: float
    riesgo: str


# ============================================================
# HEALTH
# ============================================================

@app.get("/")
def estado():

    return {
        "servicio": "API Enfermedad Cardíaca",
        "modelo_cargado": modelo is not None,
        "modelo": "champion"
    }


# ============================================================
# PREDICCIÓN
# ============================================================

@app.post(
    "/predecir",
    response_model=PacienteOutput
)
def predecir(
    paciente: PacienteInput
):

    if modelo is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no cargado"
        )

    # Convertimos Pydantic a diccionario
    fila = paciente.model_dump()


    # Transformación que usábamos
    # durante el entrenamiento
    fila["Familia"] = (
        1
        if fila["Familia"] == "Presente"
        else 0
    )


    # Convertir a DataFrame
    X_nuevo = pd.DataFrame(
        [fila]
    )


    # Predecir
    prediccion = modelo.predict(
        X_nuevo
    )[0]


    probabilidad = modelo.predict_proba(
        X_nuevo
    )[0, 1]


    return PacienteOutput(
        chd_predicho=int(prediccion),
        probabilidad=round(
            float(probabilidad),
            4
        ),
        riesgo=(
            "alto"
            if prediccion == 1
            else "bajo"
        )
    )