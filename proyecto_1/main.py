from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

NOMBRE_BUNDLE = "modelo_bundle_e_cardiaca.pkl"

estado_servicio = {"bundle": None}

#Cargar el bundle antes de la API
@asynccontextmanager
async def lifespan(app: FastAPI):

    estado_servicio["bundle"] = joblib.load(NOMBRE_BUNDLE)
    print("Bundle cargado correctamente")
    yield
    estado_servicio["bundle"] = None


# Configuración de la API
app = FastAPI(
    title="API de Predicción de Enfermedad Cardiaca",
    description="Recibe datos clínicos de un paciente y predice riesgo de cardiopatía coronaria (chd)",
    version="1.0.0",
    lifespan = lifespan
    )

# Clase para la entrada (valores que yo ingresaré al modelo para que este devuelva algo)
# Las llaves deben estar escritas tal cual como las columnas que se usaron para entrenar el modelo, de lo contrario no funcionará
# Esto lo ingresa el usuario
class PacienteInput(BaseModel):
    sbp: int = Field(..., description="Presión arterial sistólica"),
    Tabaco: float = Field(..., description="Tabaco acumulado (kg)"),
    ldl: float = Field(..., description="Colesterol LDL"),
    Adiposidad: float = Field(..., description="Adiposidad"),
    Familia: Literal['Presente','Ausente'] = Field(..., description="Antecedentes familiares de enfermedad cardiaca"),
    Tipo: int = Field(..., description="Comportamiento tipo-A"),
    Obesidad: float = Field(..., description="Obesidad"),
    Alcohol: float = Field(..., description="Consumo actual de alcohol"),
    Edad: int = Field(..., description="Edad")

#Clase para la salida (lo que el modelo me devuelve)
class PacienteOutput(BaseModel):
    chd_predicho: int
    probabilidad: float
    riesgo: str



# Construir el endpoint de verificación
@app.get("/")
def estado():
    return {
        "servicio": "API de Predicción de Enfermedad Cardiaca",
        "modelo_cargado": estado_servicio["bundle"] is not None
    }


# Construir el endpoint de predicción
#Predecir
@app.post("/predecir", response_model=PacienteOutput)
def predecir(paciente: PacienteInput):

    #Validar el modelo
    bundle = estado_servicio["bundle"]

    if bundle is None:
        raise HTTPException(status_code=503, detail="El modelo aún no está cargado")

    fila = paciente.model_dump()

    #Aplicar transformación de mapeo
    fila["Familia"] = bundle["mapeo_familia"][fila['Familia']]

    #Convertir en DataFrame el diccionario
    X_nuevo = pd.DataFrame([fila])[bundle["columnas"]]

    #Predicciones
    prediccion = bundle["pipeline"].predict(X_nuevo)[0]
    probabilidad = bundle["pipeline"].predict_proba(X_nuevo)[0,1]


    #Devolver resultados
    return PacienteOutput(
        chd_predicho=prediccion,
        probabilidad=round(probabilidad,4),
        riesgo="Alto" if prediccion == 1 else "Bajo"
    )


# El método Post nos permite enviar información, y que nos devuelva la predicción

