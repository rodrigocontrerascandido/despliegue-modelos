from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from proyecto_final.esquema import Cliente

# Antes de subir a Github era: NOMBRE_BUNDLE = "proyecto_final/modelo_clic_anuncio.joblib"
NOMBRE_BUNDLE = "modelo_clic_anuncio.joblib"

estado_servicio = {"bundle": None}

@asynccontextmanager
async def lifespan(app: FastAPI):

    estado_servicio["bundle"] = joblib.load(NOMBRE_BUNDLE)
    print("Bundle cargado correctamente")
    yield
    estado_servicio["bundle"] = None

app = FastAPI(
    title="API - Servicio de Pronóstico de Efectividad de Publicidad",
    description="Recibe datos de un cliente y predice la probabilidad de que de clic en el anuncio",
    version="1.0.0",
    lifespan = lifespan
    )

class ClienteInput(BaseModel):
    Tiempo_en_el_sitio: float = Field(..., description="Tiempo navegando en el sitio en minutos desde 1 minuto, no puede ser negativo"),
    Edad: int = Field(..., description="Edad del cliente desde 15 años, no puede ser negativo"),
    Ingresos_de_la_zona: float = Field(..., description="Ingresos de la zona del cliente desde 0, no puede ser negativo"),
    Uso_de_Internet: float = Field(..., description="Número de minutos que navega en internet desde 1 minuto, no puede ser negativo"),
    Sexo: Literal['Femenino', 'Masculino'] = Field(..., description="Sexo del cliente"),

class ClienteOutput(BaseModel):
    clic_predicho: int
    probabilidad: float
    resultado: str

@app.get("/") #Usamos metodo GET para traer informacion
def estado():
    return {
        "servicio":"API de servicio de predicción de efectividad de publicidad",
        "modelo_cargado": estado_servicio["bundle"] is not None
    }

#Predecir
@app.post("/predecir", response_model=ClienteOutput)
def predecir(cliente: Cliente):
    
    #Validar el modelo
    bundle = estado_servicio["bundle"]

    if bundle is None:
        raise HTTPException(status_code=503, detail="El modelo aún no está cargado")

    fila = cliente.model_dump()

     # Mapeo manual de la variable categórica
    map_sexo = {"Femenino": 0, "Masculino": 1}
    fila["Sexo"] = map_sexo[fila["Sexo"]]

    # Construir DataFrame con las columnas esperadas
    X_nuevo = pd.DataFrame([fila])[bundle["columnas"]]
    
    prediccion = bundle["pipeline"].predict(X_nuevo)[0]
    probabilidad = bundle["pipeline"].predict_proba(X_nuevo)[0, 1]

    #Devolver resultados
    return ClienteOutput(
        clic_predicho=int(prediccion),
        probabilidad=round(probabilidad,4),
        resultado="Favorable" if prediccion >= 0.5 else "Desfavorable"
    )