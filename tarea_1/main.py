from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

NOMBRE_BUNDLE = "tarea_1/modelo_churn.joblib"

estado_servicio = {"bundle": None}

@asynccontextmanager
async def lifespan(app: FastAPI):

    estado_servicio["bundle"] = joblib.load(NOMBRE_BUNDLE)
    print("Bundle cargado correctamente")
    yield
    estado_servicio["bundle"] = None


app = FastAPI(
    title="API - Servicio de Pronóstico de Cancelación de Clientes",
    description="Recibe datos de un cliente y predice la probabilidad de que cancele su suscripción",
    version="1.0.0",
    lifespan = lifespan
    )

class ClienteInput(BaseModel):
    antiguedad_meses: int = Field(..., description="Antiguedad en uso del servicio en meses"),
    gasto_mensual: float = Field(..., description="Gastos mensuales del cliente"),
    visitas_ultimo_mes: int = Field(..., description="Número de visitas del cliente en el último mes"),
    dias_desde_ultima_visita: int = Field(..., description="Número de días desde la última visita del cliente"),
    tickets_soporte: Literal['Sí', 'No'] = Field(..., description="Si cliente ha contactado al soporte técnico"),
    plan: Literal['Básico', 'Estándar', 'Premium'] = Field(..., description="Tipo de plan del cliente"),
    metodo_pago: Literal['Tarjeta de crédito', 'Efectivo', 'Transferencia bancaria'] = Field(..., description="Método de pago del cliente")
    descuento_activo: int = Field(..., description="Si el cliente tiene un descuento activo"),

class ClienteOutput(BaseModel):
    cancelo_predicho: int
    probabilidad: float
    riesgo: str


@app.get("/") #Usamos metodo GET para traer informacion
def estado():
    return {
        "servicio":"API de servicio de predicción de cancelación de clientes",
        "modelo_cargado": estado_servicio["bundle"] is not None
    }


#Predecir
@app.post("/predecir", response_model=ClienteOutput)
def predecir(cliente: ClienteInput):

    #Validar el modelo
    bundle = estado_servicio["bundle"]

    if bundle is None:
        raise HTTPException(status_code=503, detail="El modelo aún no está cargado")

    fila = cliente.model_dump()

    X_nuevo = pd.DataFrame([fila])[bundle["columnas"]]
    
    prediccion = bundle["pipeline"].predict(X_nuevo)[0]
    probabilidad = bundle["pipeline"].predict_proba(X_nuevo)[0, 1]

    return {
        ""
    }