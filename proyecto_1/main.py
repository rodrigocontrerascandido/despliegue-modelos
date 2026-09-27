from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

NOMBRE_BUNDLE = "modelo_bundle_e_cardiaca.pkl"
estado_servicio = {"bundle": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    estado_servicio["bundle"] = joblib.load(NOMBRE_BUNDLE)
    print("Bundle cargado correctamente")
    yield
    estado_servicio["bundle"] = None


app = FastAPI(
    title="API de Predicción de Enfermedad Cardíaca",
    description=(
        "Recibe datos clínicos de un paciente y predice riesgo "
        "de cardiopatía coronaria (chd)"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class PacienteInput(BaseModel):
    sbp: int = Field(..., description="Presión arterial sistólica")
    Tabaco: float = Field(..., description="Tabaco acumulado (kg)")
    ldl: float = Field(..., description="Colesterol LDL")
    Adiposidad: float = Field(..., description="Adiposidad")
    Familia: Literal["Presente", "Ausente"] = Field(
        ...,
        description="Antecedentes familiares de enfermedad cardíaca",
    )
    Tipo: int = Field(..., description="Comportamiento tipo-A")
    Obesidad: float = Field(..., description="Obesidad")
    Alcohol: float = Field(..., description="Consumo actual de alcohol")
    Edad: int = Field(..., description="Edad")


class PacienteOutput(BaseModel):
    chd_predicho: int
    probabilidad: float
    riesgo: str
    model_version: str


@app.get("/")
def estado():
    bundle = estado_servicio["bundle"]

    return {
        "servicio": "API de predicción de enfermedad Cardíaca",
        "modelo_cargado": bundle is not None,
        "model_version": (
            bundle["metadata"]["model_version"] if bundle is not None else None
        ),
    }


@app.post("/predecir", response_model=PacienteOutput)
def predecir(paciente: PacienteInput):
    bundle = estado_servicio["bundle"]

    if bundle is None:
        raise HTTPException(status_code=503, detail="El modelo aún no está cargado")

    fila = paciente.model_dump()
    fila["Familia"] = bundle["mapeo_familia"][fila["Familia"]]

    X_nuevo = pd.DataFrame([fila])[bundle["columnas"]]

    prediccion = int(bundle["pipeline"].predict(X_nuevo)[0])
    probabilidad = float(bundle["pipeline"].predict_proba(X_nuevo)[0, 1])

    return PacienteOutput(
        chd_predicho=prediccion,
        probabilidad=round(probabilidad, 4),
        riesgo="alto" if prediccion == 1 else "bajo",
        model_version=bundle["metadata"]["model_version"],
    )
