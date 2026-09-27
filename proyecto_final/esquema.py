
from typing import Literal
from pydantic import BaseModel, Field

class Cliente(BaseModel):
    Tiempo_en_el_sitio: float = Field(ge=1, description="Tiempo navegando en el sitio en minutos desde 1 minuto, no puede ser negativo")

    Edad: int = Field(ge=15, description="Edad del cliente desde 15 años, no puede ser negativo")

    Ingresos_de_la_zona: float = Field(ge=0, description="Ingresos de la zona del cliente desde 0, no puede ser negativo")

    Uso_de_Internet: float = Field(ge=1, description="Número de minutos que navega en internet desde 1 minuto, no puede ser negativo")

    Sexo: Literal['Femenino', 'Masculino'] = Field(..., description="Sexo del cliente")
