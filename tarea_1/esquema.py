
from typing import Literal
from pydantic import BaseModel, Field

class Cliente(BaseModel):
    antiguedad_meses: int = Field(ge=1, description="Antiguedad en uso del servicio en meses desde 1 mes, no puede ser negativo")

    gasto_mensual: float = Field(ge=5, description="Gastos mensuales del cliente desde 5, no puede ser negativo")

    visitas_ultimo_mes: int = Field(ge=0, description="Número de visitas del cliente en el último mes desde 0 vez, no puede ser negativo")

    dias_desde_ultima_visita: int = Field(ge=0, description="Número de días desde la última visita del cliente desde 0 días, no puede ser negativo")

    tickets_soporte: int = Field(ge=0, description="Número de veces que el cliente ha contactado al soporte técnico desde 0 vez, no puede ser negativo")

    plan: Literal['Básico', 'Estándar', 'Premium'] = Field(..., description="Tipo de plan del cliente")

    metodo_pago: Literal['Tarjeta de crédito', 'Efectivo', 'Transferencia bancaria'] = Field(..., description="Método de pago del cliente")

    descuento_activo: int = Field(..., description="Si el cliente tiene un descuento activo (0 = No, 1 = Sí)")

