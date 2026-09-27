import requests

url = 'http://127.0.0.1:8000'

r = requests.get(f"{url}/")
print(r.json())

cliente = {
  "antiguedad_meses": 16,
  "gasto_mensual": 17.93,
  "visitas_ultimo_mes": 9,
  "dias_desde_ultima_visita": 20,
  "tickets_soporte": 0,
  "plan": "Estándar",
  "metodo_pago": "Tarjeta de crédito",
  "descuento_activo": 0
}

r = requests.post(f"{url}/predecir", json=cliente)
print(r.json())