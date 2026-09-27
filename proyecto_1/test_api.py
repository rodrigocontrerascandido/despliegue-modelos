from fastapi.testclient import TestClient

from main import app


PACIENTE_VALIDO = {
    "sbp": 160,
    "Tabaco": 12,
    "ldl": 5.73,
    "Adiposidad": 23.11,
    "Familia": "Presente",
    "Tipo": 49,
    "Obesidad": 25.3,
    "Alcohol": 97.2,
    "Edad": 52,
}


def test_health_modelo_cargado():
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["modelo_cargado"] is True
    assert body["model_version"] == "1.0.0"


def test_prediccion_responde_contrato():
    with TestClient(app) as client:
        response = client.post("/predecir", json=PACIENTE_VALIDO)

    assert response.status_code == 200
    body = response.json()
    assert body["chd_predicho"] in [0, 1]
    assert 0.0 <= body["probabilidad"] <= 1.0
    assert body["riesgo"] in ["alto", "bajo"]
    assert body["model_version"] == "1.0.0"


def test_api_rechaza_paciente_incompleto():
    paciente_incompleto = PACIENTE_VALIDO.copy()
    paciente_incompleto.pop("sbp")

    with TestClient(app) as client:
        response = client.post("/predecir", json=paciente_incompleto)

    assert response.status_code == 422


def test_api_rechaza_categoria_familia_invalida():
    paciente_invalido = PACIENTE_VALIDO.copy()
    paciente_invalido["Familia"] = "Sí"

    with TestClient(app) as client:
        response = client.post("/predecir", json=paciente_invalido)

    assert response.status_code == 422
