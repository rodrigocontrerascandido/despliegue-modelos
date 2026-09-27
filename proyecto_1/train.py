"""Entrenamiento reproducible + tracking básico con MLflow.

Flujo de la clase:
heart.csv -> validación -> preparación -> train/test -> Pipeline -> evaluación
-> bundle joblib + metrics.json -> MLflow
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Configuración
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RUTA_DATOS = BASE_DIR / "heart.csv"
RUTA_BUNDLE = BASE_DIR / "modelo_bundle_e_cardiaca.pkl"
RUTA_METRICAS = BASE_DIR / "metrics.json"

VARIABLE_OBJETIVO = "chd"
MAPEO_FAMILIA = {"Presente": 1, "Ausente": 0}
VERSION_MODELO = "1.0.0"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Parámetro que cambiaremos en clase para generar distintos experimentos.
MODEL_C = 0.25

COLUMNAS_ESPERADAS = [
    "sbp",
    "Tabaco",
    "ldl",
    "Adiposidad",
    "Familia",
    "Tipo",
    "Obesidad",
    "Alcohol",
    "Edad",
    "chd",
]


# -----------------------------------------------------------------------------
# 1. Fuente de datos
# -----------------------------------------------------------------------------
def cargar_datos(ruta: Path = RUTA_DATOS) -> pd.DataFrame:
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el dataset: {ruta}")

    df = pd.read_csv(ruta)
    print(f"[1/8] Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


# -----------------------------------------------------------------------------
# 2. Validación
# -----------------------------------------------------------------------------
def validar_datos(df: pd.DataFrame) -> None:
    faltantes = [col for col in COLUMNAS_ESPERADAS if col not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")

    if df[COLUMNAS_ESPERADAS].isnull().any().any():
        columnas_con_nulos = df[COLUMNAS_ESPERADAS].columns[
            df[COLUMNAS_ESPERADAS].isnull().any()
        ].tolist()
        raise ValueError(f"Hay valores nulos en: {columnas_con_nulos}")

    categorias_familia = set(df["Familia"].unique())
    categorias_validas = set(MAPEO_FAMILIA)
    categorias_desconocidas = categorias_familia - categorias_validas
    if categorias_desconocidas:
        raise ValueError(
            f"Categorias no reconocidas en Familia: {sorted(categorias_desconocidas)}"
        )

    valores_objetivo = set(df[VARIABLE_OBJETIVO].unique())
    if not valores_objetivo.issubset({0, 1}):
        raise ValueError(
            f"La variable {VARIABLE_OBJETIVO} debe contener solo 0 y 1. "
            f"Se encontraron: {sorted(valores_objetivo)}"
        )

    print("[2/8] Validación de datos: OK")


# -----------------------------------------------------------------------------
# 3. Preparación
# -----------------------------------------------------------------------------
def preparar_datos(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df_preparado = df.copy()
    df_preparado["Familia"] = df_preparado["Familia"].map(MAPEO_FAMILIA)

    if df_preparado["Familia"].isnull().any():
        raise ValueError("Revisar categorías no mapeadas en Familia")

    X = df_preparado.drop(columns=VARIABLE_OBJETIVO)
    y = df_preparado[VARIABLE_OBJETIVO]

    print("[3/8] Datos preparados")
    return X, y


def dividir_datos(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"[4/8] Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# -----------------------------------------------------------------------------
# 4. Modelo
# -----------------------------------------------------------------------------
def construir_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "modelo",
                LogisticRegression(
                    random_state=RANDOM_STATE,
                    max_iter=1000,
                    C=MODEL_C,
                ),
            ),
        ]
    )


def entrenar_modelo(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    pipeline.fit(X_train, y_train)
    print("[5/8] Modelo entrenado")
    return pipeline


# -----------------------------------------------------------------------------
# 5. Evaluación
# -----------------------------------------------------------------------------
def evaluar_modelo(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metricas = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision_chd": round(
            float(precision_score(y_test, y_pred, zero_division=0)), 4
        ),
        "recall_chd": round(
            float(recall_score(y_test, y_pred, zero_division=0)), 4
        ),
        "f1_chd": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "n_test": int(len(y_test)),
    }

    print("[6/8] Evaluación")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Sin CHD", "Con CHD"],
            zero_division=0,
        )
    )
    print(f"AUC: {metricas['roc_auc']}")
    return metricas


# -----------------------------------------------------------------------------
# 6. Trazabilidad y artefactos locales
# -----------------------------------------------------------------------------
def calcular_hash_archivo(ruta: Path) -> str:
    sha256 = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            sha256.update(bloque)
    return sha256.hexdigest()


def crear_bundle(
    pipeline: Pipeline,
    columnas: list[str],
    metricas: dict,
    ruta_datos: Path = RUTA_DATOS,
) -> dict:
    return {
        "pipeline": pipeline,
        "columnas": columnas,
        "mapeo_familia": MAPEO_FAMILIA,
        "metadata": {
            "model_version": VERSION_MODELO,
            "modelo": "LogisticRegression",
            "sklearn_version": sklearn.__version__,
            "fecha_entrenamiento_utc": datetime.now(timezone.utc).isoformat(),
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "C": MODEL_C,
            "dataset": ruta_datos.name,
            "dataset_sha256": calcular_hash_archivo(ruta_datos),
            "metricas": metricas,
        },
    }


def guardar_artefactos(
    bundle: dict,
    metricas: dict,
    ruta_bundle: Path = RUTA_BUNDLE,
    ruta_metricas: Path = RUTA_METRICAS,
) -> None:
    joblib.dump(bundle, ruta_bundle)

    reporte = {
        "model_version": VERSION_MODELO,
        "fecha_entrenamiento_utc": bundle["metadata"]["fecha_entrenamiento_utc"],
        "dataset": bundle["metadata"]["dataset"],
        "dataset_sha256": bundle["metadata"]["dataset_sha256"],
        "sklearn_version": bundle["metadata"]["sklearn_version"],
        "parametros": {
            "C": MODEL_C,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
        },
        "metricas": metricas,
    }

    with ruta_metricas.open("w", encoding="utf-8") as archivo:
        json.dump(reporte, archivo, indent=4, ensure_ascii=False)

    print("[7/8] Artefactos locales guardados")
    print(f"      Bundle:   {ruta_bundle.name}")
    print(f"      Métricas: {ruta_metricas.name}")


# -----------------------------------------------------------------------------
# 7. MLflow básico
# -----------------------------------------------------------------------------
def registrar_en_mlflow(
    pipeline: Pipeline,
    metricas: dict,
    bundle: dict,
) -> None:
    # Se importa aquí para que las funciones de datos puedan probarse incluso
    # antes de instalar MLflow.
    import mlflow
    import mlflow.sklearn

    # SQLite local: no necesitamos PostgreSQL ni un servidor externo.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("enfermedad-cardiaca")

    with mlflow.start_run(run_name=f"logreg-C-{MODEL_C}"):
        mlflow.log_param("modelo", "LogisticRegression")
        mlflow.log_param("model_version", VERSION_MODELO)
        mlflow.log_param("C", MODEL_C)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("sklearn_version", sklearn.__version__)

        for nombre, valor in metricas.items():
            mlflow.log_metric(nombre, valor)

        mlflow.set_tag("dataset", bundle["metadata"]["dataset"])
        mlflow.set_tag("dataset_sha256", bundle["metadata"]["dataset_sha256"])

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="modelo",
        )

        mlflow.log_artifact(str(RUTA_METRICAS), artifact_path="reportes")

    print("[8/8] Experimento registrado en MLflow")


# -----------------------------------------------------------------------------
# Pipeline completo
# -----------------------------------------------------------------------------
def main() -> None:
    df = cargar_datos()
    validar_datos(df)

    X, y = preparar_datos(df)
    X_train, X_test, y_train, y_test = dividir_datos(X, y)

    pipeline = construir_pipeline()
    pipeline = entrenar_modelo(pipeline, X_train, y_train)

    metricas = evaluar_modelo(pipeline, X_test, y_test)

    bundle = crear_bundle(
        pipeline=pipeline,
        columnas=list(X.columns),
        metricas=metricas,
    )

    guardar_artefactos(bundle, metricas)
    registrar_en_mlflow(pipeline, metricas, bundle)

    print("\nEntrenamiento completado correctamente.")
    print("Para abrir MLflow:")
    print("mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000")
    print("Luego visita: http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
