import mlflow
from mlflow import MlflowClient


TRACKING_URI = "sqlite:///mlflow.db"

EXPERIMENTO = "enfermedad-cardiaca"

MODELO_REGISTRADO = "modelo-enfermedad-cardiaca"


# --------------------------------------------------
# Configurar MLflow
# --------------------------------------------------

mlflow.set_tracking_uri(TRACKING_URI)

client = MlflowClient(
    tracking_uri=TRACKING_URI
)


# --------------------------------------------------
# Buscar experimento
# --------------------------------------------------

experimento = mlflow.get_experiment_by_name(
    EXPERIMENTO
)

if experimento is None:
    raise RuntimeError(
        f"No existe el experimento: {EXPERIMENTO}"
    )


# --------------------------------------------------
# Buscar el mejor Run
# --------------------------------------------------

runs = mlflow.search_runs(
    experiment_ids=[
        experimento.experiment_id
    ],
    order_by=[
        "metrics.roc_auc DESC"
    ],
    max_results=1
)


if runs.empty:
    raise RuntimeError(
        "No existen entrenamientos registrados"
    )


mejor_run = runs.iloc[0]

run_id = mejor_run["run_id"]

roc_auc = mejor_run["metrics.roc_auc"]


print(
    f"Mejor Run encontrado: {run_id}"
)

print(
    f"ROC AUC: {roc_auc}"
)


# --------------------------------------------------
# Registrar el modelo
# --------------------------------------------------

model_uri = (
    f"runs:/{run_id}/modelo"
)

version_modelo = mlflow.register_model(
    model_uri=model_uri,
    name=MODELO_REGISTRADO
)


print(
    f"Modelo registrado como versión "
    f"{version_modelo.version}"
)


# --------------------------------------------------
# Asignar alias champion
# --------------------------------------------------

client.set_registered_model_alias(
    name=MODELO_REGISTRADO,
    alias="champion",
    version=version_modelo.version
)


print(
    f"Versión {version_modelo.version} "
    f"asignada como CHAMPION"
)