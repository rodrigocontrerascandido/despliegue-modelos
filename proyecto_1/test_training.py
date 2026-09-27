import pandas as pd
import pytest

from train import (
    MAPEO_FAMILIA,
    RUTA_DATOS,
    cargar_datos,
    preparar_datos,
    validar_datos,
)


def test_dataset_puede_cargarse():
    df = cargar_datos(RUTA_DATOS)
    assert not df.empty


def test_validacion_permite_mas_filas():
    df = cargar_datos(RUTA_DATOS)
    df_expandido = pd.concat([df, df.iloc[:20]], ignore_index=True)

    validar_datos(df_expandido)
    assert len(df_expandido) > len(df)


def test_validacion_detecta_columna_faltante():
    df = cargar_datos(RUTA_DATOS).drop(columns=["Edad"])

    with pytest.raises(ValueError, match="Faltan columnas requeridas"):
        validar_datos(df)


def test_validacion_rechaza_categoria_familia_invalida():
    df = cargar_datos(RUTA_DATOS).copy()
    df.loc[0, "Familia"] = "Desconocido"

    with pytest.raises(ValueError, match="Categorias no reconocidas"):
        validar_datos(df)


def test_preparacion_codifica_familia():
    df = cargar_datos(RUTA_DATOS)
    validar_datos(df)

    X, _ = preparar_datos(df)

    assert set(X["Familia"].unique()).issubset({0, 1})
    assert MAPEO_FAMILIA == {"Presente": 1, "Ausente": 0}


def test_variable_objetivo_no_esta_en_features():
    df = cargar_datos(RUTA_DATOS)
    validar_datos(df)

    X, y = preparar_datos(df)

    assert "chd" not in X.columns
    assert y.name == "chd"
