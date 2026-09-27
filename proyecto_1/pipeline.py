import subprocess
import sys


def ejecutar(
    archivo: str
) -> None:

    print(
        f"\nEjecutando {archivo}"
    )

    resultado = subprocess.run(
        [
            sys.executable,
            archivo
        ],
        check=True
    )

    if resultado.returncode != 0:
        raise RuntimeError(
            f"Error ejecutando {archivo}"
        )


def main() -> None:

    # 1. Entrenar
    ejecutar(
        "train.py"
    )

    # 2. Seleccionar mejor
    ejecutar(
        "seleccionar_mejor.py"
    )

    print(
        "\nPipeline MLOps completado."
    )


if __name__ == "__main__":
    main()