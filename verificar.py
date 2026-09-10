"""Comprueba ambos algoritmos con los datos entregados.

Ejecutar desde la raíz del repositorio: python verificar.py
Las respuestas de referencia se usan únicamente en estas comprobaciones.
"""

from pathlib import Path
import cv2
import numpy as np
from problema_1 import ecualizacion_local
from problema_2 import leer_gris, corregir_examen


def main():
    ejemplo = np.array([[0, 0, 10], [10, 10, 20], [20, 30, 0]], dtype=np.uint8)
    copia = ejemplo.copy()
    prueba = ecualizacion_local(ejemplo, (3, 3))
    assert prueba[1, 1] == 170
    assert np.all(ejemplo == copia)
    assert ecualizacion_local(ejemplo, (2, 4)).shape == ejemplo.shape
    assert np.all(ecualizacion_local(np.ones((3, 3), dtype=np.uint8), (3, 3)) == 255)
    print('Comprobaciones de ecualización: OK')

    archivos = sorted(Path('Consignas').glob('examen_*.png'))
    assert len(archivos) == 5
    examenes = [leer_gris(ruta) for ruta in archivos]
    resultados = [corregir_examen(examen) for examen in examenes]

    referencia = [
        [[], [], [], [], [], [], [], [], [], []],
        [['B'], ['B', 'C'], ['B'], ['D'], [], ['B'], ['A'], [], [], ['D']],
        [[letra] for letra in 'CBADBBABDD'],
        [[letra] for letra in 'BCDBAACDBA'],
        [[letra] for letra in 'CBADBBABDD'],
    ]
    encabezados_esperados = [
        [False, True, True], [False, True, True], [True, True, True],
        [True, False, True], [True, True, True],
    ]
    for i, (examen, resultado) in enumerate(zip(examenes, resultados)):
        assert resultado['respuestas'] == referencia[i]
        assert [resultado['campos'][k]['ok'] for k in ['Name', 'Date', 'Class']] == encabezados_esperados[i]
        desplazado = cv2.copyMakeBorder(examen, 17, 13, 23, 11, cv2.BORDER_CONSTANT, value=255)
        nuevo = corregir_examen(desplazado)
        assert nuevo['respuestas'] == resultado['respuestas']
        assert all(nuevo['campos'][k]['ok'] == resultado['campos'][k]['ok'] for k in resultado['campos'])
    assert [r['nota'] for r in resultados] == [0, 4, 10, 0, 10]
    print('50 preguntas, 15 validaciones de campos y 5 formularios desplazados: OK')


if __name__ == '__main__':
    main()
