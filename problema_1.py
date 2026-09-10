"""Problema 1: ecualización local del histograma.

Ejecutar desde la raíz del repositorio: python problema_1.py
Las figuras se guardan en resultados/.
"""

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

ENTRADAS = Path('Consignas')
SALIDAS = Path('resultados')


def leer_gris(ruta):
    imagen = cv2.imdecode(np.fromfile(ruta, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if imagen is None:
        raise ValueError(f'No se pudo leer {ruta}')
    return imagen


def ecualizacion_local(imagen, ventana):
    m, n = ventana
    if m < 1 or n < 1:
        raise ValueError('Las dimensiones de la ventana deben ser positivas.')
    arriba, izquierda = m // 2, n // 2
    extendida = cv2.copyMakeBorder(imagen, arriba, m - arriba - 1,
                                 izquierda, n - izquierda - 1, cv2.BORDER_REPLICATE)
    salida = np.zeros_like(imagen)
    for fila in range(imagen.shape[0]):
        for columna in range(imagen.shape[1]):
            vecindario = extendida[fila:fila + m, columna:columna + n]
            histograma = cv2.calcHist([vecindario], [0], None, [256], [0, 256]).flatten()
            acumulada = (histograma / (m * n)).cumsum()
            salida[fila, columna] = np.round(255 * acumulada[imagen[fila, columna]])
    return salida

def main():
    entradas = ENTRADAS
    salidas = SALIDAS
    salidas.mkdir(exist_ok=True)
    plt.rcParams['figure.dpi'] = 110
    plt.rcParams['font.size'] = 12

    imagen = leer_gris(entradas / 'Imagen_con_detalles_escondidos.tif')
    print('Forma:', imagen.shape, '| Tipo:', imagen.dtype)
    print('Mínimo:', imagen.min(), '| Máximo:', imagen.max())
    print('Niveles presentes:', np.unique(imagen))

    histograma = cv2.calcHist([imagen], [0], None, [256], [0, 256]).flatten()
    cdf_global = (histograma / imagen.size).cumsum()
    global_ecualizada = np.round(255 * cdf_global[imagen]).astype(np.uint8)

    fig, ejes = plt.subplots(1, 3, figsize=(9, 3.5))
    ejes[0].imshow(imagen, cmap='gray', vmin=0, vmax=255)
    ejes[0].set_title('Original')
    ejes[0].axis('off')
    ejes[1].plot(histograma)
    ejes[1].set_title('Histograma global')
    ejes[1].set_xlabel('Intensidad')
    ejes[1].set_ylabel('Cantidad de píxeles')
    ejes[2].plot(cdf_global)
    ejes[2].set_title('Frecuencia acumulada')
    ejes[2].set_xlabel('Intensidad')
    ejes[2].set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(salidas / 'histograma_original.png', dpi=160)
    plt.close(fig)

    ventanas = [(3, 3), (15, 15), (31, 31), (71, 71), (151, 151), (15, 31)]
    locales = {}
    for ventana in ventanas:
        locales[ventana] = ecualizacion_local(imagen, ventana)

    imagenes_comparacion = [imagen, global_ecualizada] + [locales[v] for v in ventanas]
    titulos = ['Original', 'Ecualización global'] + [f'Local {m} × {n}' for m, n in ventanas]
    fig, ejes = plt.subplots(2, 4, figsize=(10, 5.5))
    for eje, resultado, titulo in zip(ejes.flat, imagenes_comparacion, titulos):
        eje.imshow(resultado, cmap='gray', vmin=0, vmax=255)
        eje.set_title(titulo)
        eje.axis('off')
    plt.tight_layout()
    fig.savefig(salidas / 'comparacion_ecualizacion.png', dpi=160)
    plt.close(fig)

    guardada = cv2.imwrite(str(salidas / 'ecualizacion_local_15x15.png'), locales[(15, 15)])
    assert guardada

    ejemplo = np.array([[0, 0, 10], [10, 10, 20], [20, 30, 0]], dtype=np.uint8)
    copia = ejemplo.copy()
    prueba = ecualizacion_local(ejemplo, (3, 3))
    assert prueba[1, 1] == 170
    assert np.all(ejemplo == copia)
    assert ecualizacion_local(ejemplo, (2, 4)).shape == ejemplo.shape
    assert np.all(ecualizacion_local(np.ones((3, 3), dtype=np.uint8), (3, 3)) == 255)
    print('Comprobaciones de ecualización: OK')
    print('Detalles: cuadrado, diagonal, letra a, cuatro líneas horizontales y círculo.')
    print('Figuras guardadas en resultados/.')


if __name__ == '__main__':
    main()
