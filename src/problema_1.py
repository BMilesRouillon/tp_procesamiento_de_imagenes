import cv2
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 1. Definición de la función de ecualización local
# ==========================================

def ecualizacion_local_histograma(img, ksize=(15, 15)):
    """
    Realiza la ecualización local del histograma píxel a píxel.

    Parámetros
    ----------
    img : numpy.ndarray
        Imagen en escala de grises (uint8).
    ksize : tuple
        Tamaño de la ventana de procesamiento (M, N).

    Retorna
    -------
    numpy.ndarray
        Imagen resultante de la ecualización local.
    """

    M, N = ksize

    pad_y = M // 2
    pad_x = N // 2

    # Agregar borde replicando los valores de los píxeles del borde
    img_padded = cv2.copyMakeBorder(
        img,
        top=pad_y,
        bottom=pad_y,
        left=pad_x,
        right=pad_x,
        borderType=cv2.BORDER_REPLICATE
    )

    alto, ancho = img.shape
    img_salida = np.zeros((alto, ancho), dtype=np.uint8)

    total_pixeles = M * N

    # Recorrido píxel a píxel
    for i in range(alto):
        for j in range(ancho):

            # Extraer el vecindario local M x N
            vecindario = img_padded[i:i + M, j:j + N]

            # Calcular histograma del vecindario
            histograma = cv2.calcHist(
                [vecindario],
                [0],
                None,
                [256],
                [0, 256]
            ).flatten()

            # Calcular frecuencia acumulada (CDF)
            cdf = (histograma / total_pixeles).cumsum()

            # Obtener el valor del píxel central
            val_central = img[i, j]

            # Aplicar transformación de ecualización
            nuevo_val = np.round(255 * cdf[val_central])

            img_salida[i, j] = nuevo_val

    return img_salida


# ==========================================
# 2. Programa principal
# ==========================================

def main():

    ruta_imagen = 'images/raw/Imagen_con_detalles_escondidos.tif'
    ruta_salida = 'images/output/problema_1.png'

    # --------------------------------------
    # Cargar imagen
    # --------------------------------------

    img = cv2.imread(
        ruta_imagen,
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        raise FileNotFoundError(
            f'No se pudo cargar la imagen: {ruta_imagen}'
        )

    # --------------------------------------
    # Ecualización global
    # --------------------------------------

    img_global = cv2.equalizeHist(img)

    # --------------------------------------
    # Ecualización local
    # --------------------------------------

    img_loc_5 = ecualizacion_local_histograma(
        img,
        (5, 5)
    )

    img_loc_9 = ecualizacion_local_histograma(
        img,
        (9, 9)
    )

    img_loc_15 = ecualizacion_local_histograma(
        img,
        (15, 15)
    )

    img_loc_31 = ecualizacion_local_histograma(
        img,
        (31, 31)
    )

    # --------------------------------------
    # Visualización y comparación
    # --------------------------------------

    fig, axs = plt.subplots(
        2,
        3,
        figsize=(15, 10)
    )

    axs[0, 0].imshow(
        img,
        cmap='gray',
        vmin=0,
        vmax=255
    )
    axs[0, 0].set_title('Imagen Original')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(
        img_global,
        cmap='gray',
        vmin=0,
        vmax=255
    )
    axs[0, 1].set_title('Ecualización Global')
    axs[0, 1].axis('off')

    axs[0, 2].imshow(
        img_loc_5,
        cmap='gray',
        vmin=0,
        vmax=255
    )
    axs[0, 2].set_title('Ecualización Local (5x5)')
    axs[0, 2].axis('off')

    axs[1, 0].imshow(
        img_loc_9,
        cmap='gray',
        vmin=0,
        vmax=255
    )
    axs[1, 0].set_title('Ecualización Local (9x9)')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(
        img_loc_15,
        cmap='gray',
        vmin=0,
        vmax=255
    )
    axs[1, 1].set_title('Ecualización Local (15x15)')
    axs[1, 1].axis('off')

    axs[1, 2].imshow(
        img_loc_31,
        cmap='gray',
        vmin=0,
        vmax=255
    )
    axs[1, 2].set_title('Ecualización Local (31x31)')
    axs[1, 2].axis('off')

    plt.tight_layout()

    # --------------------------------------
    # Guardar resultado
    # --------------------------------------

    plt.savefig(
        ruta_salida,
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()


# ==========================================
# Punto de entrada
# ==========================================

if __name__ == '__main__':
    main()