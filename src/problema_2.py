"""
Problema 2 - Corrección de multiple choice
===========================================

Corrección automática de exámenes multiple choice de 10 preguntas
(opciones A, B, C, D) tomando como única entrada la imagen de cada examen.

Respuestas correctas: 1.C 2.B 3.A 4.D 5.B 6.B 7.A 8.B 9.D 10.D

Estructura del script:
    Punto a - Corrección de las respuestas (OK/MAL)
    Punto b - Validación de los datos del encabezado
    Punto c - Evaluación de los exámenes resueltos
    Punto d - Imagen de salida con aprobados y desaprobados
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


RESPUESTAS_CORRECTAS = [
    'C', 'B', 'A', 'D', 'B',
    'B', 'A', 'B', 'D', 'D'
]

UMBRAL_APROBADO = 6


# ---------------------------------------------------------------------------
# Punto a - Corrección de las respuestas (OK/MAL)
# ---------------------------------------------------------------------------
#
# Se detectan las celdas de cada pregunta a partir de las líneas verticales y
# horizontales (sumando los píxeles por fila y por columna), se aísla la
# respuesta marcada en cada una y se identifica la opción elegida. Luego se
# compara con las respuestas correctas mostrando OK o MAL por pregunta.


def agrupar_indices(indices):
    """
    Agrupa índices consecutivos en intervalos.

    Recibe un array de índices (por ejemplo, filas o columnas que contienen
    una línea negra) y devuelve una lista de tuplas (inicio, fin) donde cada
    tupla representa una corrida de índices consecutivos. Se usa para pasar
    de "estas filas tienen tinta" a "hay una línea entre la fila X y la Y".

    Ejemplo: [3, 4, 5, 10, 11] -> [(3, 5), (10, 11)]
    """

    grupos = []

    if len(indices) == 0:
        return grupos

    inicio = indices[0]
    anterior = indices[0]

    for indice in indices[1:]:
        if indice == anterior + 1:
            anterior = indice
        else:
            grupos.append((inicio, anterior))
            inicio = indice
            anterior = indice

    grupos.append((inicio, anterior))

    return grupos


def obtener_cuadrantes(img, grupos_x, grupos_y):
    """
    Recorta las 10 celdas de preguntas de la grilla del examen.

    El examen está organizado en una grilla de 5 filas x 2 columnas de
    preguntas. `grupos_x` y `grupos_y` son los intervalos (inicio, fin) de
    las líneas verticales y horizontales que arman esa grilla (obtenidos con
    `agrupar_indices`). Cada cuadrante se recorta hacia adentro del borde de
    sus líneas, con un margen proporcional a la resolución de la imagen
    (0.3% de su ancho/alto), para no incluir la línea misma en el recorte
    sin importar el tamaño en píxeles de la imagen de entrada.

    El orden de recorrido es fila por fila y, dentro de cada fila, primero
    la columna izquierda (columna == 0) y después la derecha, por lo que la
    lista resultante NO respeta todavía la numeración real de las preguntas
    (eso se reordena después, en `procesar_examen`).
    """

    cuadrantes = []

    # Márgenes proporcionales a la resolución
    margen_x = int(img.shape[1] * 0.003)
    margen_y = int(img.shape[0] * 0.003)

    for fila in range(5):
        for columna in range(2):

            if columna == 0:
                x_inicio = grupos_x[0][1] + margen_x
                x_fin    = grupos_x[1][0] - margen_x
            else:
                x_inicio = grupos_x[2][1] + margen_x
                x_fin    = grupos_x[3][0] - margen_x

            y_inicio = grupos_y[fila][1]     + margen_y
            y_fin    = grupos_y[fila + 1][0] - margen_y

            cuadrante = img[
                y_inicio:y_fin,
                x_inicio:x_fin
            ]

            cuadrantes.append(cuadrante)

    return cuadrantes


def recortar_respuesta(respuesta):
    """
    Recorta la región en la que se encuentra la respuesta.

    Binariza la imagen recibida y busca la primera y última fila/columna
    que contienen al menos un píxel negro, para quedarse solo con el
    "bounding box" de la marca/letra, sin margen de fondo blanco alrededor.
    Esto estandariza la posición de la letra dentro del recorte,
    independientemente de en qué parte de la celda haya sido dibujada.
    """

    _, binaria = cv2.threshold(
        respuesta,
        200,
        255,
        cv2.THRESH_BINARY
    )

    # Filas que contienen al menos un píxel negro
    filas_con_negro = np.any(
        binaria == 0,
        axis=1
    )

    # Columnas que contienen al menos un píxel negro
    columnas_con_negro = np.any(
        binaria == 0,
        axis=0
    )

    # Límites verticales
    indices_filas = np.where(filas_con_negro)[0]

    # Límites horizontales
    indices_columnas = np.where(columnas_con_negro)[0]

    if (
        len(indices_filas) == 0
        or len(indices_columnas) == 0
    ):
        return None

    y_inicio = indices_filas[0]
    y_fin = indices_filas[-1]

    x_inicio = indices_columnas[0]
    x_fin = indices_columnas[-1]

    # Recorte final
    respuesta_recortada = respuesta[
        y_inicio:y_fin + 1,
        x_inicio:x_fin + 1
    ]

    return respuesta_recortada


def detectar_linea_respuesta(cuadrante):
    """
    Encuentra la línea horizontal en la que el alumno escribe su respuesta.

    Invierte y binariza la celda (tinta en blanco sobre fondo negro) y
    calcula sus componentes conectadas. De todas ellas, se queda con la
    que tiene mayor relación ancho/alto (`w / h`).
    """

    _, binaria = cv2.threshold(
        cuadrante,
        200,
        255,
        cv2.THRESH_BINARY_INV
    )

    n, _, stats, _ = cv2.connectedComponentsWithStats(
        binaria,
        8,
        cv2.CV_32S
    )

    candidatos = []

    for i in range(1, n):

        x, y, w, h, area = stats[i]

        if h == 0:
            continue

        relacion = w / h

        candidatos.append(
            (relacion, x, y, w, h, area)
        )

    if len(candidatos) == 0:
        return None

    # La línea buscada debería ser el objeto
    # con mayor relación ancho / alto
    relacion, x, y, w, h, area = max(
        candidatos,
        key=lambda candidato: candidato[0]
    )

    return x, y, w, h


def obtener_respuesta(cuadrante):
    """
    Aísla, dentro de una celda de pregunta, la letra marcada por el alumno.

    Cada celda tiene 4 opciones (A, B, C, D) alineadas sobre una línea
    horizontal.

    El proceso es:
        1. Ubicar la línea horizontal con `detectar_linea_respuesta`.
        2. Recortar la región por encima de la línea.
        3. Verificar si hay píxeles negros en el 30% inferior de esa región.
        4. Detectar los renglones que contienen tinta y quedarse con el último.
        5. Analizar las columnas del renglón seleccionado para detectar cuántas
           opciones fueron marcadas.
        6. Si hay más de una opción marcada, devolver None.
        7. Recortar la respuesta marcada.
    """

    linea = detectar_linea_respuesta(cuadrante)

    if linea is None:
        return None

    x, y, w, h = linea

    # Región de opciones sobre la línea
    region_opciones = cuadrante[0:y, x : x + w]

    if region_opciones.size == 0:
        return None

    # Binarización
    _, binaria = cv2.threshold(
        region_opciones,
        200,
        255,
        cv2.THRESH_BINARY
    )

    # ----------------------------------------------------
    # PASO 1: Evaluar el 30% inferior de filas
    # ----------------------------------------------------
    alto_30 = max(
        1,
        int(region_opciones.shape[0] * 0.30)
    )

    filas_30_inferior = binaria[-alto_30:, :]

    hay_tinta_abajo = np.any(filas_30_inferior == 0)

    # Si no hay ningún píxel negro, no hay respuesta
    if not hay_tinta_abajo:
        return None

    # ----------------------------------------------------
    # PASO 2: Detectar renglones con tinta
    # ----------------------------------------------------
    proyeccion_y = np.any(binaria == 0, axis=1)
    indices_negros = np.where(proyeccion_y)[0]

    grupos_renglones = agrupar_indices(indices_negros)

    if len(grupos_renglones) == 0:
        return None

    # Tomar el último renglón
    y_inicio, y_fin = grupos_renglones[-1]

    # Extraer el renglón de la respuesta
    respuesta = region_opciones[
        y_inicio : y_fin + 1,
        :
    ]

    # ----------------------------------------------------
    # PASO 3: Detectar cuántas opciones fueron marcadas
    # ----------------------------------------------------
    _, binaria_respuesta = cv2.threshold(
        respuesta,
        200,
        255,
        cv2.THRESH_BINARY
    )

    # Para cada columna:
    # True  -> hay al menos un píxel negro
    # False -> no hay ningún píxel negro
    proyeccion_x = np.any(
        binaria_respuesta == 0,
        axis=0
    )

    indices_negros_x = np.where(proyeccion_x)[0]

    # Agrupar columnas consecutivas con tinta.
    # Cada grupo representa una opción marcada.
    grupos_opciones = agrupar_indices(indices_negros_x)

    # Si hay más de una opción marcada, la respuesta es inválida
    if len(grupos_opciones) > 1:
        return None

    # ----------------------------------------------------
    # PASO 4: Recortar la respuesta
    # ----------------------------------------------------
    return recortar_respuesta(respuesta)


def procesar_examen(ruta_imagen):
    """
    Carga un examen y extrae sus 10 respuestas, en orden de pregunta.

    Pasos:
        1. Cargar la imagen en escala de grises y binarizarla.
        2. Sumar píxeles negros por fila y por columna (proyecciones) para
           encontrar dónde están las líneas de la grilla: una fila/columna
           que acumula muchos píxeles negros es parte de una línea. El
           umbral para considerarla "línea" es proporcional al tamaño de la
           imagen (40% del alto o del ancho), en vez de un número fijo de
           píxeles, para no depender de la resolución del escaneo.
        3. Agrupar esas filas/columnas en intervalos con `agrupar_indices`
           para obtener la posición de cada línea de la grilla.
        4. Recortar las 10 celdas de pregunta con `obtener_cuadrantes`.
        5. Extraer la respuesta marcada de cada celda con `obtener_respuesta`.
        6. Reordenar la lista de respuestas: `obtener_cuadrantes` las entrega
           en orden fila-por-fila (columna izquierda primero), pero la
           numeración real del examen intercala ambas columnas
           (1-5 en la izquierda, 6-10 en la derecha), así que se reindexan
           a la numeración 1 a 10.
    """

    img = cv2.imread(
        ruta_imagen,
        cv2.IMREAD_GRAYSCALE
    )

    # Binarización
    _, binary = cv2.threshold(
        img,
        127,
        255,
        cv2.THRESH_BINARY
    )

    # Proyecciones
    proyeccion_x = np.sum(
        binary == 0,
        axis=0
    )

    proyeccion_y = np.sum(
        binary == 0,
        axis=1
    )

    # Umbrales proporcionales: una línea completa de la grilla recorre casi
    # todo el alto (para una línea vertical) o el ancho (para una línea
    # horizontal) de la imagen, así que se pide que la cantidad de píxeles
    # negros supere el 40% de esa dimensión.
    umbral_x = int(img.shape[0] * 0.4)
    umbral_y = int(img.shape[1] * 0.4)

    # Detección de líneas
    lineas_x = np.where(
        proyeccion_x >= umbral_x
    )[0]

    lineas_y = np.where(
        proyeccion_y >= umbral_y
    )[0]

    grupos_x = agrupar_indices(
        lineas_x
    )

    grupos_y = agrupar_indices(
        lineas_y
    )

    # Extracción de cuadrantes (se descarta grupos_y[0], que corresponde
    # al subrayado del encabezado, no a la grilla de preguntas)
    cuadrantes = obtener_cuadrantes(
        img,
        grupos_x,
        grupos_y[1:]
    )

    # Detección de respuestas
    respuestas = []

    for cuadrante in cuadrantes:

        respuesta = obtener_respuesta(
            cuadrante
        )

        respuestas.append(
            respuesta
        )

    # Reordenar según la numeración real: la columna izquierda trae las
    # preguntas 1,3,5,7,9 (índices 0,2,4,6,8) y la columna derecha trae
    # 2,4,6,8,10 (índices 1,3,5,7,9), entrelazadas fila por fila.
    respuestas = [
        respuestas[0],  # Pregunta 1
        respuestas[2],  # Pregunta 2
        respuestas[4],  # Pregunta 3
        respuestas[6],  # Pregunta 4
        respuestas[8],  # Pregunta 5
        respuestas[1],  # Pregunta 6
        respuestas[3],  # Pregunta 7
        respuestas[5],  # Pregunta 8
        respuestas[7],  # Pregunta 9
        respuestas[9]   # Pregunta 10
    ]

    return img, cuadrantes, respuestas


def obtener_descriptor(respuesta, tam=(8, 10)):
    """
    Normaliza el tamaño de la letra a dimensiones fijas y calcula
    sus proyecciones por fila y por columna.

    Primero redimensiona el recorte a un tamaño fijo `tam` (ancho, alto),
    para que el descriptor resultante siempre tenga la misma longitud sin
    importar el tamaño original del recorte de cada respuesta. Luego
    binariza esa versión normalizada y cuenta, para cada fila y cada
    columna, cuántos píxeles negros tiene. El descriptor final es la
    concatenación de ambos perfiles (perfil por fila + perfil por columna),
    y funciona como una "firma" de la forma de la letra: letras distintas
    (A, B, C, D) producen vectores distintos, que después se comparan por
    distancia en `identificar_respuesta`.
    """
    if respuesta is None or respuesta.size == 0:
        return None

    # Redimensionar a tamaño fijo para que el descriptor siempre tenga la misma longitud
    # independientemente de las dimensiones originales del recorte
    respuesta_norm = cv2.resize(
        respuesta, tam, interpolation=cv2.INTER_NEAREST
    )

    _, binaria = cv2.threshold(respuesta_norm, 200, 255, cv2.THRESH_BINARY)

    # Conteo de píxeles negros por fila (eje Y) y por columna (eje X)
    perfil_y = np.sum(binaria == 0, axis=1)
    perfil_x = np.sum(binaria == 0, axis=0)

    # Vector concatenado de longitud fija
    return np.concatenate([perfil_y, perfil_x])


def crear_referencias(examen, indices_letras):
    """
    Construye el diccionario de descriptores de referencia (uno por letra).

    Toma un examen ya procesado (donde se sabe, de antemano, qué letra
    corresponde a cada pregunta) y arma, para cada letra A/B/C/D, su
    descriptor con `obtener_descriptor`. Estas referencias son las que
    después se usan en `identificar_respuesta` para reconocer, por
    comparación de distancia, qué letra marcó el alumno en otros exámenes.

    Además, guarda en disco el recorte original (sin normalizar) de cada
    letra de referencia, como `images/letters/letter_{letra}.png`, para
    poder inspeccionarlas visualmente.
    """

    referencias = {}

    for indice, letra in indices_letras.items():
        respuesta = examen['respuestas'][indice]
        referencias[letra] = obtener_descriptor(respuesta)

        ruta = f'images/letters/letter_{letra}.png'
        cv2.imwrite(ruta, respuesta)  # guarda el recorte original, sin normalizar

    return referencias


def identificar_respuesta(respuesta, referencias, umbral=40):
    """
    Clasifica una letra comparándola contra los descriptores de referencia.

    Calcula el descriptor de `respuesta` (con `obtener_descriptor`, usando
    el mismo `tam` por defecto que se usó al armar `referencias`) y lo
    compara, con distancia euclídea, contra cada descriptor de
    `referencias`. Se queda con la letra de referencia más cercana. Si esa
    distancia mínima supera `umbral`, o si no se pudo calcular el
    descriptor, se devuelve "No válido".
    """
    if respuesta is None:
        return 'No válido'

    descriptor = obtener_descriptor(respuesta)

    if descriptor is None:
        return 'No válido'

    distancias = {}

    for letra, referencia in referencias.items():
        if referencia is None:
            continue
        # Distancia euclídea entre los vectores (siempre tienen la misma longitud)
        distancias[letra] = np.linalg.norm(descriptor - referencia)

    if not distancias:
        return 'No válido'

    letra_mas_cercana = min(distancias, key=distancias.get)
    distancia_minima = distancias[letra_mas_cercana]

    # Si la diferencia mínima es demasiado grande, se descarta
    if distancia_minima > umbral:
        return 'No válido'

    return letra_mas_cercana


def corregir_examen(ruta_imagen, referencias):
    """
    Corrige un examen completo e imprime OK/MAL por pregunta.

    Procesa la imagen del examen, identifica la letra marcada en cada una
    de las 10 preguntas (comparando contra `referencias`, con el umbral
    por defecto de `identificar_respuesta`) y compara cada letra
    identificada contra `RESPUESTAS_CORRECTAS`, imprimiendo el resultado
    pregunta por pregunta.
    """

    # Procesar examen
    img, cuadrantes, respuestas = procesar_examen(ruta_imagen)

    # Identificar letras
    letras = []

    for respuesta in respuestas:

        letra = identificar_respuesta(
            respuesta,
            referencias
        )

        letras.append(letra)

    # Mostrar resultados
    for i, (respuesta, correcta) in enumerate(
        zip(letras, RESPUESTAS_CORRECTAS),
        start=1
    ):

        if respuesta == correcta:
            resultado = "OK"
        else:
            resultado = "MAL"

        print(
            f"Pregunta {i}: {resultado}"
        )

    return letras


# ---------------------------------------------------------------------------
# Punto b - Validación de los datos del encabezado
# ---------------------------------------------------------------------------
#
# Se reutiliza la técnica del punto a (proyecciones y componentes
# conectadas) para aislar el encabezado. Las tres líneas horizontales
# (Name, Date y Class) se detectan como las componentes conectadas con
# mayor relación ancho/alto (líneas finas y largas).
#
# Para validar cada campo NO se reconoce el texto, solo se cuentan
# cantidades: segmentando por columnas, cada corrida de columnas con tinta
# es un carácter, y una separación (gap) mayor a medio alto del texto
# indica un cambio de palabra.
#
# Restricciones:
#   - Name: al menos dos palabras y no más de 25 caracteres.
#   - Date: 8 caracteres formando una sola palabra.
#   - Class: un único carácter.


def detectar_lineas_encabezado(img, ancho_minimo=40, umbral_ratio=15):
    """
    Detecta las 3 líneas horizontales del encabezado (Name, Date, Class).

    Primero encuentra el subrayado general del encabezado (la primera línea
    horizontal larga de la imagen, vía proyección de píxeles negros) para
    recortar la franja donde está el encabezado. Dentro de esa franja,
    calcula las componentes conectadas del texto/líneas (imagen invertida:
    tinta en blanco sobre fondo negro) y se queda solo con las que son
    "finas y largas" (ancho > `ancho_minimo` y relación ancho/alto >
    `umbral_ratio`), que son justamente las líneas de subrayado de cada
    campo y no el texto escrito sobre ellas.
    """
    # El subrayado del encabezado es la primera línea horizontal larga
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    proyeccion_y = np.sum(binary == 0, axis=1)
    y_subrayado = np.where(proyeccion_y >= 200)[0][0]

    encabezado = img[0:y_subrayado + 3, :]

    # Componentes conectadas (texto y líneas en blanco sobre fondo negro)
    _, binaria_inv = cv2.threshold(
        encabezado, 127, 255, cv2.THRESH_BINARY_INV
    )
    n, _, stats, _ = cv2.connectedComponentsWithStats(
        binaria_inv, 8, cv2.CV_32S
    )

    lineas = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if w > ancho_minimo and w / max(h, 1) > umbral_ratio:
            lineas.append((x, y, w, h))

    lineas.sort(key=lambda linea: linea[0])

    return lineas


def segmentar_caracteres(region):
    """
    Devuelve los intervalos (x_inicio, x_fin) de cada carácter de una región.

    Usa la proyección por columnas: se considera que una columna "tiene
    tinta" si algún píxel de esa columna es oscuro (< 128), y cada corrida
    de columnas consecutivas con tinta se toma como un carácter. Es más
    robusto que usar componentes conectadas, ya que una letra partida en
    varios trazos (por ejemplo, una "i" con su punto separado, o una barra
    "/" fragmentada) sigue siendo una única corrida de columnas.
    """
    tinta = region < 128
    columnas_con_tinta = tinta.any(axis=0)

    caracteres = []
    inicio = None

    for x, tiene_tinta in enumerate(columnas_con_tinta):
        if tiene_tinta and inicio is None:
            inicio = x
        elif not tiene_tinta and inicio is not None:
            caracteres.append((inicio, x - 1))
            inicio = None

    if inicio is not None:
        caracteres.append((inicio, len(columnas_con_tinta) - 1))

    return caracteres


def analizar_campo(img, linea):
    """
    Cuenta caracteres y palabras del texto escrito sobre una línea del encabezado.

    Recorta la región de texto (por encima de la línea del campo y dentro
    de su ancho), la segmenta en caracteres con `segmentar_caracteres` y
    agrupa esos caracteres en palabras: un espacio (gap) entre dos
    caracteres consecutivos mayor a `umbral_gap` píxeles se interpreta como
    un cambio de palabra.
    """
    x, y, w, h = linea

    # Región del texto: por encima de la línea y dentro de su ancho
    region = img[0:y - 1, x:x + w]

    caracteres = segmentar_caracteres(region)
    cantidad_caracteres = len(caracteres)
    if cantidad_caracteres == 0:
        return 0, 0

    # Palabras: un gap mayor a umbral_gap (en píxeles) separa dos palabras
    umbral_gap = 5
    cantidad_palabras = 1

    for actual, siguiente in zip(caracteres, caracteres[1:]):
        gap = siguiente[0] - actual[1] - 1
        if gap > umbral_gap:
            cantidad_palabras += 1

    return cantidad_caracteres, cantidad_palabras


def validar_encabezado(ruta_imagen):
    """
    Valida los campos Name, Date y Class de un examen e imprime su estado.

    Detecta las 3 líneas del encabezado, cuenta caracteres/palabras de cada
    campo con `analizar_campo` y aplica las reglas de validación:
        - Name: al menos 2 palabras y entre 1 y 25 caracteres.
        - Date: exactamente 8 caracteres formando una única palabra
          (pensado para un formato tipo DD/MM/AA sin espacios).
        - Class: un único carácter.
    """
    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)

    lineas = detectar_lineas_encabezado(img)
    nombres = ['Name', 'Date', 'Class']

    conteos = {}
    for linea, nombre in zip(lineas, nombres):
        conteos[nombre] = analizar_campo(img, linea)

    # Name: al menos dos palabras y no más de 25 caracteres
    n_caracteres, n_palabras = conteos['Name']
    name_ok = (n_palabras >= 2) and (1 <= n_caracteres <= 25)

    # Date: 8 caracteres formando una sola palabra
    d_caracteres, d_palabras = conteos['Date']
    date_ok = (d_caracteres == 8) and (d_palabras == 1)

    # Class: un único carácter
    c_caracteres, _ = conteos['Class']
    class_ok = (c_caracteres == 1)

    print(f"Name: {'OK' if name_ok else 'MAL'}")
    print(f"Date: {'OK' if date_ok else 'MAL'}")
    print(f"Class: {'OK' if class_ok else 'MAL'}")

    return {'Name': name_ok, 'Date': date_ok, 'Class': class_ok}


# ---------------------------------------------------------------------------
# Punto d - Imagen de salida con aprobados y desaprobados
# ---------------------------------------------------------------------------
#
# Se genera una imagen que reúne los recortes (crop) del campo Name de
# todos los exámenes del punto c. Un alumno aprueba con al menos 6
# respuestas correctas. Los aprobados se marcan en verde y los
# desaprobados en rojo.


def recortar_campo_nombre(img):
    """
    Recorta la sub-imagen del campo Name del encabezado.

    Reutiliza la detección de líneas del encabezado (punto b): la primera
    línea (la de más a la izquierda) corresponde al campo Name. Se recorta
    la región de texto ubicada por encima de esa línea.
    """
    lineas = detectar_lineas_encabezado(img)
    x, y, w, h = lineas[0]
    return img[0:y - 1, x:x + w]


def contar_respuestas_correctas(letras):
    """
    Cuenta cuántas respuestas coinciden con las correctas.

    Compara, posición a posición, la lista de letras identificadas contra
    `RESPUESTAS_CORRECTAS` y cuenta las coincidencias.
    """
    return sum(
        1
        for letra, correcta in zip(letras, RESPUESTAS_CORRECTAS)
        if letra == correcta
    )


def generar_imagen_resultados(referencias, ruta_salida='images/output/problema_2.png'):
    """
    Genera y guarda una imagen resumen con el Name de cada examen y su resultado.

    Para cada examen del 1 al 5:
        1. Lo procesa y corrige (reutilizando `procesar_examen`,
           `identificar_respuesta` y `contar_respuestas_correctas`).
        2. Determina si aprobó (>= `UMBRAL_APROBADO` respuestas correctas).
        3. Recorta su campo Name con `recortar_campo_nombre`.
        4. Dibuja ese recorte en una figura de 5 filas (una por examen),
           con el título y el borde en verde si aprobó o rojo si no.
    """
    fig, axs = plt.subplots(5, 1, figsize=(7, 9))

    for idx, numero_examen in enumerate(range(1, 6)):

        ruta = f'images/raw/examen_{numero_examen}.png'
        img, cuadrantes, respuestas = procesar_examen(ruta)

        # Corrección (punto a) para contar respuestas correctas
        letras = [
            identificar_respuesta(respuesta, referencias)
            for respuesta in respuestas
        ]
        correctas = contar_respuestas_correctas(letras)
        aprobado = correctas >= UMBRAL_APROBADO

        # Crop del campo Name
        crop_nombre = recortar_campo_nombre(img)

        # Verde si aprobó, rojo si no
        color = 'green' if aprobado else 'red'
        estado = 'APROBADO' if aprobado else 'DESAPROBADO'

        ax = axs[idx]
        ax.imshow(crop_nombre, cmap='gray', vmin=0, vmax=255)
        ax.set_title(
            f'Examen {numero_examen}: {estado} ({correctas}/10)',
            color=color,
            fontsize=12,
            fontweight='bold'
        )
        ax.set_xticks([])
        ax.set_yticks([])

        # Borde de color alrededor del crop
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(3)

    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150, bbox_inches='tight')
    plt.show()

    print(f'Imagen de salida guardada en: {ruta_salida}')


# ---------------------------------------------------------------------------
# Ejecución principal
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    # --- Punto b: validar encabezados de los 5 exámenes ---
    for numero_examen in range(1, 6):
        print(f"Exámen {numero_examen}:")
        ruta = f'images/raw/examen_{numero_examen}.png'
        validar_encabezado(ruta)
        print()

    # --- Punto c: procesar los 5 exámenes y corregirlos ---
    examenes = {}

    for i in range(1, 6):
        ruta = f'images/raw/examen_{i}.png'
        img, cuadrantes, respuestas = procesar_examen(ruta)

        examenes[i] = {
            'imagen': img,
            'cuadrantes': cuadrantes,
            'respuestas': respuestas
        }

    # El examen 4 se usa como "hoja de referencia": se sabe de antemano
    # qué letra corresponde a cada índice de pregunta.
    indices_letras = {0: 'B', 5: 'A', 1: 'C', 2: 'D'}

    referencias = crear_referencias(examenes[4], indices_letras)

    for numero_examen in range(1, 6):
        print(f"Exámen {numero_examen}:")
        ruta = f'images/raw/examen_{numero_examen}.png'
        corregir_examen(ruta, referencias)
        print()

    # --- Punto d: imagen resumen de aprobados/desaprobados ---
    generar_imagen_resultados(referencias)