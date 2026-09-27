import os
import sys

# 1. Forzar a Python a mirar dentro de la carpeta 'src'
ruta_src = os.path.abspath('src')
if ruta_src not in sys.path:
    sys.path.insert(0, ruta_src)

# Si ya estabas parado dentro de src:
ruta_actual = os.path.abspath('.')
if ruta_actual not in sys.path:
    sys.path.insert(0, ruta_actual)


import cv2
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
# 2. Ahora el import encuentra el archivo sí o sí
from problema_2 import (
    RESPUESTAS_CORRECTAS,
    UMBRAL_APROBADO,
    agrupar_indices,
    contar_respuestas_correctas,
    crear_referencias,
    detectar_linea_respuesta,
    detectar_lineas_encabezado,
    identificar_respuesta,
    obtener_cuadrantes,
    obtener_descriptor,
    obtener_respuesta,
    procesar_examen,
    recortar_campo_nombre,
    validar_encabezado,
)

# Detecta automáticamente dónde está la carpeta images
if os.path.exists('images'):
    carpeta_base = '.'
elif os.path.exists('../images'):
    carpeta_base = '..'
else:
    carpeta_base = os.path.abspath('.')

RUTA_EXAMEN = os.path.join(carpeta_base, 'images', 'raw', 'examen_2.png')
RUTA_REFERENCIA = os.path.join(carpeta_base, 'images', 'raw', 'examen_4.png')
INDICES_LETRAS = {0: 'B', 5: 'A', 1: 'C', 2: 'D'}


# ==============================================================================
# Paso 1 - Cargar la imagen y binarizarla
# ==============================================================================
img = cv2.imread(RUTA_EXAMEN, cv2.IMREAD_GRAYSCALE)
_, binaria = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

fig, axs = plt.subplots(1, 2, figsize=(10, 7))
axs[0].imshow(img, cmap='gray', vmin=0, vmax=255)
axs[0].set_title('Imagen original')
axs[0].axis('off')
axs[1].imshow(binaria, cmap='gray', vmin=0, vmax=255)
axs[1].set_title('Imagen binarizada')
axs[1].axis('off')
plt.tight_layout()
plt.show(block=False)


# ==============================================================================
# Paso 2 - Detectar las líneas de la grilla
# ==============================================================================
proyeccion_x = np.sum(binaria == 0, axis=0)
proyeccion_y = np.sum(binaria == 0, axis=1)

umbral_x = int(img.shape[0] * 0.4)
umbral_y = int(img.shape[1] * 0.4)

lineas_x = np.where(proyeccion_x >= umbral_x)[0]
lineas_y = np.where(proyeccion_y >= umbral_y)[0]

grupos_x = agrupar_indices(lineas_x)
grupos_y = agrupar_indices(lineas_y)

print(f'Líneas verticales detectadas:   {len(grupos_x)}')
print(f'Líneas horizontales detectadas: {len(grupos_y)}')

fig, ax = plt.subplots(figsize=(6, 8))
ax.imshow(img, cmap='gray', vmin=0, vmax=255)
for x0, x1 in grupos_x:
    ax.axvline((x0 + x1) / 2, color='red', linewidth=1)
for y0, y1 in grupos_y:
    ax.axhline((y0 + y1) / 2, color='lime', linewidth=1)
ax.set_title(
    'Líneas de la grilla detectadas\n(rojo = verticales, verde = horizontales)'
)
ax.axis('off')
plt.show(block=False)


# ==============================================================================
# Paso 3 - Recortar las 10 celdas de pregunta
# ==============================================================================
cuadrantes = obtener_cuadrantes(img, grupos_x, grupos_y[1:])

fig, axs = plt.subplots(5, 2, figsize=(8, 11))
for i, cuadrante in enumerate(cuadrantes):
    ax = axs[i // 2, i % 2]
    ax.imshow(cuadrante, cmap='gray', vmin=0, vmax=255)
    ax.set_title(f'Celda {i}')
    ax.axis('off')
plt.tight_layout()
plt.show(block=False)


# ==============================================================================
# Paso 4 - Detectar, en cada celda, la línea de las opciones (A B C D)
# ==============================================================================
fig, axs = plt.subplots(5, 2, figsize=(8, 11))
for i, cuadrante in enumerate(cuadrantes):
    linea = detectar_linea_respuesta(cuadrante)
    ax = axs[i // 2, i % 2]
    ax.imshow(cuadrante, cmap='gray', vmin=0, vmax=255)
    if linea is not None:
        x, y, w, h = linea
        ax.add_patch(
            Rectangle(
                (x, y), w, h, edgecolor='red', facecolor='none', linewidth=2
            )
        )
    ax.set_title(f'Celda {i}')
    ax.axis('off')
plt.tight_layout()
plt.show(block=False)


# ==============================================================================
# Paso 5 - Extraer la letra marcada por el alumno en cada celda
# ==============================================================================
respuestas_sin_reordenar = [obtener_respuesta(c) for c in cuadrantes]

fig, axs = plt.subplots(5, 2, figsize=(8, 11))
for i, respuesta in enumerate(respuestas_sin_reordenar):
    ax = axs[i // 2, i % 2]
    if respuesta is not None:
        ax.imshow(respuesta, cmap='gray', vmin=0, vmax=255)
        ax.set_title(f'Celda {i+1}')
    else:
        ax.set_title(f'Celda {i+1} - sin respuesta')
    ax.axis('off')
plt.tight_layout()
plt.show(block=False)


# ==============================================================================
# Paso 6 - Procesar el examen completo
# ==============================================================================
img, cuadrantes, respuestas = procesar_examen(RUTA_EXAMEN)

fig, axs = plt.subplots(5, 2, figsize=(8, 11))
for i, respuesta in enumerate(respuestas):
    ax = axs[i // 2, i % 2]
    if respuesta is not None:
        ax.imshow(respuesta, cmap='gray', vmin=0, vmax=255)
        ax.set_title(f'Pregunta {i + 1}')
    else:
        ax.set_title(f'Pregunta {i + 1} - sin respuesta')
    ax.axis('off')
plt.tight_layout()
plt.show(block=False)


# ==============================================================================
# Paso 7 - Armar las referencias A / B / C / D
# ==============================================================================
_, _, respuestas_examen_referencia = procesar_examen(RUTA_REFERENCIA)
examen_referencia = {'respuestas': respuestas_examen_referencia}

referencias = crear_referencias(examen_referencia, INDICES_LETRAS)

fig, axs = plt.subplots(1, 4, figsize=(10, 3))
for ax, (indice, letra) in zip(axs, INDICES_LETRAS.items()):
    ax.imshow(
        respuestas_examen_referencia[indice], cmap='gray', vmin=0, vmax=255
    )
    ax.set_title(f'Referencia: {letra}')
    ax.axis('off')
plt.tight_layout()
plt.show(block=False)


# ==============================================================================
# Paso 8 - Ver el descriptor ("firma") de una letra
# ==============================================================================
for letra_ejemplo in ['A','B','C','D']:
    descriptor_ejemplo = referencias[letra_ejemplo]

    plt.figure(figsize=(7, 3))
    plt.bar(range(len(descriptor_ejemplo)), descriptor_ejemplo)
    plt.title(f"Descriptor de referencia para la letra '{letra_ejemplo}'")
    plt.xlabel('Índice del vector (perfil por fila, luego perfil por columna)')
    plt.ylabel('Cantidad de píxeles negros')
    plt.show(block=False)


# ==============================================================================
# Paso 9 - Identificar las respuestas del examen y corregirlo
# ==============================================================================
letras = [identificar_respuesta(r, referencias) for r in respuestas]

fig, axs = plt.subplots(5, 2, figsize=(8, 11))
for i, (respuesta, letra, correcta) in enumerate(
    zip(respuestas, letras, RESPUESTAS_CORRECTAS)
):
    ax = axs[i // 2, i % 2]
    if respuesta is not None:
        ax.imshow(respuesta, cmap='gray', vmin=0, vmax=255)
    resultado = 'OK' if letra == correcta else 'MAL'
    color = 'green' if resultado == 'OK' else 'red'
    ax.set_title(
        f'P{i + 1}: {letra} ({resultado})', color=color, fontweight='bold'
    )
    ax.axis('off')
plt.tight_layout()
plt.show(block=False)

correctas = contar_respuestas_correctas(letras)
print(f'Respuestas correctas: {correctas}/10')
print('APROBADO' if correctas >= UMBRAL_APROBADO else 'DESAPROBADO')


# ==============================================================================
# Paso 10 - Validar el encabezado (Name, Date, Class)
# ==============================================================================
lineas_encabezado = detectar_lineas_encabezado(img)
y_max_encabezado = max(y for (_, y, _, _) in lineas_encabezado) + 10

fig, ax = plt.subplots(figsize=(9, 2.5))
ax.imshow(img[0:y_max_encabezado, :], cmap='gray', vmin=0, vmax=255)
for (x, y, w, h), nombre in zip(lineas_encabezado, ['Name', 'Date', 'Class']):
    ax.add_patch(
        Rectangle((x, 0), w, y, edgecolor='blue', facecolor='none', linewidth=2)
    )
ax.set_title('Campos del encabezado detectados (Name, Date, Class)')
ax.axis('off')
plt.show(block=False)

resultado_encabezado = validar_encabezado(RUTA_EXAMEN)


# ==============================================================================
# Paso 11 - Resultado final: Name + estado (aprobado / desaprobado)
# ==============================================================================
crop_nombre = recortar_campo_nombre(img)
aprobado = correctas >= UMBRAL_APROBADO
color = 'green' if aprobado else 'red'
estado = 'APROBADO' if aprobado else 'DESAPROBADO'

fig, ax = plt.subplots(figsize=(6, 2.2))
ax.imshow(crop_nombre, cmap='gray', vmin=0, vmax=255)
ax.set_title(
    f'{estado} ({correctas}/10)', color=color, fontsize=13, fontweight='bold'
)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_color(color)
    spine.set_linewidth(3)
plt.show(block=False)