"""Problema 2: corrección automática de exámenes.

Ejecutar desde la raíz del repositorio: python problema_2.py
Las figuras se guardan en resultados/.
"""

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

ENTRADAS = Path('Consignas')
SALIDAS = Path('resultados')

CLAVE = ['C', 'B', 'A', 'D', 'B', 'B', 'A', 'B', 'D', 'D']


def leer_gris(ruta):
    imagen = cv2.imdecode(np.fromfile(ruta, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if imagen is None:
        raise ValueError(f'No se pudo leer {ruta}')
    return imagen


def segmentos(vector):
    # Agregar ceros permite detectar también los pulsos que tocan un extremo.
    extendido = np.zeros(len(vector) + 2, dtype=int)
    extendido[1:-1] = vector
    cambios = np.diff(extendido)
    inicios = np.where(cambios == 1)[0]
    finales = np.where(cambios == -1)[0]
    return list(zip(inicios, finales))


def detectar_celdas(tinta):
    verticales = segmentos(tinta.sum(axis=0) > 0.5 * tinta.shape[0])
    if len(verticales) != 4:
        raise ValueError('Se esperaban los cuatro bordes verticales de las dos tablas.')
    cajas = []
    comienzo_tablas = tinta.shape[0]
    for columna in range(2):
        izquierda, derecha = verticales[2 * columna:2 * columna + 2]
        tramos = segmentos(tinta[:, izquierda[0]:izquierda[1]].any(axis=1))
        y0, y1 = max(tramos, key=lambda par: par[1] - par[0])
        comienzo_tablas = min(comienzo_tablas, y0)
        x0, x1 = izquierda[1], derecha[0]
        filas = tinta[y0:y1, x0:x1].sum(axis=1)
        horizontales = segmentos(filas > 0.8 * (x1 - x0))
        if len(horizontales) != 6:
            raise ValueError('Se esperaban seis líneas horizontales por tabla.')
        for fila in range(5):
            arriba = y0 + horizontales[fila][1] + 1
            abajo = y0 + horizontales[fila + 1][0] - 1
            cajas.append((x0 + 1, arriba, x1 - 1, abajo))
    return cajas, comienzo_tablas


def recortar_respuesta(celda):
    # Dentro de la celda, el tramo horizontal continuo más largo es el subrayado.
    mejor_largo = 0
    for fila in range(celda.shape[0]):
        for inicio, final in segmentos(celda[fila]):
            if final - inicio > mejor_largo:
                mejor_largo = final - inicio
                y, x0, x1 = fila, inicio, final
    if mejor_largo < 0.1 * celda.shape[1]:
        raise ValueError('No se encontró el subrayado de la respuesta.')
    alto = round(0.14 * celda.shape[0])
    recorte = celda[max(0, y - alto):y, x0:x1].astype(np.uint8)
    _, etiquetas, estadisticas, _ = cv2.connectedComponentsWithStats(
        recorte, connectivity=8, ltype=cv2.CV_32S)
    limpia = np.zeros_like(recorte)
    # Los restos de la línea de texto superior son de poca altura.
    for etiqueta, (x, y, ancho, alto, area) in enumerate(estadisticas[1:], start=1):
        if alto >= 0.055 * celda.shape[0] and area >= 5:
            limpia[etiquetas == etiqueta] = 1
    return limpia


def reconocer_letra(letra):
    contornos, jerarquia = cv2.findContours(letra, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    huecos = sum(jerarquia[0, :, 3] != -1)
    if huecos == 0:
        return 'C'
    if huecos == 2:
        return 'B'
    # A y D tienen un hueco: D posee un trazo vertical izquierdo; A, dos oblicuos.
    izquierda = letra[:, 0].sum() / letra.shape[0]
    return 'D' if izquierda > 0.7 else 'A'


def leer_respuesta(limpia):
    _, _, estadisticas, _ = cv2.connectedComponentsWithStats(
        limpia, connectivity=8, ltype=cv2.CV_32S)
    letras = []
    for x, y, ancho, alto, area in sorted(estadisticas[1:], key=lambda st: st[0]):
        letras.append(reconocer_letra(limpia[y:y + alto, x:x + ancho]))
    return letras


def analizar_campo(recorte):
    _, _, estadisticas, _ = cv2.connectedComponentsWithStats(
        recorte.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S)
    if len(estadisticas) == 1:
        return {'caracteres': 0, 'palabras': 0}
    # Agrupar partes con columnas superpuestas: por ejemplo, punto y cuerpo de i.
    # Si dos cajas solo se tocan en x, siguen siendo caracteres distintos.
    bandas = []
    for x, y, ancho, alto, area in sorted(estadisticas[1:], key=lambda st: st[0]):
        if bandas and x < bandas[-1][1]:
            bandas[-1][1] = max(bandas[-1][1], x + ancho)
        else:
            bandas.append([x, x + ancho])
    filas = np.where(recorte.any(axis=1))[0]
    altura = filas[-1] - filas[0] + 1
    espacios = sum(bandas[i + 1][0] - bandas[i][1] >= altura / 2
                   for i in range(len(bandas) - 1))
    return {'caracteres': len(bandas), 'palabras': 1 + espacios}


def validar_encabezado(tinta, imagen, limite_superior):
    encabezado = tinta[:limite_superior]
    y = int(encabezado.sum(axis=1).argmax())
    lineas = [(inicio, final) for inicio, final in segmentos(encabezado[y])
              if final - inicio > 0.05 * tinta.shape[1]]
    if len(lineas) != 3:
        raise ValueError('Se esperaban tres campos en el encabezado.')
    campos = {}
    for nombre, (x0, x1) in zip(['Name', 'Date', 'Class'], lineas):
        info = analizar_campo(encabezado[:y, x0:x1])
        n, palabras = info['caracteres'], info['palabras']
        if nombre == 'Name':
            info['ok'] = palabras >= 2 and n + palabras - 1 <= 25
        elif nombre == 'Date':
            info['ok'] = n == 8 and palabras == 1
        else:
            info['ok'] = n == 1
        info['recorte'] = imagen[:y, x0:x1].copy()
        campos[nombre] = info
    return campos


def corregir_examen(imagen):
    tinta = imagen < 150
    cajas, comienzo_tablas = detectar_celdas(tinta)
    respuestas, recortes = [], []
    for x0, y0, x1, y1 in cajas:
        limpia = recortar_respuesta(tinta[y0:y1, x0:x1])
        recortes.append(limpia)
        respuestas.append(leer_respuesta(limpia))
    correctas = [respuesta == [esperada] for respuesta, esperada in zip(respuestas, CLAVE)]
    campos = validar_encabezado(tinta, imagen, comienzo_tablas)
    return {'respuestas': respuestas, 'correctas': correctas, 'nota': sum(correctas),
            'campos': campos, 'cajas': cajas, 'recortes': recortes}


def main():
    entradas = ENTRADAS
    salidas = SALIDAS
    salidas.mkdir(exist_ok=True)
    plt.rcParams['figure.dpi'] = 110
    plt.rcParams['font.size'] = 12

    archivos = sorted(entradas.glob('examen_*.png'))
    if not archivos:
        raise ValueError('No se encontraron los exámenes en Consignas/.')
    examenes = [leer_gris(ruta) for ruta in archivos]

    muestra = examenes[1]
    tinta = muestra < 150
    cajas, comienzo = detectar_celdas(tinta)
    anotada = cv2.cvtColor(muestra, cv2.COLOR_GRAY2RGB)
    for x0, y0, x1, y1 in cajas:
        cv2.rectangle(anotada, (x0, y0), (x1, y1), (255, 0, 0), 1)

    fig, ejes = plt.subplots(1, 3, figsize=(10, 4.8))
    ejes[0].imshow(anotada)
    ejes[0].set_title('Celdas detectadas')
    ejes[0].axis('off')
    ejes[1].plot(tinta.sum(axis=0))
    ejes[1].axhline(0.5 * tinta.shape[0], color='red', linestyle='--')
    ejes[1].set_title('Proyección por columnas')
    ejes[1].set_xlabel('Columna')
    ejes[1].set_ylabel('Píxeles de tinta')
    x0, _, x1, _ = cajas[0]
    ejes[2].plot(tinta[comienzo:, x0:x1].sum(axis=1))
    ejes[2].axhline(0.8 * (x1 - x0), color='red', linestyle='--')
    ejes[2].set_title('Proyección por filas\nTabla izquierda')
    ejes[2].set_xlabel('Fila desde el comienzo de tabla')
    plt.tight_layout()
    fig.savefig(salidas / 'deteccion_celdas.png', dpi=140)
    plt.close(fig)

    # Máscara de tinta utilizada para detectar la estructura del formulario.
    fig, ejes = plt.subplots(1, 2, figsize=(8, 6))
    for eje, img, titulo in zip(ejes, [muestra, 1 - tinta.astype(np.uint8)],
                               ['Examen original', 'Máscara de tinta (umbral 150)']):
        eje.imshow(img, cmap='gray', interpolation='nearest')
        eje.set_title(titulo)
        eje.axis('off')
    plt.tight_layout()
    fig.savefig(salidas / 'umbralado_examen.png', dpi=160)
    plt.close(fig)

    # La segunda pregunta del examen 2 contiene dos opciones, B y C.
    x0, y0, x1, y1 = cajas[1]
    celda = tinta[y0:y1, x0:x1]
    limpia = recortar_respuesta(celda)
    fig, ejes = plt.subplots(1, 3, figsize=(9, 3))
    for eje, img, titulo in zip(ejes, [muestra[y0:y1, x0:x1], 1 - celda.astype(np.uint8), 1 - limpia],
                               ['Celda original', 'Celda umbralada', 'Respuesta aislada: B y C']):
        eje.imshow(img, cmap='gray', interpolation='nearest')
        eje.set_title(titulo, fontsize=10)
        eje.axis('off')
    plt.tight_layout()
    fig.savefig(salidas / 'extraccion_respuesta.png', dpi=180)
    plt.close(fig)


    resultados = []
    for ruta, examen in zip(archivos, examenes):
        resultado = corregir_examen(examen)
        resultados.append(resultado)
        print(f'\n{ruta.name}')
        for numero, (respuesta, ok) in enumerate(zip(resultado['respuestas'], resultado['correctas']), start=1):
            leida = ', '.join(respuesta) if respuesta else 'sin respuesta'
            print(f"Pregunta {numero:2}: {'OK' if ok else 'MAL'} ({leida})")
        for nombre, campo in resultado['campos'].items():
            print(f"{nombre}: {'OK' if campo['ok'] else 'MAL'} "
                  f"({campo['caracteres']} caracteres sin espacios; {campo['palabras']} palabras)")
        estado = 'APROBADO' if resultado['nota'] >= 6 else 'DESAPROBADO'
        print(f"Nota: {resultado['nota']}/10 — {estado}")

    fig, ejes = plt.subplots(5, 10, figsize=(15, 6))
    for i, resultado in enumerate(resultados):
        for j, (recorte, respuesta) in enumerate(zip(resultado['recortes'], resultado['respuestas'])):
            eje = ejes[i, j]
            eje.imshow(1 - recorte, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            leida = ''.join(respuesta) if respuesta else 'vacía'
            eje.set_title(f'E{i + 1} · P{j + 1}: {leida}', fontsize=9)
            eje.axis('off')
    plt.tight_layout()
    fig.savefig(salidas / 'respuestas_detectadas.png', dpi=150)
    plt.close(fig)

    fig, ejes = plt.subplots(5, 3, figsize=(8, 6))
    for i, resultado in enumerate(resultados):
        for j, nombre in enumerate(['Name', 'Date', 'Class']):
            campo = resultado['campos'][nombre]
            ejes[i, j].imshow(campo['recorte'], cmap='gray', vmin=0, vmax=255)
            ejes[i, j].set_title(f"E{i + 1} · {nombre}: {'OK' if campo['ok'] else 'MAL'}",
                                color='green' if campo['ok'] else 'red', fontsize=10)
            ejes[i, j].axis('off')
    plt.tight_layout()
    fig.savefig(salidas / 'validacion_encabezados.png', dpi=140)
    plt.close(fig)

    # Vista ampliada de las diez respuestas del examen 2.
    fig, ejes = plt.subplots(2, 5, figsize=(8, 3))
    for j, eje in enumerate(ejes.flat):
        respuesta = resultados[1]['respuestas'][j]
        eje.imshow(1 - resultados[1]['recortes'][j], cmap='gray', vmin=0, vmax=1,
                   interpolation='nearest')
        leida = ''.join(respuesta) if respuesta else 'vacía'
        eje.set_title(f'P{j + 1}: {leida}', fontsize=11)
        eje.axis('off')
    plt.tight_layout()
    fig.savefig(salidas / 'respuestas_examen_2.png', dpi=180)
    plt.close(fig)

    # Un ejemplo de cada letra para explicar la clasificación geométrica.
    fig, ejes = plt.subplots(1, 4, figsize=(6, 2))
    for eje, indice, titulo in zip(ejes, [2, 1, 0, 3],
                                 ['A: 1 hueco', 'B: 2 huecos', 'C: 0 huecos', 'D: 1 hueco']):
        limpia = resultados[2]['recortes'][indice]
        filas = np.where(limpia.any(axis=1))[0]
        columnas = np.where(limpia.any(axis=0))[0]
        letra = limpia[filas[0]:filas[-1] + 1, columnas[0]:columnas[-1] + 1]
        eje.imshow(1 - letra, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        eje.set_title(titulo, fontsize=11)
        eje.axis('off')
    plt.tight_layout()
    fig.savefig(salidas / 'letras_reconocidas.png', dpi=180)
    plt.close(fig)


    panel = np.full((90 * len(resultados), 900, 3), 255, dtype=np.uint8)
    for i, (ruta, resultado) in enumerate(zip(archivos, resultados)):
        aprobado = resultado['nota'] >= 6
        color = (0, 120, 0) if aprobado else (190, 0, 0)  # RGB para mostrar con Matplotlib.
        estado = 'APROBADO' if aprobado else 'DESAPROBADO'
        nombre = resultado['campos']['Name']['recorte']
        nombre = cv2.resize(nombre, (2 * nombre.shape[1], 2 * nombre.shape[0]))
        nombre = cv2.cvtColor(nombre, cv2.COLOR_GRAY2RGB)
        y = 90 * i
        alto, ancho = nombre.shape[:2]
        panel[y + 10:y + 10 + alto, 12:12 + ancho] = nombre
        cv2.rectangle(panel, (3, y + 3), (896, y + 86), color, 2)
        cv2.putText(panel, f"{ruta.stem}: {resultado['nota']}/10 - {estado}",
                    (420, y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    guardada = cv2.imwrite(str(salidas / 'aprobados.png'), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    assert guardada
    plt.figure(figsize=(12, 6))
    plt.imshow(panel)
    plt.axis('off')
    plt.title('Resultado de la corrección')
    plt.tight_layout()
    plt.close()
    print('Figuras y composición de nombres guardadas en resultados/.')


if __name__ == '__main__':
    main()
