# Procesamiento de Imágenes — Trabajo Práctico N.º 1

Implementación en Python de técnicas de procesamiento de imágenes para la materia **Procesamiento de Imágenes — Licenciatura en Ciencia de Datos, Universidad Austral (2026)**.

El trabajo aborda dos problemas:

1. **Ecualización local de histograma**
2. **Corrección automática de exámenes multiple choice**

---

## Problema 1 — Ecualización local de histograma

Se implementa una función de ecualización local de histograma que procesa una imagen utilizando una ventana de tamaño configurable.

Para cada píxel, se considera su vecindario, se calcula el histograma local y se utiliza la transformación de ecualización correspondiente para obtener el nuevo nivel de intensidad del píxel central.

### Características

* Tamaño de ventana configurable.
* Tratamiento de bordes mediante extensión de la imagen.
* Análisis del efecto producido por diferentes tamaños de ventana.
* Comparación de los detalles visibles en distintas regiones de la imagen.

El objetivo es analizar cómo el tamaño del vecindario modifica la capacidad de resaltar detalles que presentan niveles de intensidad similares a los de su fondo local.

---

## Problema 2 — Corrección automática de multiple choice

Se desarrolla un algoritmo capaz de procesar automáticamente una imagen de un examen de **10 preguntas**, con cuatro opciones por pregunta:

* A
* B
* C
* D

Las respuestas correctas establecidas por el enunciado son:

| Pregunta  | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --------- | - | - | - | - | - | - | - | - | - | -- |
| Respuesta | C | B | A | D | B | B | A | B | D | D  |

El algoritmo recibe únicamente la imagen del examen y detecta automáticamente la estructura necesaria para procesarlo, sin utilizar como entrada las coordenadas de las preguntas.

### A. Corrección de respuestas

El procesamiento sigue, de forma general, las siguientes etapas:

1. Carga de la imagen en escala de grises.
2. Binarización de la imagen.
3. Detección de las líneas que delimitan la grilla del examen mediante proyecciones de píxeles.
4. Agrupación de líneas consecutivas para identificar las regiones correspondientes a las preguntas.
5. Extracción de los cuadrantes de cada pregunta.
6. Detección de la línea de respuesta dentro de cada cuadrante.
7. Identificación de la región donde se encuentra la respuesta marcada.
8. Detección de múltiples opciones marcadas mediante proyección horizontal.
9. Extracción de un descriptor de la respuesta.
10. Comparación con referencias conocidas para determinar si la opción corresponde a A, B, C o D.

### Identificación de las opciones

Para reconocer las respuestas se utilizan como referencia las cuatro alternativas obtenidas de un examen conocido.

Cada respuesta se normaliza a un tamaño fijo de **8 × 10 píxeles** y se construye un descriptor combinando:

* cantidad de píxeles negros por fila;
* cantidad de píxeles negros por columna.

Esto produce un descriptor de 18 valores.

La respuesta detectada se compara mediante **distancia euclídea** con los cuatro descriptores de referencia. Se selecciona la opción cuya distancia sea menor, siempre que la distancia no supere el umbral establecido. En caso contrario, la respuesta se considera **No válida**.

Si se detectan dos o más opciones marcadas, la respuesta también se considera inválida, de acuerdo con el enunciado.

Finalmente, cada respuesta identificada se compara con la clave de respuestas y se informa:

```text
Pregunta 1: OK
Pregunta 2: MAL
...
Pregunta 10: OK
```

---

### B. Validación del encabezado

El mismo examen se utiliza para validar automáticamente los tres campos del encabezado:

* **Name**
* **Date**
* **Class**

Primero se detectan las líneas correspondientes a los campos mediante componentes conexas y relaciones entre ancho y alto.

Luego, cada campo se segmenta utilizando una proyección vertical de los píxeles de tinta para detectar los caracteres.

Las condiciones establecidas por el enunciado son:

| Campo | Condición                                   |
| ----- | ------------------------------------------- |
| Name  | Al menos 2 palabras y máximo 25 caracteres  |
| Date  | Exactamente 8 caracteres y una sola palabra |
| Class | Exactamente 1 carácter                      |

La separación entre palabras se determina a partir de la distancia horizontal entre caracteres consecutivos.

El resultado se muestra como:

```text
Name: OK
Date: MAL
Class: OK
```

---

### C. Evaluación de los exámenes

El algoritmo se ejecuta sobre las imágenes `examen_x.png` proporcionadas para el trabajo práctico.

Para cada examen se realizan tanto:

* la corrección de las 10 preguntas;
* la validación de los campos del encabezado.

Además, el proyecto permite generar una imagen de resultados sobre el examen procesado, mostrando visualmente la información obtenida durante la corrección.

---

## Tecnologías utilizadas

* **Python**
* **OpenCV**
* **NumPy**
* **Matplotlib**

---

## Ejecución

Clonar el repositorio:

```bash
git clone https://github.com/BMilesRouillon/tp_procesamiento_de_imagenes.git
cd tp_procesamiento_de_imagenes
```

Instalar las dependencias necesarias:

```bash
pip install opencv-python numpy matplotlib
```

Luego ejecutar el script correspondiente:

```bash
python problema_1.py
```

o:

```bash
python problema_2.py
```

El segundo script procesa los exámenes, valida el encabezado y realiza la corrección automática de las respuestas.

---

## Resultados

El proyecto permite automatizar el procesamiento completo de los exámenes a partir de sus imágenes, incluyendo:

* detección automática de la estructura del examen;
* extracción de las preguntas;
* identificación de respuestas;
* detección de respuestas múltiples o ausentes;
* comparación contra la clave de respuestas;
* validación de Name, Date y Class;
* generación de resultados visuales.

El desarrollo se realizó procurando que el algoritmo obtenga la información necesaria directamente desde la imagen, sin utilizar las coordenadas de las preguntas como información proporcionada externamente.
