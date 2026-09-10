# TP1 - Procesamiento de Imágenes

Universidad Austral - Licenciatura en Ciencia de Datos - 2026.

Resolución en **scripts Python (.py)** e **[informe PDF](output/pdf/Informe_TP1.pdf)**.
El informe describe ambos problemas, explica los algoritmos y sus parámetros, muestra
capturas de los pasos intermedios y presenta los resultados y las conclusiones.

## Archivos de la entrega

```text
problema_1.py                 Ecualización local y comparación de ventanas
problema_2.py                 Corrección de los cinco exámenes
verificar.py                  Comprobaciones de ambos algoritmos
requirements.txt             Bibliotecas necesarias
README.md                    Instrucciones de ejecución
Consignas/                   Enunciado original e imágenes de entrada
resultados/                  Figuras y composición de nombres
output/pdf/Informe_TP1.pdf    Informe de la resolución
```

## Preparación del entorno

Se utilizó **Python 3.14.7**. Las versiones exactas de NumPy, OpenCV y Matplotlib
están en `requirements.txt`. No se necesita Jupyter para ejecutar los scripts.

Clonar el repositorio y entrar en la rama de entrega:

```text
git clone --branch lsanfilippo https://github.com/BMilesRouillon/tp_procesamiento_de_imagenes.git
cd tp_procesamiento_de_imagenes
```

**Todos los comandos siguientes se ejecutan desde la raíz del repositorio**, donde
están los scripts y la carpeta `Consignas`. Si el repositorio ya está descargado,
abrir una terminal en esa carpeta.

### Windows - PowerShell

El entorno se guarda en una ruta corta del disco local para evitar errores de
instalación por rutas largas en carpetas sincronizadas:

```powershell
$entornoPdi = Join-Path $env:USERPROFILE '.virtualenvs\tp-pdi-lsanfilippo'
python -m venv $entornoPdi
& "$entornoPdi\Scripts\python.exe" -m pip install -r requirements.txt
& "$entornoPdi\Scripts\python.exe" -m pip check
```

En esta computadora ese entorno ya está instalado. En una terminal nueva basta con
volver a definir `$entornoPdi` antes de ejecutar los scripts. No es necesario activar
el entorno: los comandos usan directamente su intérprete.

### Linux o macOS

Con Python 3.14 instalado:

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

## Ejecutar cada script

### 1. Ecualización local

Windows:

```powershell
& "$entornoPdi\Scripts\python.exe" problema_1.py
```

Linux o macOS:

```bash
.venv/bin/python problema_1.py
```

Lee `Consignas/Imagen_con_detalles_escondidos.tif`, calcula una referencia global y
seis ecualizaciones locales: 3 x 3, 15 x 15, 31 x 31, 71 x 71, 151 x 151 y 15 x 31.
Imprime las propiedades de la imagen, comprueba la función con una matriz pequeña
y genera:

- `resultados/histograma_original.png`.
- `resultados/comparacion_ecualizacion.png`.
- `resultados/ecualizacion_local_15x15.png`.

La función `ecualizacion_local(imagen, ventana)` recibe una imagen en escala de
grises de tipo `uint8` y una tupla `(M, N)` de enteros positivos. Devuelve una imagen
del mismo tamaño. Para comparar otros tamaños se modifica la lista `ventanas` dentro
de `main()`.

### 2. Corrección automática de exámenes

Windows:

```powershell
& "$entornoPdi\Scripts\python.exe" problema_2.py
```

Linux o macOS:

```bash
.venv/bin/python problema_2.py
```

Procesa los cinco archivos `Consignas/examen_*.png`. Muestra por consola el estado
`OK` o `MAL` de cada pregunta y de Name, Date y Class, además de la nota y la condición
de aprobado o desaprobado.

Genera `resultados/aprobados.png` con los recortes de los nombres, las notas y el
estado de cada examen. También guarda las figuras intermedias de umbralado,
detección de celdas, extracción y lectura de respuestas, reconocimiento de letras
y validación de encabezados.

La función `corregir_examen(imagen)` recibe **únicamente la imagen en escala de grises**.
Detecta las posiciones de las tablas y los campos a partir de la tinta; no recibe
coordenadas de preguntas ni utiliza el nombre del archivo para decidir respuestas.
La clave es `C, B, A, D, B, B, A, B, D, D`. Se aprueba con al menos seis aciertos.
Las respuestas vacías o múltiples son incorrectas. La validación del encabezado
se informa por separado y no modifica la nota.

### 3. Comprobaciones

Windows:

```powershell
& "$entornoPdi\Scripts\python.exe" verificar.py
```

Linux o macOS:

```bash
.venv/bin/python verificar.py
```

Comprueba la ecualización con un resultado conocido, la conservación de la entrada,
ventanas rectangulares y el caso constante. Contrasta las 50 respuestas y los
15 estados de campos con una lectura visual de referencia. También verifica que
los cinco formularios conserven los resultados al agregar márgenes blancos y
desplazarlos. Las referencias están exclusivamente en este script de comprobación.

La ejecución correcta termina con:

```text
Comprobaciones de ecualización: OK
50 preguntas, 15 validaciones de campos y 5 formularios desplazados: OK
```

Los tres scripts finalizan sin requerir interacción. Las figuras se guardan en
`resultados/` para abrirlas después; al volver a ejecutar los scripts se reemplazan
sus salidas. Los archivos de entrada se conservan.

## Resultados

La ecualización local revela un cuadrado, una diagonal ascendente, una letra **a**,
cuatro líneas horizontales y un círculo. La ventana **15 x 15** permite distinguir
los cinco detalles con claridad.

| Examen | Nota | Name | Date | Class | Condición |
| --- | ---: | --- | --- | --- | --- |
| 1 | 0/10 | MAL | OK | OK | Desaprobado |
| 2 | 4/10 | MAL | OK | OK | Desaprobado |
| 3 | 10/10 | OK | OK | OK | Aprobado |
| 4 | 0/10 | OK | MAL | OK | Desaprobado |
| 5 | 10/10 | OK | OK | OK | Aprobado |

En Name se cuenta un espacio entre palabras para evaluar el máximo de 25 caracteres.
Date se valida por cantidad de caracteres y palabras, sin interpretar una fecha de
calendario. El método está diseñado para la plantilla derecha y la tipografía de
las imágenes entregadas. Los parámetros y las limitaciones se explican en el informe.

## Enlace de entrega

Usar el enlace de la rama que contiene esta resolución:

**https://github.com/BMilesRouillon/tp_procesamiento_de_imagenes/tree/lsanfilippo**
