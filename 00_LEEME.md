# KeyWord Spotting (KWS) para M5Stack LLM Kit (Axera AX630C) 
# Con leves cambios es funcional en AX650N

## 1) Datos iniciales:
- Desarrollado por:     ASRC.
- Fecha:                       23/09/2025.
- Versión:                     V1.0.0.
- Procesador destino:  Axera AX630C de bajo costo.
- Python Versión:         3.13.7
- SO:                            MS Windows 11 Pro   24H2.

### Datos Axera AX630C [https://en.axera-tech.com/Product/126.html]

## 2) Introducción:
Dado que no existen algoritmos basados en redes neuronales del tipo Keyword Spotting (KWS) que comprendan variantes del idioma español como el hablado en Argentina, se diseña una serie de scripts en Python con los cuales se realizarán las tareas de generación de dataset en base a un solo comando de voz grabado por clase y una serie limitada de audios de fondo con diferentes ruidos ambientales.

Para esto se utilizará una red neuronal muy simple que pueda leer tensores con dimensión **float32[1,1,98,64]** dentro de los cuales se encuentran espectrogramas en formato log-Mel, el significado de las dimensiones es:

```
[ batch , channel , time , frequency ]
[   1   ,    1    ,  98  ,     64    ]

Batch      = 1 → un ejemplo.
Channel    = 1 → “imagen” mono canal.
Time       = 98 → ~1 segundo en ventanas de 10 ms.
Frequency  = 64 → resolución espectral en escala Mel.
```

1. Primer 1 → batch size:
Es el tamaño del lote de entrada. En inferencia casi siempre se fija a 1 (procesás un solo ejemplo a la vez). Si entrenaras con batches más grandes, aquí aparecería el valor >1.
2. Segundo 1 → canal (input channel):
Es el número de canales de entrada. En visión sería algo como 3 (RGB). En tu caso, como los espectrogramas log-Mel se tratan como “imágenes de un canal” → 1. Mantener la dimensión permite que convoluciones 2D funcionen igual que con imágenes.
3. Tercer valor 98 → dimensión temporal (frames):
Son los frames en el tiempo de tu log-Mel. Cada frame corresponde a una ventana de ~25 ms desplazada cada ~10 ms (según tu win_ms y hop_ms). Así, 98 frames ≈ 98 × 10 ms ≈ ~1 s de audio procesado. Este es el eje que captura la dinámica temporal del audio y lo que consume la parte RNN (GRU) de tu modelo.
4. Cuarto valor 64 → dimensión frecuencial (bandas Mel):
Es el número de coeficientes log-Mel extraídos por frame. Cada valor representa la energía en una banda de frecuencia. 64 se eligió como parámetro n_mels al calcular el espectrograma. Es el eje que representa la información espectral del audio.

## 3) Diagrama de flujo de funcionamiento del software:
```
┌───────────┐      ┌──────────────────────────┐       ┌──────────────────────┐
│ Micrófono │ ───▶│ Front-end de Audio       │  ───▶ │ Extracción de        │
└───────────┘      │ (llm_audio: VAD/AGC/AEC) │       │ log-Mel (64 bandas)  │
                   └──────────────────────────┘       └──────────────────────┘
                                 │                               │
                                 │  ventanas 1.0 s (hop 100 ms)  ▼
                                 │                      ┌──────────────────────┐
                                 │                      │ Normalización        │
                                 │                      │ (mean/var por clip)  │
                                 │                      └──────────────────────┘
                                 │                               ▼
                                 │                      ┌──────────────────────┐
                                 │                      │ Bloque Convolucional │
                                 │                      │ (Conv2D + BN + ReLU  │
                                 │                      │  + MaxPool x N)      │
                                 │                      └──────────────────────┘
                                 │                               ▼
                                 │                      ┌──────────────────────┐
                                 │                      │ Reordenar a secuencia│
                                 │                      │ [T, C·F]             │
                                 │                      └──────────────────────┘
                                 │                               ▼
                                 │                      ┌──────────────────────┐
                                 │                      │ Bloque Recurrente    │
                                 │                      │ (GRU bidireccional)  │
                                 │                      └──────────────────────┘
                                 │                               ▼
                                 │                      ┌──────────────────────┐
                                 │                      │ Capa Lineal final    │
                                 │                      │ (dim=8)              │
                                 │                      └──────────────────────┘
                                 │                               ▼
                                 │                      ┌──────────────────────┐
                                 │                      │ Softmax (8 clases)   │
                                 │                      │ 7 keywords + FONDO   │
                                 │                      └──────────────────────┘
                                 │                               ▼
                                 │          ┌───────────────────────────────────────────┐
                                 └────────▶│ Post-proceso temporal                      │
                                            │ (media móvil, histéresis, refractario)    │
                                            └───────────────────────────────────────────┘
                                                         │
                                                         ▼
                                   ┌──────────────────────────────────────────────┐
                                   │ Decisión / Eventos:                          │
                                   │  - CONTINUAR                                 │
                                   │  - REPETIR                                   │
                                   │  - SALIR                                     │
                                   │  - PRINCIPAL                                 │
                                   │  - ACOPLADO                                  │
                                   │  - CANCELAR                                  │
                                   │  - LEVANTADO                                 │
                                   │  - (FONDO → ignorar)                         │
                                   └──────────────────────────────────────────────┘
```

## 4) Esquema de directorios y archivos del proyecto:
```
kws_project/
  ├── .venv (Ambiente virtual con módulos de Python instalados)
  ├── data (Carpeta que contiene: audios seed, audios de calibración, audios generados con aumentación automática y tensores.)
  ├── src (Código fuente conteniendo los scripts escritos en Python.)
  ├── README.md (Archivo de ayuda en formato MarkDown.)
  └── requirements.txt (Archivo con los módulos de Python requeridos para que funcionen los scripts. Se instala con "pip install -r requirements.txt")
```

## 5) Pasos a seguir y uso de los scripts:
### a) Grabar un archivo .WAV con cada una de la órdenes y ruidos de fondo:
Se puede utilizar cualquier grabador de voz de cualquier sistema operativo (ej. Audacity, Adobe Premiere Pro, Grabadora de Sonido de Windows, etc.).  
En cuanto a los ruidos de fondo, es necesario evaluar cuáles prevalecerán durante el funcionamiento del KWS. Es posible obtener sonidos de Internet y convertirlos a .WAV con Audacity.

Es recomendable generar un árbol de carpetas como este:
```
data
  ├── audios (carpeta conteniendo un audio .WAV por cada una de las 7 clases de los comandos y 1 clase con los fondos generados por grabación)
  |     ├── ACOPLAR
  |     |     └── 000_acoplar.WAV
  |     ├── CANCELAR
  |     |     └── 000_cancelar.WAV
  |     ├── CONTINUAR
  |     |     └── 000_continuar.WAV
  |     ├── FONDO
  |     |     └── 000_fondo.WAV (Se buscan diferentes sonidos de fondo con duració variable y se guardan en esta carpeta)
  |     ├── LEVANTADO
  |     |     └── 000_levantado.WAV
  |     ├── PRINCIPAL
  |     |     └── 000_principal.WAV
  |     ├── REPETIR
  |     |     └── 000_repetir.WAV
  |     └── SALIR
  |           └── 000_salir.WAV
  └── seeds
```

### b) Generar el dataset para entrenamiento de la red neuronal:
Debido a que es necesario contar con entre 1000 y 3000 muestras de audio de cada comando y a que es complejo poder grabar 3000 veces cada comando con diferentes voces y ruidos de fondo, utilizaremos un algoritmo para poder ampliar nuestra única muestra guardada dentro de un archivo .WAV. 

Para esto el algoritmo planteado generará diferentes cambio: frecuencia, tiempos, corriemientos en tiempo, entre otros y sumará los diferentes ruidos de fondo; al final obtendremos 3000 muestras de cada comando con diferentes características, las 3000 muestras de cada comando serirán para poder entrenar nuestra red neuronal y de esta manera hacer que sea utilizable en un entorno real.

Es recomendable generar una carpeta llamada **seeds** en donde se encontrarán los mismos audios que los contenidos en la carpeta **audios**. Esto con la idea de que la carpeta **audios** sea un backup y se utilice la carpeta **seeds** para generar el dataset.

```
data
  ├── audios  
  └── seeds (carpeta conteniendo un audio .WAV por cada una de las 7 clases de los comandos y 1 clase con los fondos generados por grabación)
        ├── ACOPLAR
        |    └── 000_acoplar.WAV
        ├── CANCELAR
        |    └── 000_cancelar.WAV
        ├── CONTINUAR
        |    └── 000_continuar.WAV
        ├── FONDO
        |    └── 000_fondo.WAV (Se buscan diferentes sonidos de fondo con duració variable y se guardan en esta carpeta)
        ├── LEVANTADO
        |    └── 000_levantado.WAV
        ├── PRINCIPAL
        |    └── 000_principal.WAV
        ├── REPETIR
        |    └── 000_repetir.WAV
        └── SALIR
            └── 000_salir.WAV
```

Variantes del comando **make_dataset.py**:
```
python.exe .\make_dataset.py --help
usage: make_dataset.py [-h] [--seeds-root SEEDS_ROOT] [--noises-root NOISES_ROOT] [--rirs-root RIRS_ROOT] [--out-root OUT_ROOT] [--per-class PER_CLASS] [--fondo-minutes FONDO_MINUTES] [--p-shift P_SHIFT] [--p-stretch P_STRETCH]
                       [--p-pitch P_PITCH] [--max-shift-ms MAX_SHIFT_MS] [--stretch-low STRETCH_LOW] [--stretch-high STRETCH_HIGH] [--pitch-semitones PITCH_SEMITONES] [--p-noise P_NOISE] [--p-reverb P_REVERB] [--snr-list SNR_LIST]
                       [--use-fondo-as-noise]

Generador de dataset KWS (ES) con aumentaciones

options:
  -h, --help            show this help message and exit
  --seeds-root SEEDS_ROOT
  --noises-root NOISES_ROOT
  --rirs-root RIRS_ROOT
  --out-root OUT_ROOT
  --per-class PER_CLASS
  --fondo-minutes FONDO_MINUTES
  --p-shift P_SHIFT
  --p-stretch P_STRETCH
  --p-pitch P_PITCH
  --max-shift-ms MAX_SHIFT_MS
  --stretch-low STRETCH_LOW
  --stretch-high STRETCH_HIGH
  --pitch-semitones PITCH_SEMITONES
  --p-noise P_NOISE
  --p-reverb P_REVERB
  --snr-list SNR_LIST   Lista de SNRs en dB separadas por coma (ej: '20,10,5,0')
  --use-fondo-as-noise  Usar seeds/FONDO como ruidos adicionales
```

Ejemplos de uso:
Con parámetros personalizados:
```
python make_dataset.py --seeds-root D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\src\seeds --out-root D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\generado_automaticamente --per-class 3000 --fondo-minutes 20 --p-noise 0.9 --p-reverb 0.2 --p-stretch 0.4 --stretch-low 0.95 --stretch-high 1.05 --p-pitch 0.5 --pitch-semitones 1.0 --use-fondo-as-noise
```

Con parámetros por defecto (SE UTILIZA ESTA FOMA DE EJECUCIÓN):
```
python make_dataset.py --seeds-root D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\seeds --noises-root D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\seeds\FONDO --out-root D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\generados_automaticamente
```

Esquema del director ./data al terminar de ejecutarse el script:
```
data
  ├── audios
  ├── generados_automaticamente (carpeta conteniendo las ampliaciones de muestra con el dataset)
  |     ├── ACOPLAR
  |     |     ├── acoplar_00000.WAV
  |     |     ├── ...
  |     |     └── acoplar_02999.WAV
  |     ├── CANCELAR
  |     |     ├── cancelar_00000.WAV
  |     |     ├── ...
  |     |     └── cancelar_02999.WAV
  |     ├── CONTINUAR
  |     |     ├── continuar_00000.WAV
  |     |     ├── ...
  |     |     └── continuar_02999.WAV
  |     ├── FONDO
  |     |     ├── fondo_00000.WAV
  |     |     ├── ...
  |     |     └── fondo_02999.WAV
  |     ├── LEVANTADO
  |     |     ├── levantado_00000.WAV
  |     |     ├── ...
  |     |     └── levantado_02999.WAV
  |     ├── PRINCIPAL
  |     |     ├── principal_00000.WAV
  |     |     ├── ...
  |     |     └── principal_02999.WAV
  |     ├── REPETIR
  |     |     ├── repetir_00000.WAV
  |     |     ├── ...
  |     |     └── repetir_02999.WAV
  |     └── SALIR
  |     |     ├── salir_00000.WAV
  |     |     ├── ...
  |     |     └── salir_02999.WAV
  └── seeds
```

En caso de que solo se especifiquen las carpetas de **seeds** y **salida** (en nuestro ejemplo **generados_automaticamente**) el script utilizará los siguientes valores por defecto:
```
--seeds-root, default="seeds" (Carpeta con subcarpetas por clase.)
--noises-root, default="noises" (Carpeta con ruidos/charlas.)
--rirs-root", default="rirs" (Carpeta con RIR (Room Impulse Response/Respuesta al Impulso de la Habitación).)
--out-root, default="data" (Carpeta de salida.)
--per-class, default=3000 (Cantidad de ejemplos por clase (excluyendo FONDO).)
--fondo-minutes, default=60 (Minutos de FONDO (ruido/charlas).)
--p-shift, default=0.8 (Probabilidad de shift temporal.)
--p-stretch, default=0.5 (Probabilidad de estiramiento temporal.)
--p-pitch, default=0.5 (Probabilidad de cambio de pitch.)
--max-shift-ms, default=100 (Shift máximo en ms.)
--stretch-low, default=0.9 (Factor mínimo de estiramiento.)
--stretch-high, default=1.1 (Factor máximo de estiramiento.)
--pitch-semitones, default=1.0 (Semitonos máximos para pitch.)
--p-noise, default=0.9 (Probabilidad de agregar ruido.)
--p-reverb, default=0.3 (Probabilidad de agregar reverb.)
--snr-list, default="20,10,5,0" (Lista de SNRs en dB separadas por coma, 20dB ruido muy bajo, y 0dB ruido muy alto).
--use-fondo-as-noise", default=True (Aparte de ruido gaussiano, utiliza los .WAV guardado dentro de ./seeds/FONDO como ruidos adicionales.)

Resumen:
"p_*" controlan la probabilidad de aplicar cada transformación.
"use_fondo_as_noise" es especialmente útil en tu caso porque ya se hayan generado audios con diferentes fondos.
```

### c) Uso de **tensors_generator.py** para generar archivos **.pt** en forma automática a partir de **.WAV**:
**tensors_generator.py** utiliza las funciones dentro de **features.py** para poder generar los tensores .pt basados en los audios .WAV que contienen los espectrogramas en formato log-Mel con dimensiones **float32[1,1,98,64]**.
Tiene dos formas de funcionamiento, la primera forma es utilizarlo para generar tensores (archivos .pt) en forma individual a partir de un .WAV, la otra forma es generar un lote (batch) de tensores (archivos .pt) en base a una serie de archivos .WAV.
Para esto se utilizan los argumentos del **tensors_generator.py** en forma diferente.

#### c.1) Generación de un solo tensor a partir de un audio en formato .WAV dado:
```
python tensors_generator.py D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\audios\REPETIR\000_repetir.wav --plot --db --out-root D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\tensores
```

#### c.2) Generación de lotes de tensores a partir de audios en formato .WAV (SE UTILIZA ESTE MODO DE EJECUCIÓN):
```
python tensors_generator.py --generate-cache --wav-root D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\generados_automaticamente --out-root D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\tensores --sr 16000 --n_mels 64 --db --every 50
```

A la salida del script se verá el siguiente mensaje:
```
✔ Guardado D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\tensores\SALIR\salir_02999.pt
[24600/24600] procesados
Listo: 24600/24600 archivos procesados.
```

## 6) Entrenamiento de la red neuronal CRNN para uso en KWS con el script **train.py**:
Para poder armar y entrenar el modelo KWS basado en redes neuronales se utilizan los siguientes scripts:
- dataset_from_cache.py: Dataloader que se encargar de leer los tensores guardados en archivos .pt dentro de la carpeta ./tensores.
- model_crnn.py: Generador de CRNN pequeña (Conv2D + BiGRU + FC).
- train.py: Blucle de entrenamiento con validación y checkpoints.
- export_onnx.py: Exportador de modelo entrenado a ONNX (con T dinámico).

#### Uso de los scripts:
a) Uso de **train.py**:
```
python train.py --cache-root D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\tensores\ --epochs 20 --batch-size 64 --n-mels 64 --ckpt-dir  D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\trained_tensor\ --ckpt-name kws_crnn_small.pt --early-stop --patience 5 --min-delta 1e-4
```

b) Uso de **train.py** (SE UTILIZA ESTA FORMA SIN --early-stop):
```
python train.py --cache-root D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\tensores\ --epochs 20 --batch-size 64 --n-mels 64 --ckpt-dir  D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\trained_tensor\ --ckpt-name kws_crnn_small.pt
```

- Usa el directorio ./tensores/ por defecto para buscar los tensores (archivos .pt).
- Guarda el mejor checkpoint en **kws_crnn_small.pt** en donde se ejecutó el script.
- Muestra métricas de train/val, el test y la matriz de confusión.

Variantes del script **train.py**:
```
usage: train.py [-h] --cache-root CACHE_ROOT [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--lr LR] [--ckpt-dir CKPT_DIR] [--ckpt-name CKPT_NAME] [--n-mels N_MELS] [--early-stop] [--patience PATIENCE] [--min-delta MIN_DELTA]

options:
  -h, --help            show this help message and exit
  --cache-root CACHE_ROOT
  --epochs EPOCHS
  --batch-size BATCH_SIZE
  --lr LR
  --ckpt-dir CKPT_DIR   Directorio donde se guardará el checkpoint
  --ckpt-name CKPT_NAME Nombre de archivo de checkpoint
  --n-mels N_MELS       Número de bandas Mel (default: 64)
  --early-stop          Activa early stopping basado en val loss
  --patience PATIENCE   Epochs sin mejora antes de frenar (default: 5)
  --min-delta MIN_DELTA
                        Mejora mínima en val loss para resetear paciencia (default: 1e-4)
```

Valores por defecto en caso de sólo ejecutar el script:
```
--cache-root, required=True
--epochs,default=20
--batch-size, default=32
--lr, default=1e-3
--ckpt-dir, default="/data/trained_tensor" (Directorio donde se guardará el checkpoint.)
--ckpt-name, default="kws_crnn.pt" (Nombre de archivo de checkpoint.)
--n-mels, default=64 (Número de bandas Mel (default: 64).)
# Early stopping opcional
--early-stop, (Activa early stopping basado en val loss.)
--patience", default=5 (Epochs sin mejora antes de frenar (default: 5).)
--min-delta", default=1e-4 (Mejora mínima en val loss para resetear paciencia (default: 1e-4).)
```

#### La salida cuando se ejecuta **train.py** se trata del modelo tensorial .pt y una matriz de confusión:
Ejemplo de salida:
```
device: cpu
D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\.venv\Lib\site-packages\torch\utils\data\dataloader.py:666: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  warnings.warn(warn_msg)
Epoch 01 | train 0.5472/0.838 | val 0.0466/0.993
Epoch 02 | train 0.0253/0.996 | val 0.0187/0.996
Epoch 03 | train 0.0093/0.999 | val 0.0083/0.999
Epoch 04 | train 0.0025/1.000 | val 0.0044/0.999
Epoch 05 | train 0.0012/1.000 | val 0.0038/0.999
Epoch 06 | train 0.0008/1.000 | val 0.0038/0.999
Epoch 07 | train 0.0005/1.000 | val 0.0032/0.999
Epoch 08 | train 0.0004/1.000 | val 0.0031/0.999
Epoch 09 | train 0.0003/1.000 | val 0.0031/0.999
Epoch 10 | train 0.0002/1.000 | val 0.0028/0.999
Epoch 11 | train 0.0002/1.000 | val 0.0030/0.999
Epoch 12 | train 0.0002/1.000 | val 0.0033/0.999
Epoch 13 | train 0.0001/1.000 | val 0.0032/0.999
Epoch 14 | train 0.0001/1.000 | val 0.0029/0.999
Epoch 15 | train 0.0001/1.000 | val 0.0030/0.999
Early stop en epoch 15 (mejor val 0.0028 en 10)
Test   | loss 0.0052 | acc 0.998
Checkpoint guardado en: D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\trained_tensor\kws_crnn_small.pt | tiempo total: 431.3
Confusion matrix:
 [[305   0   0   0   0   0   0   0]
 [  0 294   0   0   0   0   0   0]
 [  0   0 305   1   0   0   1   0]
 [  0   0   0 369   0   0   0   0]
 [  0   0   0   1 310   0   0   0]
 [  0   1   0   0   0 278   1   0]
 [  0   0   0   0   0   0 295   0]
 [  0   0   0   0   0   0   0 299]]

```

Qué conviene hacer:
- Dejá correr las épocas (20 por ejemplo) y mirar si la validación se mantiene alta.
- Chequeo en test: al terminar, el script ya evalúa en test con el mejor checkpoint (early stopping).
- Exportá a ONNX una vez conforme con test.

#### Ejemplo 1 - Igual que ahora (sin early stop):
> python train.py --cache-root ./tensores --ckpt-dir ./models --ckpt-name kws_crnn_small.pt

#### Ejemplo 2 - Con early stop (patience 5, min_delta 1e-4):
> python train.py --cache-root ./tensores --ckpt-dir ./models --ckpt-name kws_crnn_small.pt --early-stop --patience 5 --min-delta 1e-4

## 7) Exportador a modelo ONNX **export_onnx.py**:
Este script convierte **kws_crnn.pt** en **kws_crnn.onxx**, es decir convierte el modelo tensorial en un archivo con formato Open Neural Network Exchange (ONNX). Al final genera un archivo **kws_crnn.onnx** con n-mel (número de bandas Mel) y T (número de frames temporales) configurables.
Por defecto n-Mel=64 y T=98.

Valores por defecto en caso de sólo ejecutar el script:
```
usage: export_onnx.py [-h] --ckpt-path CKPT_PATH [--onnx-out-dir ONNX_OUT_DIR] [--onnx-out-name ONNX_OUT_NAME] [--n-mels N_MELS] [--T T]

options:
  -h, --help            show this help message and exit
  --ckpt-path CKPT_PATH   Ruta al archivo .pt con los pesos entrenados
  --onnx-out-dir ONNX_OUT_DIR    Directorio donde se guardará el archivo ONNX (default: .)
  --onnx-out-name ONNX_OUT_NAME    Nombre del archivo ONNX (opcional, se genera si no se indica)
  --n-mels N_MELS    Número de bandas Mel (default: 64)
  --T T    Número de frames temporales (default: 98)
```

Ejemplos de ejecución:
#### Exporta al directorio actual, con nombre generado automáticamente:
```
python export_onnx.py --ckpt-path ./kws_crnn_small.pt
```

#### Exporta a un directorio específico con nombre por defecto:
```
python export_onnx.py --ckpt-path ./kws_crnn_small.pt --onnx-out-dir ./onnx_models/
```

#### Exporta con nombre personalizado en un directorio (SE UTILIZA ESTE):
```
python .\export_onnx.py --ckpt-path D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\trained_tensor\kws_crnn_small.pt --onnx-out-dir D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\onnx_model --onnx-out-name kws_crnn_small.onnx
```

## 8) Generación de datos de calibración para modelo *.axmodel** con script **make_calib.py**:
Este script genera los datos de calibración para que Pulsar2 pueda:
- Sirven para cuantizar bien el modelo a INT8.
- Se usan solo en la compilación, no en la inferencia.
- Mejoran la precisión del modelo cuantizado al reflejar las distribuciones reales de tus datos.
- En el .axmodel quedan guardadas las escalas y offsets, no los archivos de calibración.

¿Dónde se usan dentro del .axmodel?
- Durante el proceso de pulsar2 build con --quant.*, Pulsar2:
- Lee los calibration samples (input_00000.npy, input_00001.npy …).
- Simula el forward del modelo en FP32 usando esos datos.
- Ajusta las escalas y offsets de cada tensor (scale, zero_point) para almacenarlas en el .axmodel.
- Esas escalas quedan incrustadas en el .axmodel y el runtime del NPU ya no necesita más los datos de calibración. Solo los usás en la fase de compilación.

Ejemplo práctico para el caso del KWS:
- En tu modelo de detección de palabras clave (KWS CRNN):
- La entrada es un espectrograma Mel [1,1,98,64].
- Si cuantizás sin calibrar, Pulsar2 podría asumir un rango genérico (ej. -1.0 a 1.0) y saturar la energía de ciertas frecuencias.
- Con calibración, usando espectrogramas reales, ajusta los scale por canal para que cada banda Mel use el rango correcto → resultado:
  - menos pérdida en la predicción,
  - precisión más cercana al FP32,
  - menos falsos positivos/negativos.

Valores por defecto en caso de sólo ejecutar el script:
```
usage: make_calib.py [-h] --pt-root PT_ROOT --out-dir OUT_DIR [--calib-format {Numpy,Binary,NumpyObject,Image}] [--t T] [--n-mels N_MELS] [--max-samples MAX_SAMPLES] [--seed SEED] [--tar-name TAR_NAME]

options:
  -h, --help            show this help message and exit
  --pt-root PT_ROOT     Raíz de tensores .pt (ej: /data/tensores)
  --out-dir OUT_DIR     Directorio de salida (ej: /data/calib_samples)
  --calib-format {Numpy,Binary,NumpyObject,Image}
                        Formato de datos de calibración para Pulsar2
  --t T                 Frames T fijos (debe coincidir con ONNX)
  --n-mels N_MELS       Número de bandas Mel esperado
  --max-samples MAX_SAMPLES
                        Cantidad de muestras a exportar
  --seed SEED           Semilla para muestreo aleatorio
  --tar-name TAR_NAME   Nombre del archivo TAR de salida (por defecto: calib_kws.tar)
```

Ejemplo de ejecución:
```
python.exe .\make_calib.py --pt-root D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\tensores --out-dir D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\calib_samples --calib-format "Numpy" --t 98 --n-mels 64 --max-samples 300 --seed 123 --tar-name "calib_kws.tar"
```

## 9) Generador de archivos .bin para prueba directamente en el M5Stack - AX630C:
Este script sirve para convertir el dataset de calibración en archivos .bin que entienda directamente el comando **ax_run_model** dentro del Ubuntu 22.04 dentro del M5Stack - AX630C.

Valores por defecto en caso de sólo ejecutar el script:
```
usage: generate_test_files.py [-h] --src SRC --out-dir OUT_DIR [--t T] [--n-mels N_MELS] [--limit LIMIT] [--tensor-name TENSOR_NAME] [--allow-image] [--strict-bin]

Convert make_calib.py samples to ax_run_model inputs

options:
  -h, --help            show this help message and exit
  --src SRC             Directory containing input_* files OR a calib_kws.tar produced by make_calib.py
  --out-dir OUT_DIR     Destination root for ax_run_model inputs (will create input/, output/, list.txt)
  --t T                 Frames T expected by model
  --n-mels N_MELS       Mel bands F expected by model
  --limit LIMIT         Max number of samples to convert (0 = all)
  --tensor-name TENSOR_NAME
                        Input tensor name (affects file name: <tensor>.bin)
  --allow-image         Allow converting PNG (lossy visualization) to features
  --strict-bin          Validate .bin size equals T*F*4 bytes (float32)
```

## 10) Diagrama en Bloques del flujo previsto de la CRNN (Convolucional):
```
      ┌──────────────────────────────────────────────────────────┐
      │                INPUT  (NCHW = 1×1×T×M)                   │
      │                  features[1,1,98,64]                     │
      └───────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
                          ┌─────────────────┐
                          │    Conv2D #1    │  k=3×3, s=1, p=1
                          └───────┬─────────┘
                                  │
                                  ▼
                          ┌─────────────────┐
                          │      ReLU       │
                          └───────┬─────────┘
                                  │
                                  ▼
                          ┌─────────────────┐
                          │  MaxPool2D #1   │  k=2×2, s=2
                          │   ↓↓ T, ↓↓ M    │  ≈ 98→49, 64→32
                          └───────┬─────────┘
                                  │
                                  ▼
                          ┌─────────────────┐
                          │    Conv2D #2    │  k=3×3, s=1, p=1
                          └───────┬─────────┘
                                  │
                                  ▼
                          ┌─────────────────┐
                          │      ReLU       │
                          └───────┬─────────┘
                                  │
                                  ▼
                          ┌─────────────────┐
                          │  MaxPool2D #2   │  k=2×2, s=2
                          │  ↓↓ T, ↓↓ M     │  ≈ 49→24, 32→16
                          └───────┬─────────┘
                                  │
                                  ▼
    ┌───────────────────────────────────────────────────────────────┐
    │           Permute / Reshape a SECUENCIA (para RNN)            │
    │           N×C×T×F  ──►  N×T×(C·F)                             │
    │           (C = canales tras Conv2; F ≈ 16; T ≈ 24)            │
    └───────────────────────────────┬───────────────────────────────┘
                                    │
                                    ▼
                          ┌───────────────────┐
                          │     GRU (RNN)     │  (1 capa; posiblemente bi-dir)
                          │   secuencia T×D   │
                          └─────────┬─────────┘
                                    │  (toma el último/mean pooling sobre T)
                                    ▼
                          ┌──────────────────┐
                          │     GEMM / FC    │  → 8 logits (clases)
                          └─────────┬────────┘
                                    │
                                    ▼
                            ┌─────────────────┐
                            │     Softmax     │  (normalización p/ lectura)
                            └───────┬─────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │    OUTPUT: logits[1, 8]   │
                      │  {ACOPLAR, CANCELAR, ...} │
                      └───────────────────────────┘

```

### Explicación de cada bloque de la red neuronal para que funciones el sistema KWS:
- **Entrada (features [1,1,98,64]):**
Qué es: Un espectrograma Mel (1 canal, 98 pasos de tiempo, 64 bandas Mel).
Función: Representar la señal de audio ya transformada a características espectrales.

- **Conv2D #1:**
Qué hace: Aplica filtros convolucionales 2D sobre el espectrograma.
Función: Detectar patrones locales en tiempo-frecuencia (ej. formantes, cambios de energía).

- **ReLU:**
Qué hace: Función de activación que mantiene valores positivos y descarta negativos.
Función: Añadir no-linealidad para que la red aprenda representaciones más complejas.

- **MaxPool2D #1:**
Qué hace: Reduce la resolución tomando el valor máximo en ventanas (ej. 2×2).
Función:
Disminuir dimensionalidad → menos parámetros.
Conservar las características más relevantes (los picos de energía).

- **Conv2D #2:**
Qué hace: Otra capa de filtros convolucionales.
Función: Extraer representaciones más abstractas (ej. patrones de fonemas, estructuras repetitivas).

- **ReLU + MaxPool2D #2:**
Función conjunta: Igual que antes, pero ahora ya actúan sobre características de más alto nivel.
Después de este bloque, el espectrograma ya está reducido en tiempo y frecuencia, pero cada “pixel” tiene más significado semántico.

- **Reshape / Transpose:**
Qué hace: Convierte la salida convolucional en una secuencia [T, features].
Función: Adaptar los datos para pasarlos a la RNN (GRU).

- **GRU:**
Qué hace: Es una red recurrente que procesa la secuencia paso a paso.
Función: Captura dependencias temporales (ej. cómo evoluciona el espectro a lo largo del tiempo).
Es clave en KWS porque las palabras no se distinguen solo por un instante, sino por cómo cambian en el tiempo.

- **GEMM / Fully Connected:**
Qué hace: Toma la salida del GRU y la proyecta en un espacio de 8 dimensiones.
Función: Generar los logits (las puntuaciones no normalizadas para cada palabra clave).

- **Softmax (implícito, fuera del ONNX):**
Qué hace: Convierte los logits en probabilidades (suma 1).
Función: Facilita interpretar “cuál palabra fue dicha” y con qué confianza.

- **Salida (logits [1,8]):**
Qué es: Vector con 8 valores, uno por clase:
{ACOPLAR, CANCELAR, CONTINUAR, FONDO, LEVANTADO, PRINCIPAL, REPETIR, SALIR}
Función: Clasificación final de la palabra clave detectada.

## 11) Resumen rápido de ejecución de scripts (desde generación de dataset hasta conversión de ONNX a AXMODEL y carga en AX630C):
### Todos los script se ejecutan desde la carpeta ./src del presente proyecto: 
> 1) python.exe .\make_dataset.py --seeds-root D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\seeds --noises-root D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\seeds\FONDO --out-root D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\generados_automaticamente

> 2) python.exe .\tensors_generator.py --generate-cache --wav-root D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\generados_automaticamente --out-root D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\tensores --sr 16000 --n_mels 64 --db --every 50

> 3) python.exe .\train.py --cache-root D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\tensores\ --epochs 20 --batch-size 64 --n-mels 64 --ckpt-dir  D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\trained_tensor\ --ckpt-name kws_crnn_small.pt

> 4) python.exe .\export_onnx.py --ckpt-path D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\trained_tensor\kws_crnn_small.pt --onnx-out-dir D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\onnx_model --onnx-out-name kws_crnn_small.onnx

> 5) python.exe .\make_calib.py --pt-root D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\tensores --out-dir D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\calib_samples --calib-format "Numpy" --t 98 --n-mels 64 --max-samples 300 --seed 123 --tar-name "calib_kws.tar"

> 6) python.exe .\generate_test_files.py --src D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\calib_samples\calib_kws.tar --out-dir D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\binary_for_npu\ 

### A partir del punto 6 debe pasarse al entorno **Pulsar2** para poder convertir el modelo **.onnx** generado al modelo **.axmodel** necesario para que el chip IA Axera AX630C o AX650N puedan entender el modelo KWS.

### Seguir instrucciones de instalación y configuración de Pulsar2 con Docker:

#### Link con indicación de pasos para instalar Pulsa2: [https://pulsar2-docs.readthedocs.io/en/latest/user_guides_quick/quick_start_prepare.html]

#### Copiar el modelo **.onnx**, producto del entrenamiento, dentro del Ubuntu 22.04 del M5Stack LLM Kit, para ello debemos conectarnos al mismo via **ssh** y utilizando el comando SCP desde dentro del módulo:
> scp "User@192.168.###.###:/./data/output/compiled.axmodel" /root/kws_int8.axmodel

#### Copiar los archivos de calibración dentro del directorio **/root** dentro del Ubuntu 22.04 del M5Stack LLM Kit, para ello debemos conectarnos al mismo via **ssh** y utilizando el comando SCP desde dentro del módulo:
> scp -r User@192.168.###.###:/D:/04_ProyASRC/VSCode/20250826_M5Stack_LLM-CoreS3/20250921_kws_project/data/binary_for_npu/ /root

Se ha probado ejecutar el comando **ax_run_model** con más de un input, sin suerte. Es por esto que, si bien **generate_test_files.py** genera 300 inputs para calibración, al ejecutar el siguiente comando:
> ax_run_model -m kws_int8.axmodel -i ./binary_for_npu/input/ -o ./binary_for_npu/output/ -l ./binary_for_npu/list.txt

Con un **list.txt** con 0 a 299 inputs, no funciona, dando el error **} does not has input folder {./binary_for_npu/input/}, skipped.**:
```
root@m5stack-LLM:~# ax_run_model -m kws_int8.axmodel -i ./binary_for_npu/input/ -o ./binary_for_npu/output/ -l ./binary_for_npu/list.txt
  Total found {16} input stimulus folders.
} does not has input folder {./binary_for_npu/input/}, skipped.
} does not has input folder {./binary_for_npu/input/}, skipped.
} does not has input folder {./binary_for_npu/input/}, skipped.
} does not has input folder {./binary_for_npu/input/}, skipped.
} does not has input folder {./binary_for_npu/input/}, skipped.
} does not has input folder {./binary_for_npu/input/}, skipped.
} does not has input folder {./binary_for_npu/input/}, skipped.
} does not has input folder {./binary_for_npu/input/}, skipped.
} does not has input folder {./binary_for_npu/input/}, skipped.
} does not has input folder {./binary_for_npu/input/}, skipped.
} does not has input folder {./binary_for_npu/input/}, skipped.
} does not has input folder {./binary_for_npu/input/}, skipped.
} does not has input folder {./binary_for_npu/input/}, skipped.
} does not has input folder {./binary_for_npu/input/}, skipped.
} does not has input folder {./binary_for_npu/input/}, skipped.
  Inferring model, total 1/16. Done.
  ------------------------------------------------------
  min =   1.277 ms   max =   1.277 ms   avg =   1.277 ms
  ------------------------------------------------------
```

No se encuentra forma de ejecutar más de un input con **ax_run_model**, se prueban todos los parámetros posibles, no funciona ninguno. Es por esto que **list.txt** tiene que tener un sólo input, es decir solamente un 0 en la primera línea sin espacios, ni saltos de línea.

#### Hay dos maneras de convertir el modelo de **.onnx** a **.axmodel**, uno es tipeando desde línea de comandos dentro de Pulsar2 ejecutado en Docker toda la sentencia y la otra (mucho más fácil) es llamando a un archivo de configuración (utilizaremos esta forma): 

Copiamos el archivo de configuración **kws_build_int8_config.json** dentro de la carpeta **/data/config/kws_build_int8_config.json** dentro del sitio de ejecución de Pulsar2:
> cp kws_build_int8_config.json ./data/config/kws_build_int8_config.json

#### Lanzamos el proceso de compilación dentro de Pulsar2 en Docker que convertirá el modelo **.onnx** a **.axmodel**:
> pulsar2 build --config /data/config/kws_build_int8_config.json

#### Se verá algo similar a la siguiente salida:
```
root@c9626233eb7d:/data# pulsar2 build --config /data/config/kws_build_int8_config.json
+-------------+----------------------------+
| Model Name  |         OnnxModel          |
+-------------+----------------------------+
| Model Info  | Op Set: 13 / IR Version: 7 |
+-------------+----------------------------+
|  IN: input  |  float32: (1, 1, 98, 64)   |
| OUT: output |      float32: (1, 8)       |
+-------------+----------------------------+
|   Concat    |             1              |
|  Constant   |             8              |
|    Conv     |             2              |
|   Expand    |             1              |
|     GRU     |             1              |
|   Gather    |             2              |
|    Gemm     |             1              |
|   MaxPool   |             2              |
|    Relu     |             2              |
|   Reshape   |             2              |
|    Shape    |             1              |
|  Transpose  |             4              |
|  Unsqueeze  |             1              |
+-------------+----------------------------+
| Model Size  |         893.61 KB          |
+-------------+----------------------------+
2025-09-28 06:19:25.075 | WARNING  | yamain.command.build:fill_default:243 - force onnxsim due to found op_type in {'ConstantOfShape', 'Constant'}
2025-09-28 06:19:25.075 | WARNING  | yamain.command.build:fill_default:313 - apply default output processor configuration to ['output']
2025-09-28 06:19:25.078 | INFO     | yamain.common.util:extract_archive:148 - extract [/data/dataset/calib_kws.tar] to [/data/output/quant/dataset/input]...
2025-09-28 06:19:26.797 | WARNING  | yamain.command.load_model:optimize_onnx_model:894 - parse onnx with random data, because parse_input_samples is None or opt.input_tensor_infos is not None
2025-09-28 06:19:26.803 | INFO     | frontend.parsers.onnx_parser:parse_onnx_model_proto:87 - onnxslim...
+--------------+----------------------------+----------------------------+
|  Model Name  |         OnnxModel          |         OnnxModel          |
+--------------+----------------------------+----------------------------+
|  Model Info  | Op Set: 13 / IR Version: 7 | Op Set: 13 / IR Version: 7 |
+--------------+----------------------------+----------------------------+
|  IN: input   |  float32: (1, 1, 98, 64)   |  float32: (1, 1, 98, 64)   |
| OUT: output  |      float32: (1, 8)       |      float32: (1, 8)       |
+--------------+----------------------------+----------------------------+
|    Concat    |             1              |             0              |
|   Constant   |             8              |             0              |
|     Conv     |             2              |             2              |
|    Expand    |             1              |             0              |
|     GRU      |             1              |             1              |
|    Gather    |             2              |             1              |
|     Gemm     |             1              |             1              |
|   MaxPool    |             2              |             2              |
|     Relu     |             2              |             2              |
|   Reshape    |             2              |             2              |
|    Shape     |             1              |             0              |
|  Transpose   |             4              |             4              |
|  Unsqueeze   |             1              |             0              |
+--------------+----------------------------+----------------------------+
|  Model Size  |         893.61 KB          |         893.45 KB          |
+--------------+----------------------------+----------------------------+
| Elapsed Time |                          0.09 s                         |
+--------------+----------------------------+----------------------------+
Building onnx ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
2025-09-28 06:19:26.993 | SUCCESS  | opset.utils:check_data:1432 - check onnx parser e2e [output], (1, 8), float32 successfully!
2025-09-28 06:19:26.996 | WARNING  | yamain.command.load_model:optimize_onnx_model:912 - modify input shape to {'input': (1, 1, 98, 64)}
2025-09-28 06:19:27.228 | SUCCESS  | opset.utils:check_data:1432 - check onnx transformations e2e [output], (1, 8), float32 successfully!
2025-09-28 06:19:27.228 | INFO     | yamain.command.build:quant:817 - save optimized onnx to [/data/output/frontend/optimized.onnx]
                                           Quant Config Table
┏━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━┓
┃ Input ┃ Shape          ┃ Dataset Directory                ┃ Data Format ┃ Tensor Format ┃ Mean ┃ Std ┃
┡━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━┩
│ input │ [1, 1, 98, 64] │ /data/output/quant/dataset/input │ Numpy       │ GRAY          │ [0]  │ [1] │
└───────┴────────────────┴──────────────────────────────────┴─────────────┴───────────────┴──────┴─────┘
Quantization calibration will be executed on cpu
Transformer optimize level: 0
[Warning]Unexpected input value of operation /gru/GRU, recieving "None" at its input 4
[Warning]Unexpected input value of operation /gru/GRU, recieving "None" at its input 4
Statistics Inf tensor: 100%|█████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 216.41it/s]
[06:19:27] AX Set Float Op Table Pass Running ...
[Info]Use dynamic gru quantization
[06:19:28] AX Set MixPrecision Pass Running ...
[06:19:28] AX Set LN Quant dtype Quant Pass Running ...
[06:19:28] AX Reset Mul Config Pass Running ...
[06:19:28] AX Refine Operation Config Pass Running ...
[06:19:28] AX Tanh Operation Format Pass Running ...
[06:19:28] AX Confused Op Refine Pass Running ...
[06:19:28] AX Quantization Fusion Pass Running ...
[06:19:28] AX Quantization Simplify Pass Running ...
[06:19:28] AX Parameter Quantization Pass Running ...
[06:19:28] AX Runtime Calibration Pass Running ...
Calibration Progress(Phase 1): 100%|█████████████████████████████████████████████████| 300/300 [00:02<00:00, 148.54it/s]
[06:19:30] AX Quantization Alignment Pass Running ...
[06:19:30] AX Refine Int Parameter Pass Running ...
[06:19:30] AX Refine Scale Pass Running ...
[06:19:30] AX Passive Parameter Quantization Running ...
[06:19:30] AX Parameter Baking Pass Running ...
--------- Network Snapshot ---------
Num of Op:                    [15]
Num of Quantized Op:          [15]
Num of Variable:              [31]
Num of Quantized Var:         [31]
------- Quantization Snapshot ------
Num of Quant Config:          [45]
BAKED:                        [5]
OVERLAPPED:                   [23]
ACTIVATED:                    [5]
SOI:                          [2]
PASSIVE_BAKED:                [3]
FP32:                         [7]
Network Quantization Finished.
[Warning]File /data/output/quant/quant_axmodel.onnx has already exist, quant exporter will overwrite it.
[Warning]File /data/output/quant/quant_axmodel.json has already exist, quant exporter will overwrite it.
Do quant optimization
quant.axmodel export success:
        /data/output/quant/quant_axmodel.onnx
        /data/output/quant/quant_axmodel.data
===>export io data to folder: /data/output/quant/debug/io
Building native ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
2025-09-28 06:19:31.470 | INFO     | yamain.command.build:compile_ptq_model:1113 - group 0 compiler transformation
2025-09-28 06:19:31.473 | WARNING  | yamain.command.load_model:post_process:626 - postprocess tensor [output]
2025-09-28 06:19:31.473 | INFO     | yamain.command.load_model:ir_compiler_transformation:821 - use quant data as gt input: input, float32, (1, 1, 98, 64)
2025-09-28 06:19:31.707 | INFO     | yamain.command.build:compile_ptq_model:1134 - group 0 QuantAxModel macs: 13,437,952
2025-09-28 06:19:32.013 | INFO     | yamain.command.build:compile_ptq_model:1266 - subgraph [0], group: 0, type: GraphType.NPU
2025-09-28 06:19:32.013 | INFO     | yamain.compiler.npu_backend_compiler:compile:185 - compile npu subgraph [0]
tiling op...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1223/1223 0:00:05
new_ddr_tensor = []
build op serially...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1133/1133 0:00:09
build op...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1247/1247 0:00:00
add ddr swap...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 10080/10080 0:00:05
calc input dependencies...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 10221/10221 0:00:01
calc output dependencies...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 10221/10221 0:00:01
assign eu heuristic   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 10221/10221 0:00:00
assign eu onepass   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 10221/10221 0:00:00
assign eu greedy   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 10221/10221 0:00:00
2025-09-28 06:19:57.546 | INFO     | yasched.test_onepass:results2model:2737 - clear job deps
2025-09-28 06:19:57.546 | INFO     | yasched.test_onepass:results2model:2738 - max_cycle = 648,294
build jobs   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 10221/10221 0:00:03
2025-09-28 06:20:01.894 | INFO     | yamain.compiler.npu_backend_compiler:compile:246 - assemble model [0] [subgraph_npu_0] b1
2025-09-28 06:20:10.227 | DEBUG    | backend.ax620e.linker:link_with_dispatcher:1935 - eu_chunk time_limit: True
2025-09-28 06:20:10.228 | DEBUG    | backend.ax620e.linker:link_with_dispatcher:1936 - eu_chunk only_bysize: False
2025-09-28 06:20:36.065 | INFO     | backend.ax620e.linker:link_with_dispatcher:2004 - DispatcherQueueType.IO: Generate 233 EU chunks, 20 Dispatcher Chunk
2025-09-28 06:20:36.065 | INFO     | backend.ax620e.linker:link_with_dispatcher:2004 - DispatcherQueueType.Compute: Generate 225 EU chunks, 39 Dispatcher Chunk
2025-09-28 06:20:36.065 | INFO     | backend.ax620e.linker:link_with_dispatcher:2005 - EU mcode size: 431 KiB
2025-09-28 06:20:36.065 | INFO     | backend.ax620e.linker:link_with_dispatcher:2006 - Dispatcher mcode size: 54 KiB
2025-09-28 06:20:36.065 | INFO     | backend.ax620e.linker:link_with_dispatcher:2007 - Total mcode size: 485 KiB
2025-09-28 06:20:36.096 | INFO     | yamain.command.build:compile_ptq_model:1308 - fuse 1 subgraph(s)
```
#### En caso de no ver algo así en la salida y que de error, se deberá verificar que los pasos del 1 al 6 se hayan realizado bien y luego verificar el archivo **kws_build_int8_config.json**.

#### Si la conversión a **.axmodel** es satisfactoria, será posible ejecutar dentro del M5Stack LLM Kit con AX630C el siguiente comando para verificar que el modelo KWS basado en redes neuronales se ejecute bien:
> root@m5stack-LLM:~# ax_run_model -m /root/kws_int8.axmodel -w 10 -r 100

- -w 10 = warmup de 10 muestras para estabilizar.
- -r 100 = Se ejecutan 100 inferencias reales del modelo kws_int8.axmodel.

#### Se verá la siguiente salida:
```
root@m5stack-LLM:~# ax_run_model -m /root/kws_int8.axmodel -w 10 -r 100
   Run AxModel:
         model: /root/kws_int8.axmodel
          type: Half
          vnpu: Disable
      affinity: 0b01
        warmup: 10
        repeat: 100
         batch: { auto: 0 }
   pulsar2 ver: 4.2 751f68f9
    engine ver: 2.6.3sp
      tool ver: 2.3.3sp
      cmm size: 748716 Bytes
  ---------------------------------------------------------------------------
  min =   1.056 ms   max =   1.079 ms   avg =   1.058 ms  median =   1.058 ms
   5% =   1.062 ms   90% =   1.057 ms   95% =   1.057 ms     99% =   1.056 ms
  ---------------------------------------------------------------------------
  ```
#### Algunos puntos a tener en cuenta:
- Si bien muestra **type: Half** debería mostrar INT8, no se ha logrado que muestre ese tipo de quantización del modelo KWS, entiendo que debe ser debido al frontend de **ax_run_model**.
- La salida indica **vnpu: Disable** lo cual es incoherente debido a que, por un lado se ha configurado para que sea ejecutado en NPU1, y por otro, los tiempos de inferencia son P5=1.062ms, P90=1.057ms, P95=1.057ms y P99=1.056ms. Esto indica que no está utilizando los CPU Dual Cortex A53 hasta 1.2GHz.
- Otra cuestión que se verifica es que posiblemente los labels P5, P90, P95 y P99 estén invertidos, esto debido a que nunca P99 puede tardar menos que P5. Asimismo los valores son estables y coherentes.

#### Se verifica estabilidad y tiempos de inferencia con 5000 muestras:
```
root@m5stack-LLM:~# ax_run_model -m /root/kws_int8.axmodel -w 10 -r 5000
   Run AxModel:
         model: /root/kws_int8.axmodel
          type: Half
          vnpu: Disable
      affinity: 0b01
        warmup: 10
        repeat: 5000
         batch: { auto: 0 }
   pulsar2 ver: 4.2 751f68f9
    engine ver: 2.6.3sp
      tool ver: 2.3.3sp
      cmm size: 748716 Bytes
  ---------------------------------------------------------------------------
  min =   1.056 ms   max =   1.236 ms   avg =   1.060 ms  median =   1.058 ms
   5% =   1.064 ms   90% =   1.057 ms   95% =   1.057 ms     99% =   1.056 ms
  ---------------------------------------------------------------------------
  ```
#### Se verifica que para 5000 muestras la estabilidad se mantiene al igual que con 100 muestras.

## 12) Ejecución de **ax_run_model** con un archivo con una muestra de espectrograma real convertido en **.bin**:
#### Copiar los archivos de calibración dentro del directorio **/root** dentro del Ubuntu 22.04 del M5Stack LLM Kit, para ello debemos conectarnos al mismo via **ssh** y utilizando el comando SCP desde dentro del módulo:
> scp -r User@192.168.###.###:/D:/04_ProyASRC/VSCode/20250826_M5Stack_LLM-CoreS3/20250921_kws_project/data/binary_for_npu/ /root

#### Estructura de **list.txt** y directorio **./input**:
Contenido de **input.txt**:
```
0
```
Esquema de directorio input (tener en cuenta que sólo es posible utilizar el primer archivo dentro de **./input/0/input.bin**, es resto de los directorios no son encontrados por **ax_run_model**):
```
binary_for_npu
  └── input (carpeta conteniendo un audio .WAV por cada una de las 7 clases de los comandos y 1 clase con los fondos generados por grabación)
        ├── 0
        |   └── input.bin
        ├── 1
        |   └── input.bin
        ├── 2
        |   └── input.bin
        ├── 3
        |   └── input.bin
        ├── 4
        |   └── input.bin
        ├── 5
        |   └── input.bin
        ...
        └── 299
            └── input.bin
```

Se ha probado ejecutar el comando **ax_run_model** con más de un input, sin suerte. Es por esto que, si bien **generate_test_files.py** genera 300 inputs (0 a 299) para calibración, al ejecutar el siguiente comando:
> ax_run_model -m kws_int8.axmodel -i ./binary_for_npu/input/ -o ./binary_for_npu/output/ -l ./binary_for_npu/list.txt

Con un **list.txt** con 0 a 299 inputs, no funciona, dando el error **} does not has input folder {./binary_for_npu/input/}, skipped.**:
```
root@m5stack-LLM:~# ax_run_model -m kws_int8.axmodel -i ./binary_for_npu/input/ -o ./binary_for_npu/output/ -l ./binary_for_npu/list.txt
  Total found {300} input stimulus folders.
} does not has input folder {./binary_for_npu/input/}, skipped.
} does not has input folder {./binary_for_npu/input/}, skipped.
} does not has input folder {./binary_for_npu/input/}, skipped.
} does not has input folder {./binary_for_npu/input/}, skipped.
} does not has input folder {./binary_for_npu/input/}, skipped.
} does not has input folder {./binary_for_npu/input/}, skipped.
} does not has input folder {./binary_for_npu/input/}, skipped.
} does not has input folder {./binary_for_npu/input/}, skipped.
} does not has input folder {./binary_for_npu/input/}, skipped.
} does not has input folder {./binary_for_npu/input/}, skipped.
} does not has input folder {./binary_for_npu/input/}, skipped.
} does not has input folder {./binary_for_npu/input/}, skipped.
} does not has input folder {./binary_for_npu/input/}, skipped.
} does not has input folder {./binary_for_npu/input/}, skipped.
...
} does not has input folder {./binary_for_npu/input/}, skipped.
  Inferring model, total 1/299. Done.
  ------------------------------------------------------
  min =   1.331 ms   max =   1.331 ms   avg =   1.331 ms
  ------------------------------------------------------

root@m5stack-LLM:~# ls ./binary_for_npu/output/
299
```

Se ve que de 300 archivos **input.bin** en cada uno de los directorios de inputs, solamente termina ejecutando la inferencia de un solo archivo, el directorio creado en **./binary_for_npu/outputs** corresponde al último, siendo en este caso el número **299**. Dentro del mismo se encuentra el archivo **output.bin** con un tamaño de 32 bytes que contiene la probabilidad de cada una de las 8 clases, en 4 bytes cada una, la suma total de probabilidad de las 8 clases da 1 o 100% de probabilidad.

No se encuentra forma de ejecutar más de un input con **ax_run_model**, se prueban todos los parámetros posibles, no funciona ninguno. Es por esto que **list.txt** tiene que tener un sólo input, es decir solamente un 0 en la primera línea sin espacios, ni saltos de línea.

El resto de los archivos **input.bin** con espectrogramas Mel=64 y T=98 no son ejecutados, se debe a un problema en **ax_run_model**.

Igualmente sirve para probar un archivo real y, de esta manera, medir realmente los tiempos de inferencia con un ejemplo de la vida real. Esto confirma que lee correctamente el archivo **.bin** con datos reales y que en el directorio ./binary_for_npu/output

## 13) Script para analizar el archivo de salida **output.bin** e indicar la probabilidad inferida por **ax_run_model**:
Este script sirve para visualizar las probabilidades que generó la inferencia de archivos **.bin** con los espectrogramas log-Mel, el mismo se ejecuta en la PC y debe copiarse para esto el archivo desde el M5Stack AX630C a la PC, luego se ejecuta el scripty muestra las probabilidades detectadas por el modelo KWS entrenado para cada una de las 8 clases.

Los valores en hexadecimal dentro de **output.bin** son los siguientes:

- [0] 94 34 3D 41 = 11.825336456298828
- [1] FA 0F 2B C0 = -2.6728501319885254
- [2] 60 D1 78 3E = 0.24298620223999023
- [3] 04 3F B0 BF = -1.3769230842590332
- [4] C6 C6 06 C0 = -2.105882167816162
- [5] D4 24 91 C0 = -4.535745620727539
- [6] 20 FB C4 BF = -1.5389137268066406
- [7] 70 D1 78 BF = -0.9719457626342773

Para calcular la probabilidad de cada una de las clases se debe aplicar la función **Softmax**:
> Softmax:  P(i) = exp(x_i) / (exp(x_1) + exp(x_2) + ... + exp(x_N))

Donde:
- **x_i** es el logit de la clase i.
- **N** es el número total de clases.
- **P(i)** es la probabilidad de la clase i.

Probabilidades:

- [0] ACOPLAR = 0.999983
- [1] CANCELAR = 0.000001
- [2] CONTINUAR = 0.000009
- [3] FONDO = 0.000002
- [4] LEVANTADO = 0.000001
- [5] PRINCIPAL = 0.000000
- [6] REPETIR = 0.000002
- [7] SALIR = 0.000003

Sum: 1.000000
Top-1: [0] ACOPLAR (100.00%)

Al final la palabra detectada por el KWS es ACOPLAR.

## 14) Niveles de Seguridad Industrial de Pulsar:
Pulsar2 ha sido sometido a una evaluación de seguridad funcional de acuerdo con la norma ISO-26262 Parte 8.  
Ha sido clasificado con un Nivel de Confianza de Herramienta 3 (TCL3), respaldado por una cobertura completa de pruebas y procedimientos de manejo de anomalías.  
La herramienta cuenta con más de 2 años de implementación en múltiples proyectos de clientes y ha superado la certificación de seguridad de terceros.  
Cuando se utiliza según las instrucciones, Pulsar2 puede aplicarse al desarrollo de modelos con requisitos de seguridad de hasta ASIL-D.  
Esto garantiza su idoneidad para aplicaciones automotrices e industriales críticas en seguridad.  

## 15) Disclosure:
Axera, AX630C, AX650N, M5Stack, Docker, Pulsar2, Audacity, Adobe Premiere Pro Microsoft, Windows 11 y Grabadora de Sonido de Windows son marcas comerciales o marcas registradas de sus respectivos propietarios. Todos los nombres de productos, logotipos y marcas mencionados se utilizan únicamente con fines de identificación. No reclamamos ningún derecho de propiedad sobre estas marcas, y no estamos afiliados, avalados ni somos responsables por ellas de ninguna manera.  

Asimismo, no asumimos ninguna responsabilidad por problemas, daños o consecuencias que puedan surgir de la ejecución, uso o mala interpretación de los algoritmos, ejemplos o procedimientos proporcionados. Todo el contenido es únicamente con fines informativos y educativos, y cualquier implementación se realiza bajo el propio riesgo del usuario.  
