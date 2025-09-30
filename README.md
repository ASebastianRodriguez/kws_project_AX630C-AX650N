# KeyWord Spotting (KWS) for M5Stack LLM Kit (Axera AX630C) 
# With minor changes it is functional on AX650N

## 1) Initial Data:
- Developed by:      ASRC.
- Date:                     23/09/2025.
- Version:                 V1.0.0.
- Target processor:  Low-cost Axera AX630C.
- Python Version:     3.13.7 
- OS:                         MS Windows 11 Pro   24H2.

### Axera AX630C data [https://en.axera-tech.com/Product/126.html]

## 2) Introduction:
Since there are no neural network-based Keyword Spotting (KWS) algorithms that cover variants of the Spanish language such as the one spoken in Argentina, a series of Python scripts are designed to perform dataset generation tasks based on a single voice command recorded per class and a limited set of background audios with different environmental noises.

For this, a very simple neural network will be used that can read tensors with dimension **float32[1,1,98,64]**, within which there are spectrograms in log-Mel format. The meaning of the dimensions is:

```
[ batch , channel , time , frequency ]
[   1   ,    1    ,  98  ,     64    ]

Batch      = 1 → un ejemplo.
Channel    = 1 → “imagen” mono canal.
Time       = 98 → ~1 segundo en ventanas de 10 ms.
Frequency  = 64 → resolución espectral en escala Mel.
```

1. First 1 → batch size:  
It is the size of the input batch. In inference it is almost always set to 1 (processing only one example at a time). If trained with larger batches, the value here would be >1.  

2. Second 1 → channel (input channel):  
It is the number of input channels. In vision it would be something like 3 (RGB). In this case, since log-Mel spectrograms are treated as “single-channel images” → 1. Keeping the dimension allows 2D convolutions to work the same way as with images.  

3. Third value 98 → temporal dimension (frames):  
These are the frames in the time axis of the log-Mel. Each frame corresponds to a window of ~25 ms shifted every ~10 ms (according to win_ms and hop_ms). Thus, 98 frames ≈ 98 × 10 ms ≈ ~1 s of processed audio. This axis captures the temporal dynamics of the audio, which is consumed by the RNN (GRU) part of the model.  

4. Fourth value 64 → frequency dimension (Mel bands):  
It is the number of log-Mel coefficients extracted per frame. Each value represents the energy in a frequency band. 64 was chosen as the n_mels parameter when calculating the spectrogram. This axis represents the spectral information of the audio.  

## 3) Software Workflow Diagram:
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

## 4) Project directory and file structure:
```
kws_project/
  ├── .venv (Ambiente virtual con módulos de Python instalados)
  ├── data (Carpeta que contiene: audios seed, audios de calibración, audios generados con aumentación automática y tensores.)
  ├── src (Código fuente conteniendo los scripts escritos en Python.)
  ├── README.md (Archivo de ayuda en formato MarkDown.)
  └── requirements.txt (Archivo con los módulos de Python requeridos para que funcionen los scripts. Se instala con "pip install -r requirements.txt")
```

## 5) Steps and usage of the scripts:
### a) Record a .WAV file with each command and background noises:
Any voice recorder from any operating system can be used (e.g. Audacity, Adobe Premiere Pro, Windows Sound Recorder, etc.).  
As for background noises, it is necessary to evaluate which ones will prevail during the operation of the KWS. It is possible to obtain sounds from the Internet and convert them to .WAV with Audacity.  

It is advisable to generate a folder tree like this:
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

### b) Generate the dataset for training the neural network:
Since it is necessary to have between 1000 and 3000 audio samples of each command and it is complex to record each command 3000 times with different voices and background noises, we use an algorithm to expand our single sample stored in a .WAV file.  

The algorithm will generate variations: frequency, timing, time shifts, among others, and will add the different background noises; in the end, we will obtain 3000 samples of each command with different characteristics. These 3000 samples per command will serve to train our neural network and make it usable in a real environment.  

It is recommended to generate a folder called **seeds** containing the same audios as those stored in the **audios** folder. The idea is that **audios** acts as a backup and **seeds** is used to generate the dataset.

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

Variants of the **make_dataset.py** command:
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

Examples of use:  
With custom parameters:
```
python make_dataset.py --seeds-root D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\src\seeds --out-root D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\generado_automaticamente --per-class 3000 --fondo-minutes 20 --p-noise 0.9 --p-reverb 0.2 --p-stretch 0.4 --stretch-low 0.95 --stretch-high 1.05 --p-pitch 0.5 --pitch-semitones 1.0 --use-fondo-as-noise
```

With default parameters (THIS EXECUTION MODE IS USED):  
```
python make_dataset.py --seeds-root D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\seeds --noises-root D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\seeds\FONDO --out-root D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\generados_automaticamente
```

Scheme of the ./data directory after running the script:
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

In case only the **seeds** and **output** folders are specified, the script uses default values such as:  
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

Summary:  
- The parameters `p_*` control the probability of applying each transformation.  
- `use_fondo_as_noise` is especially useful since you already have audios with different backgrounds.  
```

### c) Use of **tensors_generator.py** to automatically generate **.pt** files from **.WAV**:
**tensors_generator.py** uses functions from **features.py** to generate .pt tensors based on .WAV audios containing spectrograms in log-Mel format with dimensions **float32[1,1,98,64]**.  
It has two modes:  
1. Generate a single tensor (.pt) from one .WAV.  
2. Generate a batch of tensors from multiple .WAV files.  

#### c.1) Generation of a single tensor from a given .WAV audio:
```
python tensors_generator.py D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\audios\REPETIR\000_repetir.wav --plot --db --out-root D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\tensores
```

#### c.2) Generation of batches of tensors from .WAV audios (THIS EXECUTION MODE IS USED):
```
python tensors_generator.py --generate-cache --wav-root D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\generados_automaticamente --out-root D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\tensores --sr 16000 --n_mels 64 --db --every 50
```

Output of the script:
```
✔ Guardado D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\tensores\SALIR\salir_02999.pt
[24600/24600] procesados
Listo: 24600/24600 archivos procesados.
```

## 6) Training the CRNN neural network for use in KWS with the **train.py** script:
To build and train the KWS model based on neural networks, the following scripts are used:
- dataset_from_cache.py: Dataloader responsible for reading the tensors saved in .pt files inside the ./tensores folder.
- model_crnn.py: Generator of a small CRNN (Conv2D + BiGRU + FC).
- train.py: Training loop with validation and checkpoints.
- export_onnx.py: Exporter of the trained model to ONNX (with dynamic T).

#### Usage of the scripts:
a) Use of **train.py**:
```
python train.py --cache-root D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\tensores\ --epochs 20 --batch-size 64 --n-mels 64 --ckpt-dir  D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\trained_tensor\ --ckpt-name kws_crnn_small.pt --early-stop --patience 5 --min-delta 1e-4
```

b) Use of **train.py** (THIS MODE IS USED, without --early-stop):
```
python train.py --cache-root D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\tensores\ --epochs 20 --batch-size 64 --n-mels 64 --ckpt-dir  D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\trained_tensor\ --ckpt-name kws_crnn_small.pt
```

- Uses the ./tensores/ directory by default to look for tensors (.pt files).
- Saves the best checkpoint as **kws_crnn_small.pt** in the script’s execution directory.
- Shows train/val metrics, the test, and the confusion matrix.

Variants of the **train.py** script:
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

Default values when only running the script:
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

#### The output when **train.py** is executed is the .pt tensor model and a confusion matrix:
Example output:
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

Recommendations:
- Let the epochs run (20 for example) and check if validation remains high.
- Test check: at the end, the script already evaluates on the test set with the best checkpoint (early stopping).
- Export to ONNX once satisfied with the test.

#### Example 1 – Same as now (without early stop):
> python train.py --cache-root ./tensores --ckpt-dir ./models --ckpt-name kws_crnn_small.pt

#### Example 2 – With early stop (patience 5, min_delta 1e-4):
> python train.py --cache-root ./tensores --ckpt-dir ./models --ckpt-name kws_crnn_small.pt --early-stop --patience 5 --min-delta 1e-4

## 7) ONNX model exporter **export_onnx.py**:
This script converts **kws_crnn.pt** into **kws_crnn.onxx**, that is, it converts the tensor model into a file in Open Neural Network Exchange (ONNX) format. In the end, it generates a **kws_crnn.onnx** file with configurable n-mel (number of Mel bands) and T (number of temporal frames).
By default n-Mel=64 and T=98.

Default values when only running the script:
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

Execution examples:
#### Exports to the current directory, with automatically generated name:
```
python export_onnx.py --ckpt-path ./kws_crnn_small.pt
```

#### Exports to a specific directory with default name:
```
python export_onnx.py --ckpt-path ./kws_crnn_small.pt --onnx-out-dir ./onnx_models/
```

#### Exports with a custom name in a directory (THIS ONE IS USED):
```
python .\export_onnx.py --ckpt-path D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\trained_tensor\kws_crnn_small.pt --onnx-out-dir D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\onnx_model --onnx-out-name kws_crnn_small.onnx
```

## 8) Calibration data generation for *.axmodel** with **make_calib.py** script:
This script generates calibration data so that Pulsar2 can:
- Properly quantize the model to INT8.  
- Be used only during compilation, not during inference.  
- Improve the accuracy of the quantized model by reflecting the real distributions of your data.  
- In the .axmodel, only the scales and offsets are stored, not the calibration files.  

Where are they used inside the .axmodel?  
- During the pulsar2 build process with --quant.*, Pulsar2:  
  - Reads the calibration samples (input_00000.npy, input_00001.npy …).  
  - Simulates the model’s forward pass in FP32 using those data.  
  - Adjusts the scales and offsets of each tensor (scale, zero_point) to store them in the .axmodel.  
  - Those scales are embedded in the .axmodel and the NPU runtime no longer needs the calibration data. You only use them during the compilation phase.  

Practical example for the KWS case:  
- In your keyword spotting model (KWS CRNN):  
  - The input is a Mel spectrogram [1,1,98,64].  
  - If you quantize without calibration, Pulsar2 might assume a generic range (e.g. -1.0 to 1.0) and saturate the energy of certain frequencies.  
  - With calibration, using real spectrograms, it adjusts the per-channel scale so that each Mel band uses the correct range → result:  
    - less prediction loss,  
    - accuracy closer to FP32,  
    - fewer false positives/negatives.  

Default values when only running the script:  
```
python.exe .\make_calib.py --pt-root D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\tensores --out-dir D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\calib_samples --calib-format "Numpy" --t 98 --n-mels 64 --max-samples 300 --seed 123 --tar-name "calib_kws.tar"
```

## 9) .bin file generator for direct testing on the M5Stack - AX630C:
This script is used to convert the calibration dataset into .bin files that can be directly understood by the **ax_run_model** command inside Ubuntu 22.04 on the M5Stack - AX630C.

Default values when only running the script:
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

## 10) Block Diagram of the expected CRNN (Convolutional) flow:
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

### Explanation of each neural network block for the KWS system to work:
- **Input (features [1,1,98,64]):**  
What it is: A Mel spectrogram (1 channel, 98 time steps, 64 Mel bands).  
Function: Represent the audio signal already transformed into spectral features.  

- **Conv2D #1:**  
What it does: Applies 2D convolutional filters over the spectrogram.  
Function: Detect local time-frequency patterns (e.g., formants, energy changes).  

- **ReLU:**  
What it does: Activation function that keeps positive values and discards negatives.  
Function: Add non-linearity so the network can learn more complex representations.  

- **MaxPool2D #1:**  
What it does: Reduces resolution by taking the maximum value in windows (e.g., 2×2).  
Function:  
  - Reduce dimensionality → fewer parameters.  
  - Preserve the most relevant features (energy peaks).  

- **Conv2D #2:**  
What it does: Another convolutional filter layer.  
Function: Extract more abstract representations (e.g., phoneme patterns, repetitive structures).  

- **ReLU + MaxPool2D #2:**  
Joint function: Same as before, but now acting on higher-level features.  
After this block, the spectrogram is reduced in time and frequency, but each “pixel” carries more semantic meaning.  

- **Reshape / Transpose:**  
What it does: Converts the convolutional output into a sequence [T, features].  
Function: Adapt the data to pass them to the RNN (GRU).  

- **GRU:**  
What it does: A recurrent network that processes the sequence step by step.  
Function: Capture temporal dependencies (e.g., how the spectrum evolves over time).  
It is key in KWS because words are distinguished not only by an instant but by how they change over time.  

- **GEMM / Fully Connected:**  
What it does: Takes the GRU output and projects it into an 8-dimensional space.  
Function: Generate the logits (unnormalized scores for each keyword).  

- **Softmax (implicit, outside ONNX):**  
What it does: Converts logits into probabilities (sums to 1).  
Function: Makes it easier to interpret “which word was spoken” and with what confidence.  

- **Output (logits [1,8]):**  
What it is: Vector with 8 values, one per class:  
{ACOPLAR, CANCELAR, CONTINUAR, FONDO, LEVANTADO, PRINCIPAL, REPETIR, SALIR}  
Function: Final classification of the detected keyword.  

## 11) Quick summary of script execution (from dataset generation to ONNX-to-AXMODEL conversion and loading on AX630C):
### All scripts are executed from the ./src folder of this project: 
> 1) python.exe .\make_dataset.py --seeds-root D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\seeds --noises-root D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\seeds\FONDO --out-root D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\generados_automaticamente

> 2) python.exe .\tensors_generator.py --generate-cache --wav-root D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\generados_automaticamente --out-root D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\tensores --sr 16000 --n_mels 64 --db --every 50

> 3) python.exe .\train.py --cache-root D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\tensores\ --epochs 20 --batch-size 64 --n-mels 64 --ckpt-dir  D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\trained_tensor\ --ckpt-name kws_crnn_small.pt

> 4) python.exe .\export_onnx.py --ckpt-path D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\trained_tensor\kws_crnn_small.pt --onnx-out-dir D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\onnx_model --onnx-out-name kws_crnn_small.onnx

> 5) python.exe .\make_calib.py --pt-root D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\tensores --out-dir D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\calib_samples --calib-format "Numpy" --t 98 --n-mels 64 --max-samples 300 --seed 123 --tar-name "calib_kws.tar"

> 6) python.exe .\generate_test_files.py --src D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\calib_samples\calib_kws.tar --out-dir D:\04_ProyASRC\VSCode\20250826_M5Stack_LLM-CoreS3\20250921_kws_project\data\binary_for_npu\ 

### Starting from step 6 you must switch to the **Pulsar2** environment to convert the generated **.onnx** model into the **.axmodel** model required for the Axera AX630C or AX650N AI chip to understand the KWS model.

### Follow installation and configuration instructions for Pulsar2 with Docker:

#### Link with installation steps for Pulsar2: [https://pulsar2-docs.readthedocs.io/en/latest/user_guides_quick/quick_start_prepare.html]

#### Copy the **.onnx** model, produced during training, into Ubuntu 22.04 on the M5Stack LLM Kit. To do this, connect via **ssh** and use the SCP command from within the module:
> scp "User@192.168.###.###:/./data/output/compiled.axmodel" /root/kws_int8.axmodel

#### Copy the calibration files into the **/root** directory inside Ubuntu 22.04 of the M5Stack LLM Kit. To do this, we must connect to it via **ssh** and use the SCP command from within the module:
> scp -r User@192.168.###.###:/D:/04_ProyASRC/VSCode/20250826_M5Stack_LLM-CoreS3/20250921_kws_project/data/binary_for_npu/ /root

It has been tested running the **ax_run_model** command with more than one input, without success. For this reason, although **generate_test_files.py** generates 300 inputs for calibration, when executing the following command:
> ax_run_model -m kws_int8.axmodel -i ./binary_for_npu/input/ -o ./binary_for_npu/output/ -l ./binary_for_npu/list.txt

With a **list.txt** containing 0 to 299 inputs, it does not work, giving the error **} does not has input folder {./binary_for_npu/input/}, skipped.**:
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

There is no way to run more than one input with **ax_run_model**. All possible parameters have been tested, but none of them work. Therefore, **list.txt** must contain only one input, that is, only a 0 on the first line without spaces or line breaks.

#### There are two ways to convert the model from **.onnx** to **.axmodel**: one is by typing the full command in Pulsar2 inside Docker, and the other (much easier) is by calling a configuration file (we will use this method): 

Copy the configuration file **kws_build_int8_config.json** into the folder **/data/config/kws_build_int8_config.json** inside the Pulsar2 execution environment:
> cp kws_build_int8_config.json ./data/config/kws_build_int8_config.json

#### Launch the compilation process inside Pulsar2 in Docker, which will convert the **.onnx** model to **.axmodel**:
> pulsar2 build --config /data/config/kws_build_int8_config.json

#### You will see something similar to the following output:
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

#### If you do not see something like this in the output and instead an error appears, you should verify that steps 1 to 6 were performed correctly and then check the **kws_build_int8_config.json** file.

#### If the conversion to **.axmodel** is successful, it will be possible to execute the following command inside the M5Stack LLM Kit with AX630C to verify that the KWS model based on neural networks runs correctly:
> root@m5stack-LLM:~# ax_run_model -m /root/kws_int8.axmodel -w 10 -r 100

- -w 10 = warmup of 10 samples for stabilization.  
- -r 100 = Runs 100 real inferences of the kws_int8.axmodel model.  

#### The following output will appear:
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

#### Some points to keep in mind:
- Although it shows **type: Half** it should display INT8. It has not been possible to make it show this quantization type for the KWS model, likely due to the **ax_run_model** frontend.  
- The output indicates **vnpu: Disable**, which is inconsistent because, on one hand, it was configured to run on NPU1, and on the other hand, the inference times are P5=1.062ms, P90=1.057ms, P95=1.057ms, and P99=1.056ms. This suggests that it is not using the Dual Cortex A53 CPUs up to 1.2GHz.  
- Another issue observed is that possibly the labels P5, P90, P95, and P99 are inverted, since P99 can never be faster than P5. However, the values are stable and coherent.  

#### Stability and inference times verified with 5000 samples:
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

#### It is verified that for 5000 samples stability is maintained just as with 100 samples.

## 12) Execution of **ax_run_model** with a file containing a real spectrogram sample converted into **.bin**:
#### Copy the calibration files into the **/root** directory inside Ubuntu 22.04 of the M5Stack LLM Kit. To do this, we must connect to it via **ssh** and use the SCP command from within the module:
> scp -r User@192.168.###.###:/D:/04_ProyASRC/VSCode/20250826_M5Stack_LLM-CoreS3/20250921_kws_project/data/binary_for_npu/ /root

#### Structure of **list.txt** and **./input** directory:
Contents of **input.txt**:
```
0
```

Input directory structure (note that it is only possible to use the first file inside **./input/0/input.bin**, the rest of the directories are not found by **ax_run_model**):
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

It has been tested running the **ax_run_model** command with more than one input, without success. For this reason, although **generate_test_files.py** generates 300 inputs (0 to 299) for calibration, when executing the following command:
> ax_run_model -m kws_int8.axmodel -i ./binary_for_npu/input/ -o ./binary_for_npu/output/ -l ./binary_for_npu/list.txt

With a **list.txt** containing 0 to 299 inputs, it does not work, giving the error **} does not has input folder {./binary_for_npu/input/}, skipped.**:
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

It can be seen that out of 300 **input.bin** files in each of the input directories, only one file is actually inferred. The directory created in **./binary_for_npu/output** corresponds to the last one, in this case number **299**. Inside it is the file **output.bin** with a size of 32 bytes containing the probability of each of the 8 classes, 4 bytes each, the total sum of the 8 class probabilities being 1 or 100%.  

There is no way to execute more than one input with **ax_run_model**. All possible parameters have been tested, none of them work. Therefore, **list.txt** must contain only one input, i.e., only a 0 on the first line without spaces or line breaks.  

The rest of the **input.bin** files with spectrograms Mel=64 and T=98 are not executed, due to a problem in **ax_run_model**.  

Nevertheless, this is useful to test a real file and, in this way, truly measure inference times with a real-world example. This confirms that it correctly reads the **.bin** file with real data and that in the ./binary_for_npu/output directory…

## 13) Script to analyze the **output.bin** file and indicate the probability inferred by **ax_run_model**:
This script is used to visualize the probabilities generated by the inference of **.bin** files containing log-Mel spectrograms. It runs on the PC, and for this, the file must first be copied from the M5Stack AX630C to the PC. Then the script is executed, and it displays the probabilities detected by the trained KWS model for each of the 8 classes.

The hexadecimal values inside **output.bin** are as follows:

- [0] 94 34 3D 41 = 11.825336456298828
- [1] FA 0F 2B C0 = -2.6728501319885254
- [2] 60 D1 78 3E = 0.24298620223999023
- [3] 04 3F B0 BF = -1.3769230842590332
- [4] C6 C6 06 C0 = -2.105882167816162
- [5] D4 24 91 C0 = -4.535745620727539
- [6] 20 FB C4 BF = -1.5389137268066406
- [7] 70 D1 78 BF = -0.9719457626342773

To calculate the probability of each class, the **Softmax** function must be applied:
> Softmax:  P(i) = exp(x_i) / (exp(x_1) + exp(x_2) + ... + exp(x_N))

Where:
- **x_i** is the logit of class i.
- **N** is the total number of classes.
- **P(i)** is the probability of class i.

Probabilities:

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

In the end, the word detected by the KWS is **ACOPLAR**.

## 14) Safety Industrial Levels of Pulsar:
Pulsar2 has undergone functional safety evaluation in accordance with ISO-26262 Part 8.
It has been classified with Tool Confidence Level 3 (TCL3), supported by complete test coverage and anomaly handling procedures.
The tool has over 2 years of deployment in multiple customer projects and has passed third-party safety certification.
When used as instructed, Pulsar2 can be applied to the development of models with safety requirements up to ASIL-D.
This ensures its suitability for safety-critical automotive and industrial applications.

## 15) Disclosure:
Axera, AX630C, AX650N, M5Stack, Docker, Pulsar2, Audacity, Adobe Premiere Pro, Microsoft, Windows 11 and Windows Sound Recorder are trademarks or registered trademarks of their respective owners. All product names, logos, and brands mentioned are for identification purposes only. We make no claim of ownership over these marks, and we are not affiliated with, endorsed by, or responsible for them in any way.  

Furthermore, we do not assume any responsibility or liability for issues, damages, or consequences that may arise from the execution, use, or misinterpretation of the algorithms, examples, or procedures provided. All content is for informational and educational purposes only, and any implementation is at the user’s sole risk.
