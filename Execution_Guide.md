## Quick summary of script execution (from dataset generation to ONNX-to-AXMODEL conversion and loading on AX630C):
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

#### There are two ways to convert the model from **.onnx** to **.axmodel**: one is by typing the full command in Pulsar2 inside Docker, and the other (much easier) is by calling a configuration file (we will use this method): 

Copy the configuration file **kws_build_int8_config.json** into the folder **/data/config/kws_build_int8_config.json** inside the Pulsar2 execution environment:
> cp kws_build_int8_config.json ./data/config/kws_build_int8_config.json

#### Launch the compilation process inside Pulsar2 in Docker, which will convert the **.onnx** model to **.axmodel**:
> pulsar2 build --config /data/config/kws_build_int8_config.json

#### If you do not see something like this in the output and instead an error appears, you should verify that steps 1 to 6 were performed correctly and then check the **kws_build_int8_config.json** file.

#### If the conversion to **.axmodel** is successful, it will be possible to execute the following command inside the M5Stack LLM Kit with AX630C to verify that the KWS model based on neural networks runs correctly:
> root@m5stack-LLM:~# ax_run_model -m /root/kws_int8.axmodel -w 10 -r 100

- -w 10 = warmup of 10 samples for stabilization.  
- -r 100 = Runs 100 real inferences of the kws_int8.axmodel model.  

#### Some points to keep in mind:
- Although it shows **type: Half** it should display INT8. It has not been possible to make it show this quantization type for the KWS model, likely due to the **ax_run_model** frontend.  
- The output indicates **vnpu: Disable**, which is inconsistent because, on one hand, it was configured to run on NPU1, and on the other hand, the inference times are P5=1.062ms, P90=1.057ms, P95=1.057ms, and P99=1.056ms. This suggests that it is not using the Dual Cortex A53 CPUs up to 1.2GHz.  
- Another issue observed is that possibly the labels P5, P90, P95, and P99 are inverted, since P99 can never be faster than P5. However, the values are stable and coherent.  

#### It is verified that for 5000 samples stability is maintained just as with 100 samples.

## 12) Execution of **ax_run_model** with a file containing a real spectrogram sample converted into **.bin**:
#### Copy the calibration files into the **/root** directory inside Ubuntu 22.04 of the M5Stack LLM Kit. To do this, we must connect to it via **ssh** and use the SCP command from within the module:
> scp -r User@192.168.###.###:/D:/04_ProyASRC/VSCode/20250826_M5Stack_LLM-CoreS3/20250921_kws_project/data/binary_for_npu/ /root

It has been tested running the **ax_run_model** command with more than one input, without success. For this reason, although **generate_test_files.py** generates 300 inputs (0 to 299) for calibration, when executing the following command:
> ax_run_model -m kws_int8.axmodel -i ./binary_for_npu/input/ -o ./binary_for_npu/output/ -l ./binary_for_npu/list.txt

## 13) Script to analyze the **output.bin** file and indicate the probability inferred by **ax_run_model**:
Probabilities:

- [0] ACOPLAR = 0.999983
- [1] CANCELAR = 0.000001
- [2] CONTINUAR = 0.000009
- [3] FONDO = 0.000002
- [4] LEVANTADO = 0.000001
- [5] PRINCIPAL = 0.000000
- [6] REPETIR = 0.000002
- [7] SALIR = 0.000003
