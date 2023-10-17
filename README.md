# ML-Testing-Docs
Repository for me to put my spaghetti docs when testing out ML stuff

# Enable NVIDIA CUDA on WSL2

Follow the guide provided here to get GPU ready on WSL2:

https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl

# Install tensorflow on WSL2

If the GPU does not show up in the list you can try installing CUDNN:

https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html


# Train object detection model:

This collab has everything needed to train using your own data:

https://colab.research.google.com/github/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Train_TFLite2_Object_Detction_Model.ipynb#scrollTo=6V7TrfUos-9E

NB: IT WILL RUN OUT OF RESOURCES IF USING THE RUNTIME.

Instead you can follow the code blocks and use them in ur WSL2 distro (Preferably Ubuntu). Make sure you have setup WSL2 with GPU support (Look at the previous header). Some changes are required, as the link above is meant to run in the collab notebook. The principle is still the same, just take the code out, modify it to work in your environment and run in the same order.

Once everything is setup you can use the final code block to start training:

```
python3 /home/sivertheisholt/models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path=/home/sivertheisholt/models/mymodel/pipeline_file.config \
    --model_dir=/home/sivertheisholt/training \
    --alsologtostderr \
    --num_train_steps=40000 \
    --sample_1_of_n_eval_examples=1
```

To make sure the gpu is being utilized you can run the following and check the GPU Utilization:

```
nvidia-smi
```
