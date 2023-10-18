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

Instead you can follow the guide under. Make sure you have setup WSL2 with GPU support (Look at the previous header).

# WSL2 Train object dection

This repo contains the necessary instruction to setup and start a training with tensorboard for visualization. 

First you will have to prepare some images (labeled) and drop the zip file in the root directory.

```
start_training.sh
```

Remember to: 

```
chmod +x start_training.sh
```

When the training starts you can access the tensorboard over at (Remember it takes a few minutes of training to show data):

```
http://localhost:6006
```

To make sure the gpu is being utilized you can run the following and check the GPU Utilization:

```
nvidia-smi
```
