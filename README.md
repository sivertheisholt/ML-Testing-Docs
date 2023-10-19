# Object-Detection-WSL2

Repository for setting up WSL2 for training object detection model. This repo is based on Ubuntu 22.04 LTS.

## Enable NVIDIA CUDA

Follow the guide provided here to get GPU ready on WSL2:

https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl

## Install tensorflow

If the GPU does not show up in the list you can try installing CUDNN:

https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html

## WSL2 train object detection model

This repo contains the necessary instruction to setup and start a training with tensorboard for visualization. 

First you will have to prepare some images (labeled) and drop the zip file in the root directory. Name the file images.zip.

### Configuration

No file for configuration changes are implemented so you will have to edit start_training.sh.

Replace with the classes you need:

```
cat <<EOF > ./labelmap.txt
class1
class2
class3
person
EOF
```

Replace with model you wanna use, must be one in the MODELS_CONFIG varibable:

```
chosen_model='ssd-mobilenet-v2-fpnlite-320'
```

### Start training

Start the training by running the following script:

```
start_training.sh
```

Remember to: 

```
chmod +x start_training.sh
```

When the training starts you can access the tensorboard over at (Tt takes a few minutes of training to show data):

```
http://localhost:6006
```

To make sure the gpu is being utilized you can run the following and check the GPU Utilization:

```
nvidia-smi
```

## Notebook train object detection model

This collab has everything needed to train using your own data:

https://colab.research.google.com/github/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Train_TFLite2_Object_Detction_Model.ipynb#scrollTo=6V7TrfUos-9E

NB: THE NOTEBOOK RAN OUT OF RESOURCES AFTER ABOUT 5 HOURS.
