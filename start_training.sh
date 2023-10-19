# CONFIGURATION VARIABLES

train_record_fname='./train.tfrecord'
val_record_fname='./val.tfrecord'
label_map_pbtxt_fname='./labelmap.pbtxt'

chosen_model='ssd-mobilenet-v2-fpnlite-320'

MODELS_CONFIG='{
    "ssd-mobilenet-v2": {
        "model_name": "ssd_mobilenet_v2_320x320_coco17_tpu-8",
        "base_pipeline_file": "ssd_mobilenet_v2_320x320_coco17_tpu-8.config",
        "pretrained_checkpoint": "ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
    },
    "efficientdet-d0": {
        "model_name": "efficientdet_d0_coco17_tpu-32",
        "base_pipeline_file": "ssd_efficientdet_d0_512x512_coco17_tpu-8.config",
        "pretrained_checkpoint": "efficientdet_d0_coco17_tpu-32.tar.gz"
    },
    "ssd-mobilenet-v2-fpnlite-320": {
        "model_name": "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8",
        "base_pipeline_file": "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config",
        "pretrained_checkpoint": "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz"
    }
}'

model_name=$(python3 -c "import json, sys; data = json.load(sys.stdin); print(data['$chosen_model']['model_name'])" <<< "$MODELS_CONFIG")
pretrained_checkpoint=$(python3 -c "import json, sys; data = json.load(sys.stdin); print(data['$chosen_model']['pretrained_checkpoint'])" <<< "$MODELS_CONFIG")
base_pipeline_file=$(python3 -c "import json, sys; data = json.load(sys.stdin); print(data['$chosen_model']['base_pipeline_file'])" <<< "$MODELS_CONFIG")

num_steps=40000
batch_size=16

pipeline_fname='./models/mymodel/'"$base_pipeline_file"
fine_tune_checkpoint='./models/mymodel/'"$model_name""/checkpoint/ckpt-0"

# Clone the tensorflow models repository from GitHub
git clone --depth 1 https://github.com/tensorflow/models

# Copy setup files into models/research folder
cd ./models/research/
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .

cd ../..

# Install the Object Detection API

pip install pyyaml==5.3

pip install ./models/research/

# Run Model Bulider Test file, just to verify everything's working properly

python3 ./models/research/object_detection/builders/model_builder_tf2_test.py

# Split images into train, validation, and test folders

mkdir images
mkdir images/all
mkdir images/train
mkdir images/validation
mkdir images/test

unzip -q ./images.zip -d ./images/all

python3 ./train_val_test_split.py

# This creates a a "labelmap.txt" file with a list of classes the object detection model will detect.

cat <<EOF > ./labelmap.txt
class1
class2
class3
EOF

# Create CSV data files and TFRecord files
python3 ./create_csv.py
python3 ./create_tfrecord.py --csv_input=images/train_labels.csv --labelmap=labelmap.txt --image_dir=images/train --output_path=train.tfrecord
python3 ./create_tfrecord.py --csv_input=images/validation_labels.csv --labelmap=labelmap.txt --image_dir=images/validation --output_path=val.tfrecord

# Create "mymodel" folder for holding pre-trained weights and configuration files
mkdir ./models/mymodel

cd ./models/mymodel

# Download pre-trained model weights
download_tar='http://download.tensorflow.org/models/object_detection/tf2/20200711/'"$pretrained_checkpoint"
wget --timestamping $download_tar
tar -xzf "$pretrained_checkpoint"

# Download training configuration file for model
download_config='https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/configs/tf2/'"$base_pipeline_file"
wget --timestamping $download_config

cd ../..

# Get number of classes

num_classes=$(python3 ./get_num_classes.py)

# Create custom configuration file by writing the dataset, model checkpoint, and training parameters into the base pipeline file
python3 ./create_custom_config_file.py \
    --pipeline_fname "$pipeline_fname" \
    --fine_tune_checkpoint "$fine_tune_checkpoint" \
    --train_record_fname "$train_record_fname" \
    --val_record_fname "$val_record_fname" \
    --label_map_pbtxt_fname "$label_map_pbtxt_fname" \
    --batch_size $batch_size \
    --num_steps $num_steps \
    --num_classes $num_classes \
    --chosen_model "$chosen_model"

# Display the custom configuration file's contents

cat ./models/mymodel/pipeline_file.config

# Set the path to the custom config file and the directory to store training checkpoints in

mkdir ./training

pipeline_file='./models/mymodel/pipeline_file.config'
model_dir='./training/'

tensorboard --logdir './training/train' &

python3 ./models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path=$pipeline_file \
    --model_dir=$model_dir \
    --alsologtostderr \
    --num_train_steps=$num_steps \
    --sample_1_of_n_eval_examples=1