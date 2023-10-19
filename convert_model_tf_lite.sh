output_directory='./custom_model_lite'
last_model_path='./training'
pipeline_file='./models/mymodel/pipeline_file.config'

mkdir ./custom_model_lite

python3 ./models/research/object_detection/export_tflite_graph_tf2.py \
    --trained_checkpoint_dir $last_model_path \
    --output_directory $output_directory \
    --pipeline_config_path $pipeline_file

python3 ./convert_graph_tf_lite.py