import re
import argparse

def create_custom_config_file(pipeline_fname, fine_tune_checkpoint, train_record_fname, val_record_fname, label_map_pbtxt_fname, batch_size, num_steps, num_classes, chosen_model):
    with open(pipeline_fname) as f:
        s = f.read()
    with open('./models/mymodel/pipeline_file.config', 'w') as f:

        # Set fine_tune_checkpoint path
        s = re.sub('fine_tune_checkpoint: ".*?"',
                'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), s)

        # Set tfrecord files for train and test datasets
        s = re.sub(
            '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")', 'input_path: "{}"'.format(train_record_fname), s)
        s = re.sub(
            '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")', 'input_path: "{}"'.format(val_record_fname), s)

        # Set label_map_path
        s = re.sub(
            'label_map_path: ".*?"', 'label_map_path: "{}"'.format(label_map_pbtxt_fname), s)

        # Set batch_size
        s = re.sub('batch_size: [0-9]+',
                'batch_size: {}'.format(batch_size), s)

        # Set training steps, num_steps
        s = re.sub('num_steps: [0-9]+',
                'num_steps: {}'.format(num_steps), s)

        # Set number of classes num_classes
        s = re.sub('num_classes: [0-9]+',
                'num_classes: {}'.format(num_classes), s)

        # Change fine-tune checkpoint type from "classification" to "detection"
        s = re.sub(
            'fine_tune_checkpoint_type: "classification"', 'fine_tune_checkpoint_type: "{}"'.format('detection'), s)

        # If using ssd-mobilenet-v2, reduce learning rate (because it's too high in the default config file)
        if chosen_model == 'ssd-mobilenet-v2':
            s = re.sub('learning_rate_base: .8',
                        'learning_rate_base: .08', s)

            s = re.sub('warmup_learning_rate: 0.13333',
                        'warmup_learning_rate: .026666', s)

        # If using efficientdet-d0, use fixed_shape_resizer instead of keep_aspect_ratio_resizer (because it isn't supported by TFLite)
        if chosen_model == 'efficientdet-d0':
            s = re.sub('keep_aspect_ratio_resizer', 'fixed_shape_resizer', s)
            s = re.sub('pad_to_max_dimension: true', '', s)
            s = re.sub('min_dimension', 'height', s)
            s = re.sub('max_dimension', 'width', s)

        f.write(s)
        
parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--pipeline_fname', type=str, help='Path to pipeline config file')
parser.add_argument('--fine_tune_checkpoint', type=str, help='Path to fine-tune checkpoint')
parser.add_argument('--train_record_fname', type=str, help='Path to train record file')
parser.add_argument('--val_record_fname', type=str, help='Path to validation record file')
parser.add_argument('--label_map_pbtxt_fname', type=str, help='Path to label map file')
parser.add_argument('--batch_size', type=int, help='Batch size')
parser.add_argument('--num_steps', type=int, help='Number of training steps')
parser.add_argument('--num_classes', type=int, help='Number of classes')
parser.add_argument('--chosen_model', type=str, help='Chosen model')

args = parser.parse_args()
print(args.pipeline_fname)

create_custom_config_file(args.pipeline_fname, args.fine_tune_checkpoint, args.train_record_fname, args.val_record_fname, args.label_map_pbtxt_fname, args.batch_size, args.num_steps, args.num_classes, args.chosen_model)
