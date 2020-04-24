
import glob
import json
import logging
import os
import sys
import time

# Ours
from get_arguments import get_arguments
from tfrecord_data import input_fn
from hybrid_model  import model_fn

import tensorflow.compat.v1 as tf
from tensorflow.python import ipu

from common_classes import validate_arguments, CommonRunConfig, CommonEstimator

#                               # parameterize these 
DROPOUT = 0.20
SHUFFLE_BUFFER = 1500
NBR_CLASSES = 2

def logger(prefix):
    """ """
    logging.getLogger().setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)-2s - %(levelname)-2s - %(message)s', "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(prefix + 'attn_bin_estimator.log')
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(logging.INFO)

    logging.getLogger().addHandler(fh)
    logging.getLogger().addHandler(ch)
    return logging.getLogger(__name__)


def qualify_path(directory):
    """Generate fully qualified path name from input file name."""
    return os.path.abspath(directory)


def ordered_value_list(as_dict):
    class_weights = []
    int_dict = {int(k): v for (k, v) in as_dict.items()}
    for k in sorted(int_dict):
        class_weights.append(int_dict[k])
    return class_weights


def create_estimator(params=None, steps=None):
    """ """
    ipu_options = ipu.utils.create_ipu_config()
    ipu.utils.auto_select_ipus(ipu_options, num_ipus=1)

    ipu_run_config = ipu.ipu_run_config.IPURunConfig(
        iterations_per_loop=params['iterations_per_loop'],
        ipu_options=ipu_options
    )

    config = ipu.ipu_run_config.RunConfig(
        ipu_run_config=ipu_run_config,
        log_step_count_steps=1000,              # differs from GC example
        save_summary_steps=steps,               # differs from GC example
        model_dir=params['model_dir']
    )

    estimator = ipu.ipu_estimator.IPUEstimator(
        model_fn=params['model_fn'],
        model_dir=params['model_dir'],
        config=config,
        params=params,
    )

    return estimator

def main(args):
    """ """
    # fname qualifier for inputs
    file_prefix = args.inpfx
    if file_prefix:
        file_prefix = file_prefix + '.'
    logger(file_prefix)

    epochs = args.epochs
    data_dir = qualify_path(args.data_dir)
    model_dir = qualify_path(args.model_dir)
    batch_size = args.batch_size

    metadata_file = os.path.join(data_dir, 'summary.json')
    #metadata_file = os.path.join(data_dir, file_prefix + 'tfrecords-metadata.json')
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    eval_steps = metadata['val_examples'] // batch_size
    test_steps = metadata['test_examples'] // batch_size
    epoch_steps = test_steps
    train_steps = metadata['train_examples'] // batch_size * epochs
    output_bias = metadata['output_bias']
    class_weights = ordered_value_list(metadata['class_weight'])

    params = {}
    params['epochs'] = epochs
    params['data_dir'] = data_dir
    params['model_dir'] = model_dir
    params['model_fn'] = model_fn
    params['batch_size'] = batch_size
    params['file_prefix'] = file_prefix
    params['train_steps'] = train_steps
    params['eval_steps'] = eval_steps
    params['output_bias'] = output_bias
    params['class_weights'] = class_weights
    params['iterations_per_loop'] = int(train_steps/epochs)

    params['mode'] = args.mode
    params['learning_rate'] = args.learning_rate
    params['log_frequency'] = args.log_frequency

    params['shuffle_buffer'] = SHUFFLE_BUFFER
    params['nbr_classes'] = NBR_CLASSES
    params['dropout'] = DROPOUT
    params['input_sizes'] = (942, 5270)

    print("*" * 130)
    print(f"Batch size is {batch_size}")
    print(f"Number of epochs: {epochs}")
    print(f"Model directory: {model_dir}")
    print(f"Data directory: {data_dir}")
    print("params:", params)
    print("args:", args)
    print("*" * 130)

    # establish common Estimator and associated Config classes
    partition = None
    if 'predict' in args.mode:
        partition = 'test'
        steps = test_steps
    if 'eval' in args.mode:
        partition = 'val'
        steps = eval_steps
    if 'train' in args.mode:
        partition = 'train'
        steps = train_steps
    params['partition'] = partition

    model = create_estimator(params, steps)

    # predict
    if 'predict' in args.mode:
        print("\nPredicting...")
        predict_not_impl = "PREDICT mode not yet implemented"
        tf.logging.error(predict_not_impl)
        assert False, predict_not_impl

    # train
    if 'train' in args.mode:
        print(f"epochs: {epochs} via train steps: {train_steps} steps per epoch {epoch_steps}")
        model.train(input_fn, steps=train_steps)
        print("Training complete")

    # evaluate 
    if 'eval' in args.mode:
        print("\nEvaluating...")
        eval_result = model.evaluate(input_fn, steps=eval_steps)

        print("global step:%7d" % eval_result['global_step'])
        print("accuracy:   %7.2f" % round(eval_result['accuracy'] * 100.0, 2))
        print("loss:       %7.2f" % round(eval_result['loss'], 2))
        print("Evaluation complete")

##______________________________________________________________________________
if __name__ == '__main__':
    arguments = get_arguments()
    main(arguments)
