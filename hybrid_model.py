import numpy as np
import tensorflow.compat.v1 as tf
from keras_model import build_model
import estimator_hooks as hooks
import pdb

def model_fn(features, labels, mode, params):
    """ """
    LR = params['learning_rate']
    batch_size = params['batch_size']
    nbr_classes = params['nbr_classes']

    class_weights = params['class_weights']
    class_weights = np.array(class_weights)
    class_weights = np.tile(class_weights, (batch_size, 1 ))

    # from Cerebras MNIST hybrid_model.py
    # tf.set_random_seed(0)       # --seed arg not yet implemented
    loss = None
    train_op = None
    logging_hook = None
    training_hook = None
    eval_metric_ops = None
    logging_op = None

    # living in the past?
    get_or_create_global_step_fn = tf.train.get_or_create_global_step
    get_global_step_fn = tf.train.get_global_step
    get_collection_fn = tf.get_collection
    set_verbosity_fn = tf.logging.set_verbosity
    optimizer_fn = tf.train.MomentumOptimizer
    accuracy_fn = tf.metrics.accuracy
    loss_fn = tf.losses.sparse_softmax_cross_entropy		# see loss_fn below

    logging_INFO = tf.logging.INFO
    GraphKeys = tf.GraphKeys
    summary_scalar = tf.summary.scalar

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    is_evaluate = (mode == tf.estimator.ModeKeys.EVAL)
    is_predict  = (mode == tf.estimator.ModeKeys.PREDICT)

    inputs = features #### changed to singleton return RRT 3/11 ['data']
    keras_model = build_model(params, tensor=inputs)
    logits = keras_model.output
    predictions = tf.argmax(logits, 1)

    global_step = get_or_create_global_step_fn()
    loss = loss_fn(labels=labels, logits=logits)
   #pdb.set_trace()
   #loss = loss_fn(labels=labels, logits=logits, weights=class_weights)
    hook_list = []

    accuracy = accuracy_fn(
        labels=labels,
        predictions=predictions,
        name='accuracy_op')

    eval_metric_ops = dict(accuracy=accuracy)
    summary_scalar('accuracy', accuracy[1])
    set_verbosity_fn(logging_INFO)
    #GC
    """
    logging_hook = tf.estimator.LoggingTensorHook(
        {"loss": loss, "accuracy": accuracy[1]},
        every_n_iter = 1000) #### every_n_secs = 60)
    hook_list.append(logging_hook)
    """
    #end GC
    if is_training:
        optimizer = optimizer_fn(learning_rate=LR, momentum=0.9)
        update_ops = get_collection_fn(GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss)
        #GC
        training_hook = hooks.TrainingHook(params)
        logging_op = training_hook.enqueue({"loss": loss, "accuracy": accuracy[1]})  #Add loss to outfeed
        training_op = tf.group([train_op, logging_op])      #Ensure logging_op is attached to graph?
        #end GC
        hook_list.append(training_hook)

    estimator = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=training_op,
        eval_metric_ops=eval_metric_ops,
        training_hooks=hook_list)

    return estimator

