#%% Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import unet_model_classifcation as unet_model

#%%
def create_unet_model(data, num_classes, conv_ksize, num_channels_base,
                      scale_depth, learning_rate, bn_decay, model_dir_name):

    (_, data_height, data_width, num_in_channels) = data.shape
    #tf.reset_default_graph()
    model_params = {"data_height" : data_height,
                    "data_width" : data_width,
                    "num_in_channels" : num_in_channels,
                    "num_classes" : num_classes,
                    "conv_ksize" : conv_ksize,
                    "num_channels_base" : num_channels_base,
                    "scale_depth" : scale_depth,
                    "learning_rate" : learning_rate,
                    "bn_decay" : bn_decay
                    }
    classifier = tf.estimator.Estimator(
            model_fn = unet_model.model_fn, params = model_params,
            model_dir = model_dir_name)

    return classifier


def train_unet_model(data, labels, classifier, batch_size, training_steps,
                     log_interval):

    (b, h, w, c) = data.shape
    data_flat = np.reshape(data, (b, h * w, c)).astype(np.float32)
    print('labels shape:', labels.shape)
    labels_flat = np.reshape(labels, (-1, 1)).astype(np.float32)
    print('flat labels shape:', labels_flat.shape)


    # Log the training
    tensors_to_log = {"loss" : "pw_loss", "loss_norm": "pw_loss_normalized"}
    logging_hook = tf.train.LoggingTensorHook(
            tensors = tensors_to_log, every_n_iter = log_interval)

    # Create input function for estimator
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x" : data_flat},
        y = labels_flat,
        batch_size = batch_size,
        num_epochs = None,
        shuffle = True)

    # Actual training
    tf.logging.set_verbosity(tf.logging.INFO)
    c1 = classifier.train(
            input_fn=train_input_fn,
            steps = training_steps,
            hooks = [logging_hook])

    return c1

def predict_unet_model(data, classifier):
    # c = color channels (?) = 1
    (nr_images, h, w, c) = data.shape
    data_flat = np.reshape(data, (nr_images, h * w, c)).astype(np.float32)
    predictions = np.zeros((nr_images, 1))
    for i in range(0, nr_images):
        print("Predicting: " + str(i) + " of " + str(nr_images))
        image = np.expand_dims(data_flat[i,:,:], axis = 0)
        print(image.shape)
        prediction_fun = tf.estimator.inputs.numpy_input_fn(
                x={"x": image},
                num_epochs=1,
                shuffle=False)

        pred = list(classifier.predict(input_fn = prediction_fun))
        predictions[i,:] = pred[0]['classes']
    return predictions


def compute_accuracy(labels, predictions):
    correct = np.equal(labels, predictions)
    acc = sum(correct)/len(correct)
    return acc
