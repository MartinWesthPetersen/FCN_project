#%% Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import unet_model_bn as unet_model

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
    labels_flat = np.reshape(labels, (b, h * w)).astype(np.float32)
    
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
    (b, h, w, c) = data.shape
    data_flat = np.reshape(data, (b, h * w, c)).astype(np.float32)
    predictions = np.zeros((b, h, w))
    for i in range(0, b):
        print("Predicting: " + str(i) + " of " + str(b))
        image = np.expand_dims(data_flat[i,:,:], axis = 0)
        prediction_fun = tf.estimator.inputs.numpy_input_fn(
                x = {"x" : image},
                num_epochs = 1,
                shuffle = False)
        
        pred = list(classifier.predict(input_fn = prediction_fun))
        predictions[i,:,:] = pred[0]['classes']
    return predictions

def accuracy(pred, true):
    accs = []
    for i in range(pred.shape[0]):
        accs.append(np.sum(pred[i,:,:] == true[i,:,:]) / 
                    (pred.shape[1] * pred.shape[2]))
    return accs


def mean_dice(pred, true, num_classes):
    D = []
    for i in range(0, num_classes):
        pos_true = true == i
        T = np.sum(pos_true)
        pos_pred = pred == i
        P = np.sum(pos_pred)
        if (P == 0 and T == 0):
            continue
        TP = np.sum(np.logical_and(pos_true, pos_pred))
        #FP = np.sum(np.logical_and(np.logical_not(pos_true), pos_pred))
        D.append((2 * TP) / (P + T))
    return sum(D) / len(D)

def mean_dice_no_background(pred, true, num_classes):
    D = []
    for i in range(1, num_classes):
        pos_true = true == i
        T = np.sum(pos_true)
        pos_pred = pred == i
        P = np.sum(pos_pred)
        if (P == 0 and T == 0):
            continue
        TP = np.sum(np.logical_and(pos_true, pos_pred))
        #FP = np.sum(np.logical_and(np.logical_not(pos_true), pos_pred))
        D.append((2 * TP) / (P + T))
    return sum(D) / len(D)


def eval_pred(predictions, labels, num_classes):
    (b, h, w) = predictions.shape
    dices = np.zeros(b)
    dices_no_background = np.zeros(b)
    for i in range(0, b):
        dices[i] = mean_dice(predictions[i,:,:], labels[i,:,:], num_classes)
        dices_no_background[i] = mean_dice_no_background(predictions[i,:,:], labels[i,:,:], num_classes)
    return (dices, dices_no_background)
