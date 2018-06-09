#%% Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

#%%
def model_fn(features, labels, mode, params):
    
    ### METHODS ###
    
    # Constructs a convolution layer
    def add_conv_layer(input_tensor, k_size, out_channels, pad):
        layer = tf.layers.conv2d(
                inputs = input_tensor,
                filters = out_channels,
                kernel_size = [k_size, k_size],
                padding = pad,
                activation=tf.nn.relu,
                kernel_initializer = tf.contrib.layers.variance_scaling_initializer())
        return layer
    
    # Constructs a batch normalization layer
    def add_batch_norm_layer(input_tensor, mode, bn_decay):
        layer = tf.contrib.layers.batch_norm(
                input_tensor,
                updates_collections = None,
                decay = bn_decay,
                is_training = mode == tf.estimator.ModeKeys.TRAIN)
        return layer
    
    # Constructs a max pooling layer
    def add_pool_layer(input_tensor, p_size, p_stride):
        layer = tf.layers.max_pooling2d(inputs = input_tensor, 
                                        pool_size = [p_size, p_size], 
                                        strides = p_stride)
        return layer

    # Constructs bilinear upsampling weights given upsampling factor
    # Taken from http://cv-tricks.com/image-segmentation/transpose-convolution-in-tensorflow/
    def get_bilinear_weights(filter_shape, upscale_factor, out_channels):
        filter_size = filter_shape[0]
        if (filter_size % 2 == 1):
            centre = upscale_factor - 1
        else:
            centre = upscale_factor - 0.5
            
        bilinear = np.zeros([filter_size, filter_size])
        for x in range(filter_size):
            for y in range(filter_size):
                value = (1 - abs((x - centre)/ upscale_factor)) * \
                (1 - abs((y - centre)/ upscale_factor))
                bilinear[x, y] = value
        weights = np.zeros(filter_shape)
        
        # Add bilinear weights to the filter channel correponding to
        # output channel, such that input channels are handled seperately.
        for i in range(filter_shape[2]):
            weights[:, :, i, i] = bilinear
        init = tf.constant_initializer(value=weights, dtype=tf.float32)
        bilinear_weights = tf.get_variable(name="bilinear_weights_" + 
                                           str(out_channels), initializer=init,
                                           shape = filter_shape)
        return bilinear_weights

    # Creates an upsampling layer using transposed 2D convolution
    # with weights initialized to bilinear interpolation
    def add_up_layer_bilinear(input_tensor, out_channels, upscale_factor):
        in_shape = tf.shape(input_tensor)
        in_channels = input_tensor.get_shape()[3]
        filter_size = 2 * upscale_factor - upscale_factor % 2
        filter_shape = [filter_size, filter_size, out_channels, in_channels]
        filter_weights = get_bilinear_weights(filter_shape, upscale_factor,
                                              out_channels)

        stride = upscale_factor
        strides = [1, stride, stride, 1]
        
        #out_h = ((in_shape[1] - 1) * stride) + 1
        #out_w = ((in_shape[2] - 1) * stride) + 1
        
        # Alteration to maintain original dimensions
        out_h = in_shape[1] * 2
        out_w = in_shape[2] * 2
        out_shape = tf.stack([in_shape[0], out_h, out_w, out_channels])
        
        layer = tf.nn.conv2d_transpose(input_tensor, filter_weights, 
                                        out_shape, strides=strides, 
                                        padding='SAME')
        
        # NEED THIS TO REGAIN THE CHANNELS SHAPE.
        layer = tf.reshape(layer, out_shape)
        return layer
    
    
    # Alternative to initializing as bilinear weights
    def add_up_layer(input_tensor, filter_size, out_channels, upscale_factor):
        
        layer = tf.layer.conv2d_transpose(input_tensor, out_channels,
                                          filter_size, strides=upscale_factor,
                                          padding='SAME')
        return layer
        
    
    # Merge tensors by depth-wise concatenation
    def merge_layers(deep_tensor, skipped_tensor):
        new_tensor= tf.concat([deep_tensor, skipped_tensor], 3)
        return new_tensor
    
    # Crop tensor1 to tensor2 height and width
    def crop_layer(tensor1, tensor2):
        t1_shape = tf.shape(tensor1)
        t2_shape = tf.shape(tensor2)
        height_diff = t1_shape[1] - t2_shape[1]
        width_diff = t1_shape[2] - t2_shape[2]
        offset_height = height_diff // 2
        offset_width = width_diff // 2
        layer = tf.image.crop_to_bounding_box(tensor1, offset_height,
                                              offset_width, t2_shape[1], 
                                              t2_shape[2])
        return layer


    ### PARAMETERS ###

    # Fixed FCN parameters
    conv_padding = 'SAME'
    up_conv_padding = 'SAME'
    scaling_factor = 2
    up_sampling_conv_ksize = 2
    score_tensor_ksize = 1
    
    # Variable FCN parameters
    num_classes = params["num_classes"]
    height = params["data_height"]
    width = params["data_width"]
    num_in_channels = params["num_in_channels"]
    k_size = params["conv_ksize"]
    num_channels_base = params["num_channels_base"]
    scale_depth = params["scale_depth"]
    learning_rate = params["learning_rate"]
    bn_decay = params["bn_decay"]
    
    ### FCN MODEL ARCHITECTURE ###
    
    input_tensor = tf.reshape(features["x"], [-1, height, width, num_in_channels])
    
    layers_to_skip = []
    prev_tensor = input_tensor
    # Convolution
    for s in range(0, scale_depth - 1):
        channels = num_channels_base * 2 ** s
        conv1 = add_conv_layer(prev_tensor, k_size, channels, conv_padding)
        conv2 = add_conv_layer(conv1, k_size, channels, conv_padding)
        bn = add_batch_norm_layer(conv2, mode, bn_decay)
        pool = add_pool_layer(bn, scaling_factor, scaling_factor)
        prev_tensor = pool
        
        # Collect tensors that also "skip" to deconvolution half of FCN
        layers_to_skip.append(bn)
    
    # Botton 2 convolution layers before upsampling
    channels = num_channels_base * 2 ** (scale_depth - 1)
    conv1 = add_conv_layer(prev_tensor, k_size, channels, conv_padding)
    conv2 = add_conv_layer(conv1, k_size, channels, conv_padding)
    prev_tensor = conv2

    # Deconvolution    
    for s in range(scale_depth - 1, 0, -1):
        channels_before_up = num_channels_base * 2 ** s
        channels_after_up = num_channels_base * 2 ** (s - 1)
        up_samp = add_up_layer_bilinear(prev_tensor, channels_before_up,
                                        scaling_factor)
        conv_up = add_conv_layer(up_samp, up_sampling_conv_ksize, 
                                 channels_after_up, up_conv_padding)
        
        bn1 = add_batch_norm_layer(conv_up, mode, bn_decay)
        # Use "skipped" tensor here
        cropped = crop_layer(layers_to_skip[s - 1], bn1)
        merge = merge_layers(bn1, cropped)
        conv1 = add_conv_layer(merge, k_size, channels_after_up, conv_padding)
        conv2 = add_conv_layer(conv1, k_size, channels_after_up, conv_padding)
        bn2 = add_batch_norm_layer(conv2, mode, bn_decay)
        prev_tensor = bn2


    # Add convolution layer to score classes
    conv_class = add_conv_layer(prev_tensor, score_tensor_ksize, num_classes, 
                                conv_padding)

    
    ### PROCESSING PREDICTIONS ###

    # Prediction
    predictions = {"classes": tf.argmax(conv_class, axis=3, name="argmax_tensor"),
                   "input_image" : input_tensor
            }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    
    # Preparing labels and predictions for pixel-wise cross entropy
    flat_preds = tf.reshape(conv_class, [-1, num_classes])
    flat_labels = tf.cast(tf.reshape(labels, [-1]), dtype = tf.int32)
    flat_labels_onehot = tf.one_hot(flat_labels, num_classes)
    
    # Add epsilon for numerical stability to prevent model to diverge with NaN loss
    epsilon = tf.constant(1e-8)
    flat_preds = flat_preds + epsilon 
    
    # Create class weights to compensate for imbalanced class frequencies
    flat_labels_float = tf.cast(flat_labels_onehot, dtype = tf.float32)
    class_occur = tf.reduce_sum(flat_labels_float, axis = 0) + 1
    total_occur = tf.cast(tf.shape(flat_labels_float)[0], dtype = tf.float32)
    class_freq = tf.divide(tf.divide(total_occur, class_occur), total_occur)
    class_weights = tf.reduce_sum(class_freq * flat_labels_float, axis = 1)
    
    # Compute unweighted cross entropy
    cross_entropies = tf.nn.softmax_cross_entropy_with_logits(
            labels = flat_labels_onehot, logits = flat_preds)
    
    # Apply the class weights
    cross_entropies_weighted = cross_entropies * class_weights
    
    pw_loss = tf.reduce_sum(cross_entropies_weighted, name = "pw_loss")
    
    # Batch size normalized loss for logging
    pw_loss_normalized = tf.divide(pw_loss, 
                                   tf.cast(tf.shape(input_tensor)[0],
                                           dtype = tf.float32),
                                           name = "pw_loss_normalized")

    
    # Choose optimizer and what loss to minimize
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        train_op = optimizer.minimize(loss = pw_loss, 
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=pw_loss_normalized, 
                                          train_op=train_op)

    eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                    labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=pw_loss, 
                                      eval_metric_ops=eval_metric_ops)
