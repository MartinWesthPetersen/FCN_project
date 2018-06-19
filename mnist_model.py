# IMPORTS
import numpy as np
import pickle
import unet_classification as unet
import pickle
import os

# LOAD TRAIN (AND TEST?) DATA AND LABELS
train_path = '../mnistSMALL/mnist100_train.p'
val_path = '../mnistSMALL/mnist10_val.p'
results_path = 'results/model_performance_val10.p'

# hacky solution to hardcode the amount of images per class in the training set
nr_examples = 100


with open(train_path, 'rb') as handle:
    train_dict = pickle.load(handle)
train_tensor = np.dstack([train_dict[digit] for digit in train_dict.keys()])
train_tensor = np.moveaxis(train_tensor,  -1, 0)
train_tensor = np.expand_dims(train_tensor, axis=3)
train_labels = []
for digit in train_dict.keys():
    for _ in range(nr_examples):
        train_labels.append(digit)
train_labels = np.array(train_labels)

with open(val_path, 'rb') as handle:
    val_dict = pickle.load(handle)
val_tensor = np.dstack([val_dict[digit] for digit in val_dict.keys()])
val_tensor = np.moveaxis(val_tensor,  -1, 0)
val_tensor = np.expand_dims(val_tensor, axis=3)
val_labels = []
for digit in val_dict.keys():
    for _ in range(val_dict[0].shape[-1]):
        val_labels.append(digit)
val_labels = np.array(val_labels)

num_classes = 10
conv_ksize = 5
num_channels_base = 16  # (or 8 for starters?)
scale_depth = 2
learning_rate = 1e-5
bn_decay = 0.9
model_dir_name = "MNIST_model"

batch_size = 16
training_steps = 150
log_interval = 50

rounds = 10
classifier = unet.create_unet_model(train_tensor, num_classes, conv_ksize,
                                    num_channels_base, scale_depth,
                                    learning_rate, bn_decay, model_dir_name)

# load performance dictionary to continue training, otherwise initialize new
if os.path.isfile(results_path):
    with open(results_path, 'rb') as handle:
        model_perf = pickle.load(handle)

model_perf = {'train': [], 'val': []}

for j in range(rounds):
    print("***** TRAINING MODEL " + str(j) + " *****")
    classifier = unet.train_unet_model(train_tensor, train_labels, classifier, batch_size,
                                       training_steps, log_interval)
    print("***** EVALUATING MODEL " + str(j) + " *****")
    # Evaluate on training
    train_pred = unet.predict_unet_model(train_tensor, classifier)
    train_acc = unet.compute_accuracy(train_labels, train_pred.flatten())
    model_perf['train'].append(train_acc)

    # Evaluate on test
    val_pred = unet.predict_unet_model(val_tensor, classifier)
    val_acc = unet.compute_accuracy(val_labels, val_pred.flatten())
    model_perf['val'].append(val_acc)

    with open('results/model_performance.p', 'wb') as handle:
        pickle.dump(model_perf, handle)

    print('\n \n ***** RESULTS *****:', model_perf)

    # Early stopping criterion
    if j >= 10:
        current_val_error = model_perf['val'][j]
        if model_perf['val'][j-1] >= current_val_error and model_perf['val'][j-2] >= current_val_error:
            print('EARLY STOPPING AFTER %s EPROCHS.' %j)
            break
