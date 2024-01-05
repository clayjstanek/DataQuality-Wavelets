#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 08:44:19 2023

@author: cstanek
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
#os.environ['TF_CUDNN_DETERMINISTIC']='1'
os.environ["TF_GPU_ALLOCATOR"] = 'cuda_malloc_async'
import sys
import pandas as pd
import numpy as np
pd.options.display.precision   =   3
pd.options.display.max_rows    = 150
pd.options.display.max_columns = 800

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.keras.backend.set_floatx('float32')
tf.random.set_seed(991871)
tf.config.experimental.enable_op_determinism()

from   tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from   sklearn.preprocessing import MinMaxScaler
import pickle
import logging
from   pathlib import Path
from   tqdm import tqdm

p = Path(__file__)
p = p.parents[0]


logging.basicConfig(
    # Set the desired log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    # Specify the filename for the log file
    filename= p.joinpath('logs', 'predictautoencoderV121323.log'),
    filemode="w",  # "w" to overwrite the file on each run, "a" to append
)
gpus = tf.config.experimental.list_physical_devices('GPU')
print(f"Num GPUs Available: {gpus}"), 
logging.info(f"Num GPUs Available: {gpus}")
print(f"Devices: {tf.config.experimental.list_physical_devices()}")
logging.debug(f"Devices: {tf.config.experimental.list_physical_devices()}")

if gpus:
    try:
        # Allow memory growth for GPUs 
        for gpu in gpus:
#            tf.config.experimental.set_memory_growth(gpu, True)
            print("GPUs configured:", gpus)
    except RuntimeError as e:
        print(e)
#load model file
        
#load test data
test_data_folder_list = []    
test_data_folders = ['pkl_braintest', 'pkl_oval', 'pkl_black', 'pkl_spikes','pkl_red']

for i, tdf in enumerate(test_data_folders):
    tdfpath = p.joinpath(tdf)
    test_data_folder_list.append(tdfpath)

pickle_test_files = []  
test_labels = []  
for i, tdfl in enumerate(test_data_folder_list):
    pickle_test_files_type = list(tdfl.glob('*.pkl'))
    labels = np.ones(len(pickle_test_files_type))*i
    pickle_test_files.append(pickle_test_files_type)
    test_labels.append(labels)
    
test_labels_array = np.concatenate(test_labels)

testdata_array = []
for sublist in tqdm(pickle_test_files):
    for item in sublist:
        with open(item, 'rb') as file:
            testdata = pickle.load(file)
            testdata = np.array(testdata, dtype = 'float32')
            testdata_array.append(testdata[0])
            logging.info(f"Loaded pickle test file {item}")

testdata_array = np.concatenate(testdata_array)  
testarray_3d = np.stack(testdata_array, axis=0)
testoriginal_shape = testarray_3d.shape
testarray_2d = testarray_3d.reshape(testoriginal_shape[0], -1)

# Normalize the data.  It is best practice to normalize the feature set to [0,1]
scaler = MinMaxScaler()
testarray_2d_scaled = scaler.fit_transform(testarray_2d)
testnan_mask = np.isnan(testarray_2d_scaled)
testarray_2d_scaled[testnan_mask] = 0

testnan_indices = np.where(testnan_mask)
print(np.shape(testnan_indices))
print("Indices of NaNs:", list(zip(testnan_indices[0], testnan_indices[1])))
logging.info(f"Indices of NaNs: {list(zip(testnan_indices[0], testnan_indices[1]))}")

testarray_3d_scaled = testarray_2d_scaled.reshape(testoriginal_shape)

x_test = testarray_3d_scaled
x_test = x_test[..., np.newaxis]  # Adds a channel dimension
x_test = x_test.reshape((757, 110, 4096, 1))
x_test = x_test.astype(np.float32)
#x_test_tensor = tf.convert_to_tensor(x_test) 

#filepath= p.joinpath('autoencoder2D_checkpointWin11.h5')
filepath= p.joinpath('autoencoder2DBatchNorm_checkpointWin11.h5')
autoencoder = load_model(filepath)
x_test_recon = autoencoder.predict(x_test, verbose=1)
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, roc_auc_score
mse_per_image = []

for i in range(x_test.shape[0]):
    mse = mean_squared_error(x_test[i].flatten(), x_test_recon[i].flatten())
    mse_per_image.append(mse)

# Convert the list to a numpy array
mse_per_image = np.array(mse_per_image)
anomaly_data = pd.DataFrame({'recon_score':mse_per_image, 'y_label':np.concatenate(test_labels)})
anom_min = np.min(mse_per_image)
anom_max = np.max(mse_per_image)
# if our reconstruction scores our normally distributed we can use their statistics
logging.info(f"anomaly data description: {anomaly_data.describe()}")

# plotting the density will give us an idea of how the reconstruction scores are distributed

# Define colors for each unique y_label
colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow', 4: 'purple'}

# Get unique y_labels
y_labels = anomaly_data['y_label'].unique()

# Sort y_labels if they are not in order
y_labels = np.sort(y_labels)

# Creating the histogram
for itype, y_label in zip(test_data_folders, y_labels):
    filtered_data = anomaly_data[anomaly_data['y_label'] == y_label]['recon_score']
    plt.hist(filtered_data, bins=200, range=[anom_min-.01, anom_max+.01], 
             color=colors[y_label], alpha=0.5, label=f'Label {itype}')

# Customizing the histogram
plt.title('Histogram of Reconstruction Error by Image Category')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.savefig('HistogramOfReconstructionError121323.jpg', dpi=600)
# Displaying the histogram
plt.show()

# Assuming matrix1 and matrix2 are your input matrices of shape (757, 110, 4096, 1)
#mse = mean_squared_error(x_test_recon, x_test)
#ind_mse = np.sum((x_test_recon - x_test)**2, axis=1)
#mse = np.mean(np.power(x_test_recon - x_test, 2), axis=1)
anomaly_data['y_label'] = anomaly_data['y_label'].apply(lambda x: 0 if x == 0 else 1)
sorted_df = anomaly_data.sort_values(by='recon_score')
# Compute ROC curve
fpr, tpr, thresholds = roc_curve(sorted_df['y_label'], sorted_df['recon_score'])
roc_auc = roc_auc_score(sorted_df['y_label'], sorted_df['recon_score'])
logging.info(f"fpr: {fpr}, tpr: {tpr}, thresholds: {thresholds}")
logging.info(f"ROC_AUC: {roc_auc}")
# Plot ROC curve
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], linestyle='--')  # Diagonal line for random classifier
plt.savefig('ROCAUC_Autoencoder2Dwavelet121323.jpg', dpi=600)
plt.show()
sys.exit(0)