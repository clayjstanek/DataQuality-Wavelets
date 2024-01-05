# -*- coding: utf-8 -*-                                        
"""
Created on Tue Aug 29 17:41:26 2023
#use ll053024 conda environment
@author: cstan
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CUDNN_DETERMINISTIC']='1'
import sys

import pandas as pd
pd.options.display.precision   =   3
pd.options.display.max_rows    = 150
pd.options.display.max_columns = 800

import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.keras.backend.set_floatx('float32')
tf.random.set_seed(991871)
tf.config.experimental.enable_op_determinism()
from   tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from   keras.layers import Conv2DTranspose
from   tensorflow.keras.models import Model
from   tensorflow.keras.optimizers import Adam
from   tensorflow.keras.utils import plot_model
from   tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
#from   physics import seq2seq as pjt
#from   fun import fancy_print

from   sklearn.model_selection import train_test_split
import pickle
import h5py
import joblib
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
    filename= p.joinpath('logs', 'waveletautoencoderV102823.log'),
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

coherenceraw_path = p.joinpath('pkl_brain')
if not coherenceraw_path.exists():
    Path.mkdir(coherenceraw_path, parents=True)
    
#load training data    
pickle_files = coherenceraw_path.glob('*.pkl')

# If you want to have a list of all matched files, you can convert the iterator to a list
pickle_files_list = list(pickle_files)
wavecoherencedata_list = []

for i,file_path in tqdm(zip(range(len(pickle_files_list)),pickle_files_list)):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        data = np.array(data, dtype = 'float32')
        wavecoherencedata_list.append(data[0])
        logging.info(f"Loaded pickle file {file_path}")

N = 268681
ARRAY_SIZE = 4096
BATCHES = 65
COLS = 67
logging.info(f"N= {N}, ARRAY_SIZE={ARRAY_SIZE}, COLUMNS= {COLS}, BATCHES={BATCHES}")

"""
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
        with open(file_path, 'rb') as file:
            testdata = pickle.load(file)
            testdata = np.array(testdata, dtype = 'float32')
            testdata_array.append(data[0])
            logging.info(f"Loaded pickle test file {file_path}")

testdata_array = np.concatenate(testdata_array)   
"""
N = 268681
ARRAY_SIZE = 4096
BATCHES = 65
COLS = 67
logging.info(f"N= {N}, ARRAY_SIZE={ARRAY_SIZE}, COLUMNS= {COLS}, BATCHES={BATCHES}")


logging.info(f"Set path to raw coherence data as {coherenceraw_path}")
current_dir = Path.cwd()

# Print the current working directory
print(current_dir)
# Use the glob method to match all files with the .pickle extension



# Define the encoder
input_img = Input(shape=(110, 4096, 1))  # Grayscale image

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 4), padding='same')(x)
x = Dropout(0.2)(x)
#image size is 55,1024
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((1, 2), padding='same')(x)
#image size is 55,512
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((1, 4), padding='same')(x)
x = Dropout(0.2)(x)
#image size = 55,128
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((1, 2), padding='same')(x)
bottleneck = Dropout(0.2)(x)
#image size = 11,32
decoded = bottleneck

"""
#image size = 55,64
x = Conv2D(384, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((5, 2), padding='same')(x)
bottleneck = Dropout(0.2)(x)
#image size = 11,32
decoded = bottleneck
# Define the decoder
x = Conv2DTranspose(384, (3, 3), strides=(5, 2), activation='relu', padding='same')(decoded)

#x = Conv2D(384, (3, 3), activation='relu', padding='same')(decoded)
#x = UpSampling2D((5, 2))(x)
x = Dropout(0.2)(x)
#image size = 55,64
"""
x = Conv2DTranspose(256, (3, 3), strides=(1, 2), activation='relu', padding='same')(decoded)

#x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)                                                                                                                                                                      
#x = UpSampling2D((1, 2))(x)
#image size is 55x128

x = Conv2DTranspose(128, (3, 3), strides=(1, 4), activation='relu', padding='same')(x)

#x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
#x = UpSampling2D((1, 4))(x)
x = Dropout(0.2)(x)
#image size is 55,512

x = Conv2DTranspose(64, (3, 3), strides=(1, 2), activation='relu', padding='same')(x)
#x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
#x = UpSampling2D((1, 2))(x)

#image size is 55,1024
x = Conv2DTranspose(32, (3, 3), strides=(2, 4), activation='relu', padding='same')(x)


#x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
#x = UpSampling2D((2, 4))(x)
x = Dropout(0.2)(x)
# image size = 110,4096

decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
# size is 110x4096x1
# Create the autoencoder model
autoencoder = Model(input_img, decoded)
plot_model(autoencoder, to_file=p.joinpath('autoencoder_wavelet_coherence.png'),
           show_shapes=True, show_layer_names=True)

optimizer = Adam(learning_rate = 1e-4)
autoencoder.compile(optimizer=optimizer, loss='mean_squared_error', metrics = ['accuracy'])
logging.info("using Adam with learning rate 1e-4")

logging.info(autoencoder.summary())

#sys.exit(0)

from sklearn.preprocessing import MinMaxScaler
# Normalize the data.  It is best practice to normalize the feature set to [0,1]
scaler = MinMaxScaler()

# Assuming list_of_images is your list containing 80 2D arrays each of shape (110, 4096)
array_3d = np.stack(wavecoherencedata_list, axis=0)
#testarray_3d = np.stack(testdata_array, axis=0)
original_shape = array_3d.shape
#testoriginal_shape = testarray_3d.shape
array_2d = array_3d.reshape(original_shape[0], -1)
#testarray_2d = testarray_3d.reshape(testoriginal_shape[0], -1)
array_2d_scaled = scaler.fit_transform(array_2d)
#testarray_2d_scaled = scaler.fit_transform(testarray_2d)
# Find NaNs
nan_mask = np.isnan(array_2d_scaled)
#testnan_mask = np.isnan(testarray_2d_scaled)
# Locate NaNs
nan_indices = np.where(nan_mask)
#testnan_indices = np.where(testnan_mask)
print(np.shape(nan_indices))
print("Indices of NaNs:", list(zip(nan_indices[0], nan_indices[1])))
logging.info(f"Indices of NaNs: {list(zip(nan_indices[0], nan_indices[1]))}")
#print(np.shape(testnan_indices))
#print("Indices of NaNs:", list(zip(testnan_indices[0], testnan_indices[1])))
#logging.info(f"Indices of NaNs: {list(zip(testnan_indices[0], testnan_indices[1]))}")

# Replace NaNs with 0
array_2d_scaled[nan_mask] = 0
array_3d_scaled = array_2d_scaled.reshape(original_shape)
#testarray_2d_scaled[testnan_mask] = 0
#testarray_3d_scaled = testarray_2d_scaled.reshape(testoriginal_shape)
y = np.zeros(array_3d_scaled.shape[0])
x_train, _, y_train, _ = train_test_split(array_3d_scaled, y, test_size=None, random_state=42)
x_train = x_train[..., np.newaxis]  # Adds a channel dimension
x_train = x_train.astype(np.float32)
#x_train_tensor = tf.convert_to_tensor(x_train)


from sklearn import metrics
#scorers = list(metrics.SCORERS.keys())
#logging.info(scorers)

checkpoint = ModelCheckpoint(
    filepath= p.joinpath('autoencoder2D_checkpoint.h5'),  # Specify the file path where the model will be saved
    monitor='accuracy',  # Monitor a validation metric (e.g., val_loss) to decide when to save
    save_best_only=True,  # Save only the best model based on the monitored metric
    save_weights_only=False,  # Save the entire model including architecture
    mode='max',  # In 'min' mode, it saves when the monitored quantity decreases (e.g., val_loss)
    verbose=1  # 0 = no logging, 1 = progress bar logging, 2 = epoch-wise logging
    )

#train_history = autoencoder.fit(x_train_tensor, x_train_tensor, epochs=50, batch_size=16, shuffle=True,
#                validation_data=((x_test_tensor), (x_test_tensor)), callbacks=[checkpoint]) #validation_data=(val_images, val_images))


#sys.exit(0)
#x_test = testarray_3d_scaled
#x_test = x_test[..., np.newaxis]  # Adds a channel dimension
#x_test = x_test.astype(np.float32)
#x_test_tensor = tf.convert_to_tensor(x_test)

#train_history = autoencoder.fit(x_train_tensor, x_train_tensor, epochs=50, batch_size=16, shuffle=True,
#                validation_data=((x_test_tensor), (x_test_tensor)), callbacks=[checkpoint]) #validation_data=(val_images, val_images))
train_history = autoencoder.fit(x_train, x_train, epochs=50, batch_size=8, 
                                shuffle=True, callbacks=[checkpoint]) #validation_data=(val_images, val_images))
autoencoder.save(p.joinpath('autoencoder_2D_waveletcoherence102823final.h5'))
#x_test_recon = autoencoder.predict(x_test_tensor, verbose=1)


# the reconstruction score is the mean of the reconstruction errors (relatively high scores are anomalous)
#reconstruction_scores = np.mean((x_test_tensor - x_test_recon)**2, axis=1)
# store the reconstruction data in a Pandas dataframe
#anomaly_data = pd.DataFrame({'recon_score':reconstruction_scores})

# if our reconstruction scores our normally distributed we can use their statistics
#anomaly_data.describe()

# plotting the density will give us an idea of how the reconstruction scores are distributed
plt.xlabel('Reconstruction Score')
#anomaly_data['recon_score'].plot.hist(bins=200, range=[-.01, .03])
plt.show()

from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, roc_auc_score

#mse = mean_squared_error(x_test_recon, x_test_tensor)
#ind_mse = np.sum((x_test_recon - x_test_tensor)**2, axis=1)
#mse = np.mean(np.power(x_test_recon - x_test_tensor, 2), axis=1)

plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])
plt.legend(['loss on train data', 'loss on validation data'])
plt.show()

# Compute ROC curve
#fpr, tpr, thresholds = roc_curve(y, mse)
#roc_auc = roc_auc_score(y, mse)
#logging.info(f"fpr: {fpr}, tpr: {tpr}, thresholds: {thresholds}")
#logging.info(f"ROC_AUC: {roc_auc}")
# Plot ROC curve
#plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], linestyle='--')  # Diagonal line for random classifier
plt.show()

"""
This is a basic autoencoder. In practice, you might need to adjust the architecture, 
such as the number of filters, kernel sizes, and layers, based on the complexity of the 
data and the specifics of your application.
"""