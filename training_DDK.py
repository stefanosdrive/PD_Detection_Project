# Development of a Speech Application for the Identification of Parkinson's Disease
# This program works with a database consisting of 6 different key-sentences already categorized into
# HC (Healthy Control) and PD (Parkinson's Disease patient).
# The purpose of this file is to train a model that will be used to categorise
# audio files into HC and PD.

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Import necessary libraries
import pathlib
import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from augmentation import augment_audio_files
from IPython import display
from keras.callbacks import LearningRateScheduler
from split import split_data
from tensorflow.keras import layers
from tensorflow.keras import models

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Speech commands + Waveforms (optional)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

DATASET_PATH = 'data/DDK'               # Input datapath
output_path = 'data/DDK_f'              # Output datapath 
out_path_tr = 'data/DDK_f/train'        # Training datapath
out_path_val = 'data/DDK_f/val'         # Validation datapath
data_tr = pathlib.Path(out_path_tr)
data_val = pathlib.Path(out_path_val)

# Splitting the data into training & validation sets
split_data(DATASET_PATH, output_path, random.randint(1, 1000), (0.8, 0.2))

# Augmenting the database
augment_audio_files(output_path, output_path, 10)

# Create a training dataset using the audio files in the 'data_tr' directory
train_ds= tf.keras.utils.audio_dataset_from_directory(
    directory=data_tr,
    batch_size=16,
    seed=0,
    output_sequence_length=50000        # 16000 aprox. 1s
    )

# Create a validation dataset using the audio files in the 'data_val' directory
val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_val,
    batch_size=16,
    seed=0,
    output_sequence_length=50000        # 16000 aprox. 1s
    )

label_names = np.array(train_ds.class_names)
print()
print("Label names:", label_names)      # HC & PD

def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels

train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

for example_audio, example_labels in train_ds.take(1):  
    print(example_audio.shape)
    print(example_labels.shape)

# UNCOMMENTED, USE ONLY IF YOU WANT TO VISUALISE AUDIO SIGNALS
# plt.figure(figsize=(16, 10))
# rows = 3
# cols = 1
# n = rows * cols
# for i in range(n):
#   plt.subplot(rows, cols, i+1)
#   audio_signal = example_audio[i]
#   plt.plot(audio_signal)
#   plt.title(label_names[example_labels[i]])
#   plt.yticks(np.arange(-0.2, 0.2, 0.02))
#   plt.ylim([-0.1, 0.1])


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Convert waveforms to spectrograms
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Define a function to convert waveform to a spectrogram
def get_spectrogram(waveform):
    # Convert waveform to a spectrogram via Short-Time Fourier Transform (STFT)
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    
    # Obtain the magnitude of the STFT
    spectrogram = tf.abs(spectrogram)
    
    # Add a `channels` dimension for compatibility with convolutional layers
    spectrogram = spectrogram[..., tf.newaxis]
    
    return spectrogram

# Loop over examples and display information
for i in range(3):
    label = label_names[example_labels[i]]
    waveform = example_audio[i]
    spectrogram = get_spectrogram(waveform)
    
    print('Label:', label)
    print('Waveform shape:', waveform.shape)
    print('Spectrogram shape:', spectrogram.shape)
    print('Audio playback')
    display.display(display.Audio(waveform, rate=16000))

# Define a function to plot spectrograms
def plot_spectrogram(spectrogram, ax):
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)
    
    # Convert frequencies to log scale and transpose for plotting
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)

# Define a function to create spectrogram datasets
def make_spec_ds(ds):
    return ds.map(
        map_func=lambda audio, label: (get_spectrogram(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE)

# Create spectrogram datasets for training, validation, and testing
train_spectrogram_ds = make_spec_ds(train_ds)
val_spectrogram_ds = make_spec_ds(val_ds)
test_spectrogram_ds = make_spec_ds(test_ds)

for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
  break

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Build and train the model
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(3300).prefetch(tf.data.AUTOTUNE)
val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)

input_shape = example_spectrograms.shape[1:]
print('Input shape:', input_shape)
num_labels = len(label_names)

with tf.device('/GPU:0'):
# Create a sequential model
    model = models.Sequential([
        # Input layer with the specified input shape
        layers.Input(shape=input_shape),
        
        # First convolutional layer with 32 filters, ReLU activation, and L2 regularization
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape,
                      kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        
        # Max-pooling layer to reduce spatial dimensions by a factor of 2
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional layer with 64 filters, ReLU activation, and L2 regularization
        layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.MaxPooling2D((2, 2)),
        
        # Third convolutional layer with 64 filters, ReLU activation, and L2 regularization
        layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.MaxPooling2D((2, 2)),
        
        # Fourth convolutional layer with 128 filters, ReLU activation, and L2 regularization
        layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten the feature maps into a 1D vector + dropout layer to reduce overfitting
        layers.Flatten(),
        layers.Dropout(0.5),
        
        # Fully connected (dense) layer with 128 units and ReLU activation
        layers.Dense(128, activation='relu'),
        
        # Dropout layer to reduce overfitting by randomly setting a fraction of input units to 0
        layers.Dropout(0.5),
        
        # Final dense layer with num_labels units and softmax activation for classification
        layers.Dense(num_labels)
    ])
            
model.summary()

# Learning Rate decayer 
def lr_scheduler(epoch, lr):
     decay_rate = 0.1
     decay_step = 10                    # No. of EPOCHS
     if epoch % decay_step == 0 and epoch:
         return lr * decay_rate
     return lr
callbacks = [LearningRateScheduler(lr_scheduler, verbose=1)]
    
model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

EPOCHS = 50
with tf.device('/GPU:0'):

    history = model.fit(
        train_spectrogram_ds,
        validation_data=val_spectrogram_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

# UNCOMMENTED, USE ONLY IF YOU WANT TO TRACK loss/val_loss & accuracy/val_accuracu DURING EPOCHS
# metrics = history.history
# plt.figure(figsize=(16,6))
# plt.subplot(1,2,1)
# plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
# plt.legend(['loss', 'val_loss'])
# plt.ylim([0, max(plt.ylim())])
# plt.xlabel('Epoch')
# plt.ylabel('Loss [CrossEntropy]')

# plt.subplot(1,2,2)
# plt.plot(history.epoch, 100*np.array(metrics['accuracy']), 100*np.array(metrics['val_accuracy']))
# plt.legend(['accuracy', 'val_accuracy'])
# plt.ylim([0, 100])
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy [%]')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Display a confusion matrix
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

with tf.device('/GPU:0'):
    y_pred = model.predict(test_spectrogram_ds)
    y_pred = tf.argmax(y_pred, axis=1)
    y_true = tf.concat(list(test_spectrogram_ds.map(lambda s,lab: lab)), axis=0)
    
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx,
                xticklabels=label_names,
                yticklabels=label_names,
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Export and save the model
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

save_path = "C:/Users/sauci/.spyder-py3/TenserFlow"

model.save("model_DDK.h5")
model.save("saved_DDK")     # Assets

print(f"Model model_DDK.h5 saved at {save_path}.")


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Evaluate model performance
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

model.evaluate(test_spectrogram_ds, return_dict=True)