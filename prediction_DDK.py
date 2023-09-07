# Development of a Speech Application for the Identification of Parkinson's Disease
# This program works with a database consisting of HC (healthy control) and 
# PD (Parkinson's disease patient) audio files.
# The purpose of this file is to use the trained model and make predictions based on
# input audio files.

# Import necessary libraries
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Load a pre-trained model
model = tf.keras.models.load_model("model_DDK.h5")

# Define the path to the dataset
out_path_tr = 'data/petaka_f/train'
out_path_val = 'data/petaka_f/val'
data_tr = pathlib.Path(out_path_tr)
data_val = pathlib.Path(out_path_val)

# Create a training dataset using the audio files in the 'data_tr' directory
train_ds= tf.keras.utils.audio_dataset_from_directory(
    directory=data_tr,
    batch_size=16,
    seed=0,
    output_sequence_length=50000 # 16000 aprox. 1s
    )

# Create a validation dataset using the audio files in the 'data_val' directory
val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_val,
    batch_size=15,
    seed=0,
    output_sequence_length=50000 # 16000 aprox. 1s
    )


# Get class labels from the training dataset
label_names = np.array(train_ds.class_names)

# Function to compute spectrogram from waveform
def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(
        waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

# Read and preprocess an audio file
file_name= 'AVPEPUDEAC0001_petaka.wav'  
audio_file = file_name
audio_file = tf.io.read_file(str(audio_file))
audio_file, sample_rate = tf.audio.decode_wav(audio_file, desired_channels=1, desired_samples=50000,)
audio_file = tf.squeeze(audio_file, axis=-1)
waveform = audio_file
audio_file = get_spectrogram(audio_file)
audio_file = audio_file[tf.newaxis, ...]

# Get model predictions for the preprocessed audio file
prediction = model(audio_file)

# Create a bar plot of the predicted class probabilitieséé
x_labels = label_names
plt.bar(x_labels, tf.nn.softmax(prediction[0]))
plt.title(f"File name: {file_name}")
plt.show()