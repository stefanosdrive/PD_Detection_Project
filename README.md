# Development of a Speech Application for the Identification of Parkinson’s Disease
An application that allows users to predict the risk of a patient having Parkinson's Disease based on the analysis of diadochokinesis (DDK) speech data from microphone recordings. Created during the Summer 2023 Erasmus+ Traineeship held at ETSIINF, Universidad Politécnica de Madrid 


##  Summary
Parkinson's disease is a prevalent neurodegenerative disorder that can significantly impact an individual's quality of life. Early detection and monitoring of the disease are essential for effective treatment and care. This report outlines the development of a Speech Application for the identification of Parkinson's Disease using Diadochokinetic (DDK) speech analysis. DDK analysis involves assessing an individual's ability to produce rapid, alternating speech sounds, which can reveal subtle changes in speech motor control associated with Parkinson's disease.

The primary goal of this application is to provide an accessible, non-invasive tool for early detection and continuous monitoring of Parkinson's disease through the analysis of DDK speech patterns.


##  Introduction
###   1.1 Background
Parkinson's disease is characterized by motor symptoms, including bradykinesia, tremors, and rigidity. Speech difficulties are also common and can manifest as reduced speech rate, imprecise articulation, and altered prosody. DDK speech analysis assesses an individual's ability to rapidly repeat a sequence of syllables, offering valuable insights into speech motor control and coordination.

### 1.2 Objectives
The primary objectives of developing this Speech Application for Parkinson's Disease identification using DDK speech analysis are as follows:

   * Enable early detection of Parkinson's disease through the analysis of DDK speech patterns.
   * Provide a user-friendly tool for patients and healthcare professionals to assess speech motor control.
   * Implement machine learning techniques to improve accuracy in identifying Parkinson's disease-related speech changes.
   * Facilitate remote monitoring and long-term tracking of disease progression.


## Methodology
### 2.1 Data Collection
A diverse dataset of DDK speech recordings will be collected, including recordings from both Parkinson's disease (PD) patients and healthy individuals (HC). 

Data collection consists of an initial 600 files, each being distributed into 6 types: ka-ka-ka, pakata, pa-pa-pa, pataka, petaka, ta-ta-ta, each having 50 files for every category (HC and PD). These are the base audio files which will be used throughout the whole project.

### 2.2 Data Splitting
The data will initially be split into 2 categories: training and validation.

The data from the training set is obviously used for training our model. This set will make up the majority of the total data (80%) and by utilising it, our model determines important features relevant in our study.

The data from the validation set is used to tune the hyperparameters of our classifier. The validation set contrasts with training and test sets in that it is an intermediate phase used for choosing the best model and optimizing it. It is in this phase that parameter tuning occurs for optimizing the selected model. Overfitting is checked and avoided in the validation set to eliminate errors that can be caused for future predictions and observations if an analysis corresponds too precisely to a specific dataset.

In the end, by splitting our data using the appropriate functions, our working directory will look like this:
```
data/
    └── DDK_F/
        ├── train/
        │   ├── HC/
        │   │   ├── ka1.wav
        │   │   ├── pakata1.wav
        │   │   ├── pa1.wav
        │   │   └── ...
        │   ├── PD
        │   ├── ka01.wav
        │   ├── pakata01.wav
        │   ├── pa01.wav
        │   └── ...
        └── val/
            ├── HC/
            │   ├── ka2.wav
            │   ├── pakata2.wav
            │   ├── pa2.wav
            │   └── ...
            └── PD      /
                ├── ka02.wav
                ├── pakata02.wav
                ├── pa02.wav
                └── ... 
```

### 2.3 Data Augmentation
In order to train the model to be as capable as possible in the detection of Parkinson's disease, we will need a larger dataset which can be obtained really easy through augmentation.

__Audiomentations__ is a really useful Python library for deep learning containing lots of Transforms such as ```AddBackgroundNoise```, ```AddGaussianNoise``` etc.

For our case, the most relevant Transforms that have been used for this project are:
1. ```AddGaussianNoise``` (adds Gaussian noise to the audio samples)
2. ```RoomSimulator``` (simulates the effect of a room on an audio source) 
3. ```TimeStretch``` (changes the speed without changing the pitch)
4. ```PitchShift``` (shifts the pitch up or down without changing the tempo)

The appropriate parameters have been chosen such that the augmented audio does not depart too much from a realistic sample.

The augmentation can be repeated as much as the user wishes to. (In our project, we expanded the dataset from 100 to 6600 .wav files)

### 2.4 DDK Analysis
DDK speech samples will be analysed for various parameters, including speech rate, syllable duration, and errors in articulation. These parameters will serve as input features for the machine learning model.

A ```batch_size=16``` and ```output_sequence_length=50000``` have been chosen for the training and validation datasets, as they have proven to be the most efficient regarding to training times and accuracy & loss results.

After sizing, the loaded waveforms in the dataset are converted from the time-domain signals into the time-frequency-domain signals by computing the short-time Fourier transform (STFT) to convert the waveforms to as spectrograms, which show frequency changes over time and can be represented as 2D images. These spectrogram images will be fed into our neural network to train our model.

The importance of STFT lies in its ability to analyse signals that exhibit temporal variations. In many real-world scenarios, signals change over time, and understanding these changes is essential for making informed decisions. In audio processing, STFT is used for tasks such as speech recognition, music analysis, and sound synthesis.

### 2.5 Machine Learning Model
A machine learning model, such as a convolutional neural network (CNN) will be trained using the DDK speech data. The model will be designed to differentiate between speech patterns of individuals with Parkinson's disease and those without.

Our model is composed of a ```Conv2D``` layer with 32 filters followed by a ```MaxPooling2D``` layer, two ```Conv2D``` layers each with 64 filters and each followed by ```MaxPooling2D``` layers, one ```Conv2D``` layers with 128 filters followed by a ```MaxPooling2D``` layer, a ```Flatten``` and a ```Dense``` layer with 128 neurons.

In order to prevent overfitting, the following measures have been taken:
* added to each ```Conv2D``` a kernel L2 class regularizer
* created a ```lr_scheduler``` function which decreases ```learning_rate``` with a certain value at a specified number of ```EPOCHS```.
* added ```layers.Dropout```

## Results and Discussion
### 3.1 Model Performance
The machine learning model's performance was evaluated after training by displaying a confusion matrix with the appropriate labels:

![image](https://github.com/stefanosdrive/PD_Disease_Detection_Project/assets/140710587/b06df0dc-fbbb-4fad-8b65-1618cf9dd568)

On average, due to the augmentations' part randomness factor, the model varies with an accuracy between 82% and 89%. For this particular example of the confusion matrix, the accuracy was 87.8%.

```model.evaluate(test_spectrogram_ds, return_dict=True)
42/42 [==============================] - 1s 11ms/step - loss: 0.5123 - accuracy: 0.8780
Out[8]: {'loss': 0.5123199820518494, 'accuracy': 0.8780120611190796
```

Also, by using prediction_DDk.py on a separately audio file which was not included during training but we know for certain that it is of class __PD__, we obtain the expected results:

![image](https://github.com/stefanosdrive/PD_Disease_Detection_Project/assets/140710587/e3068748-297b-4e03-8c76-5676e2b961cb)


### 3.2 User Interface
__This part is yet to be implemented.__

In the future, the mobile application's user interface will be designed for ease of use, ensuring accessibility for users of varying ages and technical abilities. It will guide users through the DDK speech tasks, provide clear instructions, and display results in an easily interpretable manner.


## Conclusion
The development of a Speech Application for the identification of Parkinson's Disease using DDK speech analysis presents a promising approach to early diagnosis and continuous monitoring of the disease. 

By leveraging machine learning and Diadochokinetic speech analysis, this application can provide a valuable tool for both patients and healthcare professionals.

Further research, testing, and validation are needed to ensure the application's accuracy and reliability in real-world scenarios. However, it has the potential to make a significant positive impact on the early detection and management of Parkinson's disease.
