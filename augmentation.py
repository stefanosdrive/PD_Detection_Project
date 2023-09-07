# Development of a Speech Application for the Identification of Parkinson's Disease
# This program works with a database consisting of HC (healthy control) and 
# PD (Parkinson's disease patient) audio files.
# The purpose of this file is to enlarge the audio database by using certain variations applied
# to the original audio files.

# Import necessary libraries
import os
from audiomentations import Compose, AddGaussianNoise, RoomSimulator, TimeStretch, PitchShift
import soundfile as sf

def augment_audio_files(input_directory, output_directory, num_iterations):
    augmentations = Compose([
        AddGaussianNoise(min_amplitude=0.0005, max_amplitude=0.002, p=0.3),
        RoomSimulator(
            min_size_x=3.6,
            max_size_x=5.6,
            min_size_y=3.6,
            max_size_y=4,
            min_size_z=2.4,
            max_size_z=3,
            min_absorption_value=0.075,
            max_absorption_value=0.4,
            p=0.3),
        TimeStretch(leave_length_unchanged=False, min_rate=0.8, max_rate=1.25, p=0.3),
        PitchShift(min_semitones=-2, max_semitones=3, p=0.3)
    ])
    
    for root, _, audio_files in os.walk(input_directory):
        for audio_file in audio_files:
            if audio_file.endswith(".wav"):
                input_path = os.path.join(root, audio_file)
                audio, sample_rate = sf.read(input_path)
                
                for i in range(num_iterations):
                    augmented_audio = augmentations(samples=audio, sample_rate=sample_rate)
                    
                    relative_path = os.path.relpath(input_path, input_directory)
                    output_file_name = os.path.splitext(relative_path)[0] + f"_{i + 1}.wav"
                    output_path = os.path.join(output_directory, output_file_name)
                    
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    sf.write(output_path, augmented_audio, sample_rate)
                    
                    print(f"Augmented {relative_path} (Iteration {i + 1}) and saved to {output_path}")