# Development of a Speech Application for the Identification of Parkinson's Disease
# This program works with a database consisting of HC (healthy control) and 
# PD (Parkinson's disease patient) audio files.
# The purpose of this function is to accordingly split the used data set intro training and validation sets.

# Import necessary libraries
import os
import shutil
import splitfolders

def split_data(input_folder, output_folder, seed, ratio):
    # Clear the contents of the output folder
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder, ignore_errors=True)
    
    splitfolders.ratio(input_folder, output=output_folder,
                       seed=seed, ratio=ratio,
                       group_prefix=None)