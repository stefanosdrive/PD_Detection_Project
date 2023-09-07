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