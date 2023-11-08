# Script and model utils

import os
from glob import glob
from pathlib import Path
from torchvision.io import read_image
from multiprocessing import Pool
import torch
import numpy as np
import sys
from time import time
from sklearn.preprocessing import MinMaxScaler
# from config import 
from multiprocessing import cpu_count

def _scale_image(path: str):
    img = read_image(path)
    img_means = np.zeros(3)
    for i in range(img.shape[0]):
        scaler = MinMaxScaler()
        t_channel = scaler.fit_transform(img[i].numpy())
        img_means[i] = np.mean(t_channel)
        
    return img_means

def calc_norm_parameters(directory: str, processes: int):
    """
    Looks for a train and val folder within directory, pulls all the images from the folder
    and calculates the mean and standard deviation over each of the channels.

    directory (str): The root directory of all the images in the dataset
    processes (int): The number of multiprocessing processes to use for the functional mapping
   
    Returns: a dictionary of channel means, std deviations, and the mean values array.
    """

    if not os.path.exists(directory):
        raise NotADirectoryError(f"The directory passed ('{directory}') is not specified correctly or does not exist. Please pass a new value to the directory argument.")
    elif not os.path.exists(os.path.join(directory, 'train')):
        raise NotADirectoryError(f"There is no 'train' directory in '{directory}'.")
    elif not os.path.exists(os.path.join(directory, 'val')):
        raise NotADirectoryError(f"There is no 'val' directory in '{directory}'.")    
    
    train_dir = os.path.join(directory, 'train')
    val_dir = os.path.join(directory, 'val')  


    train_files = [f for f in glob(train_dir+'/*') if f.endswith('.jpg')]
    val_files = [f for f in glob(val_dir+'/*') if f.endswith('jpg')]

    all_files = []
    all_files.extend(train_files)
    all_files.extend(val_files)

    chunksize = int(round(len(all_files)/6))
    
    mean_array = np.zeros(shape=(len(all_files), 3))

    pool = Pool(processes=processes)
    results = pool.map_async(_scale_image, all_files, chunksize=chunksize)
    mean_array = np.array(results.get())
    
    means = np.mean(mean_array, 0)
    std = np.std(mean_array, 0)

    return {'means': means, 'std': std, 'means_array': mean_array}
 
def collate_fn(batch):
    """
    To handle the data loading as different images may have a different number of objects
    and to handle varying size tensors as well
    :param batch:
    :return:
    """

    return tuple(zip(*batch))

if __name__ == '__main__':
    t0 = time()
    x = calc_norm_parameters('./data/images', processes=cpu_count())
    t1 = time()
    print(x)
    print(t1-t0)