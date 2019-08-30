# Important librairies

import numpy
import sys
import os
import pandas as pd

# -----------------------------------------------------------------------------

# Important py.files
sys.path.append("/content/drive/My Drive/unser_project/py_files")
from helpers import *
from data_loading import *
from fit_model import *

# -----------------------------------------------------------------------------

def read_config(name_project, name_config):
    """
    Reads configuration file from DeepPix Workflow plug-in.
    
    The string 'name_project' refers to the name of the DeepPix Worflow project
    given by the user.
    
    The string 'name_config' refers to the name of the configuration file given
    by the user.
    """
    
    # Builds path to the configuration file.
    path_to_config = '/content/drive/My Drive/unser_project/data/raw/' + name_project + '/' + name_config + '-training-settings.txt'
    
    # Load configuration file
    df = pd.read_table(path_to_config, header = None, delimiter = '=', dtype = str, skiprows = 5)
    
    input_array = []
    
    # Process input dataframe.
    for i in range(df.shape[0]):
        input_array.append(df[1][i][1:])
    
    # The following code allocates the input configurations to variables that
    # will be used for the rest of the program.
    
    label_type = input_array[3]
    
    size = input_array[4]
    target_size = ()
    
    if size == "256x256":
        target_size = (256, 256)
    
    elif size == "512x512":
        target_size = (512, 512)
    
    elif size == "1024x1024":
        target_size = (1024, 1024)
    
    else: 
        raise RuntimeError("Input size unknown")
        
    model = input_array[5]
    model_type = ""
    
    if model == "Simple U-Net":
        model_type = "unet_simple"
    
    elif model == "Weighted U-Net":
        model_type = "unet_weighted"
    
    split_ratio = float(input_array[6])/100
    
    batch_size = int(input_array[7])
    
    learning_rate = float(input_array[8])*1e-5
    
    return label_type, target_size, model_type, split_ratio, batch_size, learning_rate

# -----------------------------------------------------------------------------

def prep_data(name_project, model, label_type, split_ratio, w0 = None, sigma = None):
    """
    Prepares the data by randomizing the images and binarizing them if needed.
    
    The string 'name_project' refers to the name of the DeepPix Worflow project
    given by the user.
    
    The string 'model' refers to the type of model that will be used.
    
    The string 'label_type' corresponds to the type of model used, either
    categorical or binary.
    
    The float 'split_ratio' corresponds to the splitting ratio for the
    training and testing set.
    
    The float 'w0' corresponds to a constant used for the weighted U-Net. 
    Default value set to None.
    
    The float 'sigma' corresponds to a constant used for the weighted U-Net. 
    Default value set to None.
    """
    
    print("Initialization of preparation of data.")
    
    # Constructs useful paths.
    
    # Path for data.
    data_path = "/content/drive/My Drive/unser_project/data/"
    
    # Paths for raw data and labels.
    path_data = data_path + "raw/" + name_project + "/image/*.tif"
    path_labels = data_path + "raw/" + name_project + "/label/*.tif"
    
    # Paths for train and test directories.
    train_path = data_path + "processed/" + name_project + "/train/"
    test_path = data_path + "processed/" + name_project + "/test/"
    
    # Load data and labels.
    print("Loading data and labels.")
    data, labels = load_data(path_data, path_labels)
    print("Loading successful.")
    
    print("Label type check and binarization if needed.")
    # Checks if labels are binary or categorical.
    binary = check_binary(labels)
    
    # Check which model is desired and binarizes labels or not depending on the model.
    if model == "unet_simple":
        
        if not binary:
            labels = make_binary(labels)
    
    elif model == "unet_weighted":
        
        if label_type == "categorical":
            
            if binary:
                raise RuntimeError("Labels are said to be categorical but they are not categorical.")
            
        elif label_type == "binary":
            
            if not binary:
                labels = make_binary(labels)
            
        else:
            raise RuntimeError("Labels are neither categorical or binary.")
    
    else:
        raise RuntimeError("Model type not recognised.")
      
    print("Splitting data")
    X_train, y_train, X_test, y_test = split_data(data, labels, ratio = split_ratio)
    
    if not os.path.exists(train_path + 'image'):
        os.makedirs(train_path + 'image')
    
    if not os.path.exists(train_path + 'label'):
        os.makedirs(train_path + 'label')
    
    if not os.path.exists(test_path + 'image'):
        os.makedirs(test_path + 'image')
    
    if not os.path.exists(test_path + 'label'):
        os.makedirs(test_path + 'label')
    
    if model == "unet_weighted":
        
        if not os.path.exists(train_path + 'weight'):
            os.makedirs(train_path + 'weight')
        
        if not os.path.exists(test_path + 'weight'):
            os.makedirs(test_path + 'weight')
        
        not_connected = True
        
        if label_type == "categorical":
            not_connected = False
        
        print("Constructing weight maps.")
        do_save_wm(y_train, train_path, not_connected = not_connected, w0 = w0, sigma = sigma)
        do_save_wm(y_test, test_path, not_connected = not_connected, w0 = w0, sigma = sigma)
        print("Weight maps achieved")
        
    print("Saving data.")
    save_data(X_train, y_train, train_path)
    save_data(X_test, y_test, test_path)
    print("Preparation of data completed.")