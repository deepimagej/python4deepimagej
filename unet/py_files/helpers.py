# Important librairies.

from PIL import Image
import glob
import numpy as np
import re
import matplotlib.pyplot as plt
from skimage import measure
import scipy.ndimage
import os
import cv2
import pickle
import copy
from tifffile import imsave

# -----------------------------------------------------------------------------

def prepare_standardplot(title, xlabel):
    """
    Prepares the layout and axis for the plotting of the history from the training.
    
    The string 'title' refers to the title of the plot.
    
    The string 'xlabel' refers to the name of the x-axis.
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(title)
    
    ax1.set_ylabel('Binary cross-entropy')
    ax1.set_xlabel(xlabel)
    ax1.set_yscale('log')
    
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel(xlabel)
    
    return fig, ax1, ax2

# -----------------------------------------------------------------------------

def finalize_standardplot(fig, ax1, ax2):
    """
    Finalizes the layout of the plotting of the history from the training.
    
    The variable 'fig' refers to the created figure of the plot.
    
    The variables 'ax1' and 'ax2' refer to the axes of the plot.
    """
    
    ax1handles, ax1labels = ax1.get_legend_handles_labels()
    if len(ax1labels) > 0:
        ax1.legend(ax1handles, ax1labels)
        
    ax2handles, ax2labels = ax2.get_legend_handles_labels()
    if len(ax2labels) > 0:
        ax2.legend(ax2handles, ax2labels)
        
    fig.tight_layout()
    
    plt.subplots_adjust(top=0.9)
    
# -----------------------------------------------------------------------------

def plot_history(history, title):
    """
    Plots the history from the training of a model. More precisely, this function
    plots the training loss, the validation loss, the training accuracy and 
    the validation accuracy of a model training.
    
    The variable 'history' refers to the history file that was saved after
    the training of the model.
    
    The string 'title' represents the title that the plot will have.
    """
    
    if title == "unet_simple":
        title = "Simple U-Net"
    
    elif title == "unet_weighted":
        title = "Weighted U-Net"
    
    fig, ax1, ax2 = prepare_standardplot(title, 'Epoch')
    
    ax1.plot(history['loss'], label = "Training")
    ax1.plot(history['val_loss'], label = "Validation")
    
    ax2.plot(history['acc'], label = "Training")
    ax2.plot(history['val_acc'], label = "Validation")
    
    finalize_standardplot(fig, ax1, ax2)
    
    return fig

# -----------------------------------------------------------------------------

def natural_keys(text):
    """
    Sorts the filelist in a more "human" order.
    
    The variable 'text' represents a file list that would be imported with
    the glob library.
    """

    def atoi(text):
        return int(text) if text.isdigit() else text
    
    return [atoi(c) for c in re.split('(\d+)', text)]

# -----------------------------------------------------------------------------

def load_data(path_images, path_labels):
    """
    Loads and returns images and labels.
    
    The variables 'path_images' and 'path_labels' refer to the paths of the
    folders containing the images and labels, respectively.
    """
    
    # Creates a list of file names in the data directory.
    filelist = glob.glob(path_images)
    filelist.sort(key=natural_keys)
    
    # Loads all data images in a list.
    data = [Image.open(fname) for fname in filelist]
    
    # Creates a list of file names in the labels directory.
    filelist = glob.glob(path_labels)
    filelist.sort(key=natural_keys)
    
    # Loads all labels images in a list.
    labels = [Image.open(fname) for fname in filelist]

    return data, labels

# -----------------------------------------------------------------------------
    
def check_binary(labels):
    """
    Checks if the given labels are binary or not.
    
    The variable "labels" correspond to a list of label images.
    """
    
    # Initialize output variable.
    binary = True
    
    # Check every label.
    for k in range(len(labels)):
        
        # Number of unique values (should be = 2 for binary labels or > 2 for 
        # categorical or non-binary data).
        n_unique = len(np.unique(np.array(labels[k])))
    
        if n_unique > 2:
            binary = False
        
        # Raise exception if labels are constant images or not recognised.
        elif n_unique < 2:
            raise RuntimeError("Labels are neither binary or categorical.")
        
    return binary

# -----------------------------------------------------------------------------
    
def make_binary(labels):
    """
    Makes the given labels binary.
    
    The variable "labels" correspond to a list of label images.
    """
    
    # For each label, convert the image to a numpy array, binarizes the array
    # and converts back the array to an image.
    for i in range(len(labels)):
        tmp = np.array(labels[i])
        tmp[tmp > 0] = 255
        tmp[tmp == 0] = 0
        tmp = tmp.astype('uint8')
        tmp = Image.fromarray(tmp, 'L')
        labels[i] = tmp
        
    return labels

# -----------------------------------------------------------------------------

def save_data(data, labels, path):
    """
    Save images and labels.
    
    The variables 'data' and 'labels' refer to the processed images and labels.
    
    The string 'path' corresponds to the path where the images and labels will
    be saved.
    """
    
    # Number of images.
    n_data = len(data)
    
    # Count number of digits in n_data. This is important for the number
    # of leading zeros in the name of the images and labels.
    n_digits = len(str(n_data))
    
    # These represent the paths for the final label and images with the right
    # number of leading zeros given by n_digits.
    direc_d = path + "image/{b:0" + str(n_digits) + "d}.png"
    direc_l = path + "label/{b:0" + str(n_digits) + "d}.png"
    
    # Saves data and labels in the right folder.
    for i in range(len(data)):
        data[i].save(direc_d.format(b=i))
        labels[i].save(direc_l.format(b=i))
        
    return None

# -----------------------------------------------------------------------------

def split_data(X, y, ratio=0.8, seed=1):
    """
    The split_data function will shuffle data randomly as well as return
    a split data set that are individual for training and testing purposes.
    
    The input 'X' is a list of images. 
    
    The input 'y' is a list of images with each image corresponding to the label 
    of the corresponding sample in X. 
    
    The 'ratio' variable is a float that sets the train set fraction of
    the entire dataset to this ratio and keeps the other part for test set.
    Default value set to 0.8.
    
    The 'seed' variable represents the seed value for the randomization of the
    process. Default value set to 1.
    """
        
    # Set seed.
    np.random.seed(seed)

    # Perform shuffling.
    idx_shuffled = np.random.permutation(len(y))
    
    # Return shuffled X and y.
    X_shuff = [X[i] for i in idx_shuffled]
    y_shuff = [y[i] for i in idx_shuffled]

    # Cut the data set into train and test.
    train_num = round(len(y) * ratio)
    X_train = X_shuff[:train_num]
    y_train = y_shuff[:train_num]
    X_test = X_shuff[train_num:]
    y_test = y_shuff[train_num:]

    return X_train, y_train, X_test, y_test

# -----------------------------------------------------------------------------

def convertLabel(lab, threshold = 0.5):
    """
    Converts the given label probability maps to a binary images using a specific
    threshold.
    
    The numpy array 'lab' correspond to label probability maps.
    
    The float 'threshold' corresponds to the threshold at which we binarize
    the probability map. Default value set to 0.5.
    """
    
    # Converts the labels into boolean values using a threshold.
    label = lab[...,0] > threshold
    
    # Converts the boolean values into 0 and 1.
    label = label.astype(int)
    
    # Converts the labels to have values 0 and 255.
    label[label == 1] = 255
    
    return label

# -----------------------------------------------------------------------------

def pred_accuracy(y_true, y_pred):
    """
    Computes the prediction accuracy.
    
    The numpy array 'y_true' corresponds to the true label.
    
    The numpy array 'y_pred' corresponds to the predicted label.
    """
    
    # Compares both the predictions and labels.
    compare = (y_true == y_pred)
    
    # Convert the resulting boolean values into 0 and 1.
    compare = compare.astype(int)
    
    # Computes the percentage of correct pixels.
    accuracy = np.sum(compare)/(len(y_true)**2)
    
    return accuracy

# -----------------------------------------------------------------------------

def saveResults(save_path, results, convert = True, threshold = 0.5):
    """
    Save the predicted arrays into a folder.
    
    The string 'save_path' corresponds to the path where the predicted images
    would be saved.
    
    The numpy array 'results' corresponds to the probability maps that were
    predicted with the model.
    
    The boolean 'convert' refers to whether or not the probability maps
    should be converted to binary arrays. Defaut value set to True.
    
    The float 'threshold' corresponds to the threshold at which we binarize
    the probability map. Default value set to 0.5.
    """
    
    # Number of predictions.
    n_result = len(results)
    
    # Count number of digits in n_result. This is important for the number
    # of leading zeros in the name of the predictions.
    n_digits = len(str(n_result))
    
    # These represent the paths for the predictions (binary or not) with the right
    # number of leading zeros given by n_digits.
    if convert:
        # Selects path for data and labels.
        direc_r = save_path + "result/{b:0" + str(n_digits) + "d}.tif"
    else:
        direc_r = save_path + "result_prob/{b:0" + str(n_digits) + "d}.tif"
      
    
    for i, lab in enumerate(results):
        
        if convert:
            # Converts the given label with a threshold.
            label = convertLabel(lab, threshold)
        
        else:
            label = lab[...,0]
            
        label = label.astype('float32')
            
        # Saves the label.
        imsave(direc_r.format(b=i), label)  
        
    return None

# -----------------------------------------------------------------------------

def make_weight_map(label, binary = True, w0 = 10, sigma = 5):
    """
    Generates a weight map in order to make the U-Net learn better the
    borders of cells and distinguish individual cells that are tightly packed.
    These weight maps follow the methodololy of the original U-Net paper.
    
    The variable 'label' corresponds to a label image.
    
    The boolean 'binary' corresponds to whether or not the labels are
    binary. Default value set to True.
    
    The float 'w0' controls for the importance of separating tightly associated
    entities. Defaut value set to 10.
    
    The float 'sigma' represents the standard deviation of the Gaussian used
    for the weight map. Default value set to 5.
    """
    
    # Initialization.
    lab = np.array(label)
    lab_multi = lab
        
    # Get shape of label.
    rows, cols = lab.shape
    
    if binary:
        
        # Converts the label into a binary image with background = 0
        # and cells = 1.
        lab[lab == 255] = 1
        
        
        # Builds w_c which is the class balancing map. In our case, we want cells to have
        # weight 2 as they are more important than background which is assigned weight 1.
        w_c = np.array(lab, dtype=float)
        w_c[w_c == 1] = 1
        w_c[w_c == 0] = 0.5
        
        # Converts the labels to have one class per object (cell).
        lab_multi = measure.label(lab, neighbors = 8, background = 0)
    
    else:
        
        # Converts the label into a binary image with background = 0.
        # and cells = 1.
        lab[lab > 0] = 1
        
        
        # Builds w_c which is the class balancing map. In our case, we want cells to have
        # weight 2 as they are more important than background which is assigned weight 1.
        w_c = np.array(lab, dtype=float)
        w_c[w_c == 1] = 1
        w_c[w_c == 0] = 0.5
        
    components = np.unique(lab_multi)
    
    n_comp = len(components)-1
    
    maps = np.zeros((n_comp, rows, cols))
    
    map_weight = np.zeros((rows, cols))
    
    if n_comp >= 2:
        for i in range(n_comp):
            
            # Only keeps current object.
            tmp = (lab_multi == components[i+1])
            
            # Invert tmp so that it can have the correct distance.
            # transform
            tmp = ~tmp
            
            # For each pixel, computes the distance transform to
            # each object.
            maps[i][:][:] = scipy.ndimage.distance_transform_edt(tmp)
    
        maps = np.sort(maps, axis=0)
        
        # Get distance to the closest object (d1) and the distance to the second
        # object (d2).
        d1 = maps[0][:][:]
        d2 = maps[1][:][:]
        
        map_weight = w0*np.exp(-((d1+d2)**2)/(2*(sigma**2)) ) * (lab==0).astype(int);

    map_weight += w_c
    
    return map_weight

# -----------------------------------------------------------------------------

def do_save_wm(labels, path, binary = True, w0 = 10, sigma = 5):
    """
    Retrieves the label images, applies the weight-map algorithm and save the
    weight maps in a folder.
    
    The variable 'labels' corresponds to given label images.
    
    The string 'path' refers to the path where the weight maps should be saved.
    
    The boolean 'binary' corresponds to whether or not the labels are
    binary. Default value set to True.
    
    The float 'w0' controls for the importance of separating tightly associated
    entities. Default value set to 10.
    
    The float 'sigma' represents the standard deviation of the Gaussian used
    for the weight map. Default value set to 5.
    """
    
    # Copy labels.
    labels_ = copy.deepcopy(labels)
    
    # Perform weight maps.
    for i in range(len(labels_)):
        labels_[i] = make_weight_map(labels[i].copy(), binary, w0, sigma)
    
    maps = np.array(labels_)
    
    n, rows, cols = maps.shape
    
    # Resize correctly the maps so that it can be used in the model.
    maps = maps.reshape((n, rows, cols, 1))
    
    # Count number of digits in n. This is important for the number
    # of leading zeros in the name of the maps.
    n_digits = len(str(n))
    
    # Save path with correct leading zeros.
    path_to_save = path + "weight/{b:0" + str(n_digits) + "d}.npy"
    
    # Saving files as .npy files.
    for i in range(len(labels_)):
            np.save(path_to_save.format(b=i), labels_[i])
        
    return None

# -----------------------------------------------------------------------------

#def make_distance_map(label):
#    """Generates a distance map from labels in order to test distance-map-based
#    U-Net training."""
#    
#    lab = np.array(label)
#    
#    # Converts the label into a binary image with background = 0
#    # and cells = 1.
#    lab[lab == 255] = 1
#    
#    # Applies distance transform
#    output = cv2.distanceTransform(lab, cv2.DIST_C, 3)
#    
#    # Finds minimal cell size
#    size = 0
#    all_dist = np.unique(output)
#    blobbed_lab = measure.label(lab, neighbors = 8, background = 0)
#    number_blobs = np.max(blobbed_lab)
#    for i in all_dist[1:]:
#      tmp = (output >= i).astype(int)
#      blobbed_lab = measure.label(tmp, neighbors = 8, background = 0)
#      if number_blobs <= np.max(blobbed_lab):
#        size = i
#    
#    return output, size
#
## -----------------------------------------------------------------------------
#
#def do_make_dm(path):
#    """Retrieves the label images, applies the distance transform and save the
#    maps in the right folder."""
#    
#    path_to_labels = path + "/label/*.png"
#    
#    # Creates a list of file names in the labels directory
#    filelist = glob.glob(path_to_labels)
#    filelist.sort(key=natural_keys)
#    
#    # Loads all data images in a list
#    labels = [Image.open(fname).resize((256,256)) for fname in filelist]
#    
#    # Copy labels
#    labels_ = labels
#    
#    # Vector of sizes
#    sizes = []
#    
#    # Do maps
#    print("Doing distance maps")
#    for i in range(len(labels_)):
#        labels_[i], size = make_distance_map(labels_[i])
#        sizes.append(size)
#    print("Maps done")
#    
#    min_size = np.min(np.array(sizes))
#    print("Min size : {b}".format(b=min_size))
#    print(sizes)
#    
#    maps = np.array(labels_)
#    
#    maps[maps >= min_size] = min_size
#    
#    n, rows, cols = maps.shape
#    
#    # Makes sure the data is saved with one leading zero.
#    if (n < 100):
#        
#        # Selects path for data and labels 
#        direc_r = path + "/distance/{b:02d}.png"
#      
#    # If we have more than 100 images, we would have 2 leading zeros.
#    # We have 148 images, so there is no point doing other cases.
#    else:
#        
#        # Selects path for data and labels 
#        direc_r = path + "/distance/{b:03d}.png"
#    
#    for i, lab in enumerate(maps):
#        
#        label = lab.astype('uint8')
#        label = Image.fromarray(label, 'L')
#        
#        # Saves the label
#        label.save(direc_r.format(b=i)) 
#        
#    return None
#
## -----------------------------------------------------------------------------
#    
#def make_three_class (label):
#    
#    lab = np.array(label)
#        
#    # Get shape of label
#    rows, cols = lab.shape
#    
#    components = np.unique(lab)
#    
#    n_comp = len(components)-1
#    
#    output = np.zeros((rows, cols))
#    
#    for i in range(n_comp):
#        
#        # Only keeps current object
#        tmp = (lab == components[i+1]).astype('float32')
#        
#        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
#        
#        eroded_tmp = cv2.erode(tmp, kernel, iterations = 1)
#        
#        border = tmp - eroded_tmp
#        
#        output[border > 0] = 1
#        output[eroded_tmp > 0] = 2
#    
#    output = output.astype('uint8')
#    output = Image.fromarray(output, 'L')
#
#    return output