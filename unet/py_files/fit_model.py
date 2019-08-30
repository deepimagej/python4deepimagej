# Important librairies

import pickle
import cv2 as cv
import sys
import glob

# -----------------------------------------------------------------------------

# Important py.files
sys.path.append("/content/drive/My Drive/unser_project/py_files")
from model import *
from data_loading import *
from helpers import *
from unet_weights import *
from unet_dm import *

# -----------------------------------------------------------------------------

def fit_model(trainGen, validGen, model_type, model_name, input_size = (256, 256, 1), loss_ = 'binary_crossentropy',
              lr = 1e-4, w_decay = 5e-7, steps = 500, epoch_num = 10, val_steps = 15, save_history = True):
    """
    This function selects a model and fits the given generators with the given arguments.
    Then the history of the model and the model itself are saved.
    
    The generators 'trainGen' and 'validGen' represent the training and validation
    generators to fit the model.
    
    The string 'model_type' refers to the type of U-Net to use.
    
    The string 'model_name' refers to the name with which the model shall be saved.
    
    The tuple 'input_size' corresponds to the size of the input images and labels.
    Default value set to (256, 256, 1) (input images size is 256x256).
    
    The string 'loss_' represents the name of the loss that should be used.
    Default value set to 'binary_crossentropy'.
    
    The float 'lr' corresponds to the learning rate value for the training.
    Defaut value set to 1e-4.
    
    The float 'w_decay' corresponds to the weight decay value for the training.
    Default value set to 5e-7.
    
    The integer 'steps' refers to the number of steps between each epoch. This
    number should be big enough to allow for many augmentations.
    Default value set to 500.
    
    The integer 'epoch_num' refers to the number of epochs to be used for the training.
    Default value set to 10.
    
    The integer 'val_steps' refers to the number of steps for validation step of each
    epoch. This number should be equal to the number of validation images. 
    Default value set to 15.
    
    The boolean 'save_history' refers to whether or not the history of the training
    should be saved.
    """
    
    # Load a model.
    if model_type == "unet_simple":
        model = unet(input_size = input_size, loss_ = loss_, learning_rate = lr, weight_decay = w_decay)
        
    elif model_type == "unet_weighted":
        model = unet_weights(input_size = input_size, learning_rate = lr, weight_decay = w_decay)
        
#    elif model_type == "unet_dm":
#        model = unet_distance(learning_rate = lr, weight_decay = w_decay)
        
    else:
        raise RuntimeError("Model type not recognized")
    
    # Callbacks.
    model_checkpoint = ModelCheckpoint('/content/drive/My Drive/unser_project/models/{b}.hdf5'.format(b=model_name), monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto', restore_best_weights=True)
    
    # Fit.
    history = model.fit_generator(trainGen,
                              steps_per_epoch=steps,
                              epochs=epoch_num,
                              callbacks=[model_checkpoint, early_stopping], 
                              validation_data = validGen, 
                              validation_steps = val_steps)
    
    if save_history:
        
        # Saving the history for plotting.
        pickle.dump(history.history, open('/content/drive/My Drive/unser_project/histories/{b}.p'.format(b=model_name), "wb" ))
    
    return None

# -----------------------------------------------------------------------------

def show_predictions(model, name_project, target_size = (256, 256)):
    """
    Shows one image with its ground truth and the prediction of the model (as
    a binary image and a probability map).
    
    The string 'model' corresponds to the type of model used.
    
    The string 'name_project' refers to the name of the DeepPix Worflow project
    given by the user.
    
    The tuple "target_size" is used to specify the final sizes of the images 
    and labels. If the given size does not correspond to original size of the 
    images and labels, the data will be resized with the given size. 
    Default value is set to (256, 256) (image size of 256x256 pixels).
    """
  
    # Path of the test set
    test_path = "/content/drive/My Drive/unser_project/data/processed/" + name_project + "/test/"
    
    # List of files
    list_file = glob.glob(test_path + 'image/*.png')
    
    # Number of files (important for number of leading zeros)
    n_file = len(list_file)
    
    # Count number of digits in n_file. This is important for the number
    # of leading zeros in the name of the images and labels.
    n_digits = len(str(n_file))
    
    # Creates title depending on model type and prepares test generator 
    # depending on model type.
    if model == "unet_simple":
        title = "Simple U-Net"
        testGen = dataGenerator(batch_size = 1, subset = "test", path = test_path)
        mdl = unet(input_size = (256,256,1))
    
    elif model == "unet_weighted":
        title = "Weighted U-Net"
        testGen = weightGen(batch_size = 1, subset = "test", path = test_path)
        mdl = unet_weights(input_size = (256,256,1))       
    
    else:
        raise RuntimeError("Model not recognised.")

    # Loads one image and label.
    img_path = test_path + "image/{b:0" + str(n_digits) + "d}.png"
    lbl_path = test_path + "label/{b:0" + str(n_digits) + "d}.png"
    
    img = cv.imread(img_path.format(b=0))
    label = cv.imread(lbl_path.format(b=0))
    
    # Resizes to target size.
    img = cv.resize(img, target_size)
    label = cv.resize(label, target_size)
      
    # Load model and perform predictions.
    mdl.load_weights('/content/drive/My Drive/unser_project/models/{b}.hdf5'.format(b=name_project))
    prediction = mdl.predict_generator(testGen, 2, verbose=1, workers=1)
    
    # Binarizes one prediction.
    pred_binarized = convertLabel(prediction[0])
          
    # Perform plot.
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=((15,15)))

    ax[0,0].grid(False)
    ax[0,1].grid(False)
    ax[1,0].grid(False)
    ax[1,1].grid(False)
    
    ax[0,0].imshow(img, cmap = 'gray', aspect="auto")
    ax[0,1].imshow(label, cmap = 'gray', aspect="auto")
    ax[1,0].imshow(pred_binarized, cmap = 'gray', aspect="auto")
    ax[1,1].imshow(prediction[0,...,0], cmap = 'gray', aspect="auto", vmin=0, vmax=1)
    
    ax[0,0].set_title("Input", fontsize = 17.5)
    ax[0,1].set_title("Ground truth", fontsize = 17.5)
    ax[1,0].set_title(title + " - Binarized", fontsize = 17.5)
    ax[1,1].set_title(title + " - Probability map", fontsize = 17.5)
