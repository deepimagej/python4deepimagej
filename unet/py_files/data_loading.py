# Important librairies.

from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import glob

# -----------------------------------------------------------------------------

# Important py.files.

from helpers import *

# -----------------------------------------------------------------------------

def dataGenerator(path, batch_size = 2, subset = 'train', target_size = (256,256), seed = 1):
    
    """
    Builds generators for the U-Net. The generators can be built for
    training, testing and validation purposes. 
    
    The string "subset" is used to specify which type of data we are dealing 
    with (train, test or validation). Default value is set to 'train'.
    
    The string "path" represents a path that should lead to images and labels 
    folders named 'image' and 'label' respectively.
    
    The tuple "target_size" is used to specify the final sizes of the images 
    and labels after augmentation. If the given size does not correspond to
    original size of the images and labels, the data will be resized with the
    given size. Default value is set to (256, 256) (image size of 256x256 pixels).
    
    The variable seed is needed to ensure that images and labels will be augmented
    together in the right orders. Default value set to 1.
    """
    
    # Builds generator for training set.
    if subset == "train":
        
        # Preprocessing arguments.
        aug_arg = dict(rotation_range = 40,
                    width_shift_range = 0.2,
                    height_shift_range = 0.2,
                    shear_range = 0.2,
                    horizontal_flip = True,
                    vertical_flip = True,
                    fill_mode='nearest')
        
        # Generates tensor images and labels with augmentations provided above.
        image_datagen = ImageDataGenerator(**aug_arg)
        label_datagen = ImageDataGenerator(**aug_arg)
        
        # Generator for images.
        image_generator = image_datagen.flow_from_directory(
            path,
            classes = ['image'],
            class_mode = None,
            color_mode = "grayscale",
            target_size = target_size,
            batch_size = batch_size,
            save_to_dir = None,
            seed = seed,
            shuffle = True)
        
        # Generator for labels.
        label_generator = label_datagen.flow_from_directory(
            path,
            classes = ['label'],
            class_mode = None,
            color_mode = "grayscale",
            target_size = target_size,
            batch_size = batch_size,
            save_to_dir = None,
            seed = seed,
            shuffle = True)
        
        # Builds generator for the training set.
        train_generator = zip(image_generator, label_generator)

        for (img,label) in train_generator:
            img, label = adjustData(img, label = label)
            
            yield (img, label)
            
    # Builds generator for validation set.
    elif subset == "validation":
        
        # Generates tensor images and labels with no augmentations
        # (validation set should not have any augmentation and does not
        # have to be shuffled).
        image_datagen = ImageDataGenerator()
        label_datagen = ImageDataGenerator()
        
        # Generator for images.
        image_generator = image_datagen.flow_from_directory(
            path,
            classes = ['image'],
            class_mode = None,
            color_mode = "grayscale",
            target_size = target_size,
            batch_size = batch_size,
            save_to_dir = None,
            seed = seed,
            shuffle = False)
        
        # Generator for labels.
        label_generator = label_datagen.flow_from_directory(
            path,
            classes = ['label'],
            class_mode = None,
            color_mode = "grayscale",
            target_size = target_size,
            batch_size = batch_size,
            save_to_dir = None,
            seed = seed,
            shuffle = False)
        
        # Builds generator for the validation set.
        validation_generator = zip(image_generator, label_generator)
        
        for (img,label) in validation_generator:
            img, label = adjustData(img, label = label)
            
            yield (img, label)
    
    # Builds generator for testing set.      
    elif subset == "test":
        
        # Generates tensor images only with no augmentations (testing data
        # does not have to have labels and we do not shuffle the data
        # as it is not necessary).
        image_datagen = ImageDataGenerator()
        
        # Generator for images.
        image_generator = image_datagen.flow_from_directory(
            path,
            classes = ['image'],
            class_mode = None,
            color_mode = "grayscale",
            target_size = target_size,
            batch_size = batch_size,
            save_to_dir = None,
            seed = seed,
            shuffle = False)
        
        # Builds generator for the testing set.
        for img in image_generator:
            
            img = adjustData(img, False)
            
            yield img
    
    else:
        raise RuntimeError("Subset name not recognized")
            
# -----------------------------------------------------------------------------
            
def weightGen(path, batch_size = 2, subset = 'train', target_size = (256,256), seed = 1):
    
    """
    Builds generators for the weighted U-Net. The generators are built the
    same way as the dataGenerator function, only weight-maps are combined
    with the images. The generators can be built for  training and 
    validation purposes. 
    
    The string "subset" is used to specify which type of data we are dealing 
    with (train, test or validation). Default value is set to 'train'.
    
    The string "path" represents a path that should lead to images and labels 
    folders named 'image' and 'label' respectively.
    
    The tuple "target_size" is used to specify the final sizes of the images 
    and labels after augmentation. If the given size does not correspond to
    original size of the images and labels, the data will be resized with the
    given size. Default value is set to (256, 256) (image size of 256x256 pixels).
    
    The variable seed is needed to ensure that images and labels will be augmented
    together in the right orders. Default value set to 1.
    """
    
    # Builds generator for training set.
    if subset == "train":
        
        # Preprocessing arguments.
        aug_arg = dict(rotation_range = 40,
                    width_shift_range = 0.2,
                    height_shift_range = 0.2,
                    shear_range = 0.2,
                    horizontal_flip = True,
                    vertical_flip = True,
                    fill_mode='nearest')
        
        # Generates tensor images, weight-maps and labels with augmentations provided above.
        image_datagen = ImageDataGenerator(**aug_arg)
        label_datagen = ImageDataGenerator(**aug_arg)
        weight_datagen = ImageDataGenerator(**aug_arg)
        
        # Generator for images.
        image_generator = image_datagen.flow_from_directory(
            path,
            classes = ['image'],
            class_mode = None,
            color_mode = 'grayscale',
            target_size = target_size,
            batch_size = batch_size,
            save_to_dir = None,
            seed = seed,
            shuffle = True)
        
        # Generator for labels.
        label_generator = label_datagen.flow_from_directory(
            path,
            classes = ['label'],
            class_mode = None,
            color_mode = 'grayscale',
            target_size = target_size,
            batch_size = batch_size,
            save_to_dir = None,
            seed = seed,
            shuffle = True)
        
        # Retrieve weight-maps.
        filelist = glob.glob(path + "/weight/*.npy")
        filelist.sort(key=natural_keys)
        
        # Loads all weight-map images in a list.
        weights = [np.load(fname) for fname in filelist]
        weights = np.array(weights)
        weights = weights.reshape((len(weights),256,256,1))
        
        # Creates the weight generator.
        weight_generator = weight_datagen.flow(
                x = weights, 
                y = None,
                batch_size = batch_size,
                seed = seed)
        
        # Builds generator for the training set.
        train_generator = zip(image_generator, label_generator, weight_generator)

        for (img, label, weight) in train_generator:
            img, label = adjustData(img, label = label)
            
            # This is the final generator.
            yield ([img, weight], label)
            
    elif subset == "validation":
        
        # Generates tensor images, weight maps and labels with no augmentations 
        # and shuffling (since we are in the test set).
        image_datagen = ImageDataGenerator()
        label_datagen = ImageDataGenerator()
        weight_datagen = ImageDataGenerator()
        
        # Generator for images.
        image_generator = image_datagen.flow_from_directory(
            path,
            classes = ['image'],
            class_mode = None,
            color_mode = 'grayscale',
            target_size = target_size,
            batch_size = batch_size,
            save_to_dir = None,
            seed = seed,
            shuffle = False)
        
        # Generator for labels.
        label_generator = label_datagen.flow_from_directory(
            path,
            classes = ['label'],
            class_mode = None,
            color_mode = 'grayscale',
            target_size = target_size,
            batch_size = batch_size,
            save_to_dir = None,
            seed = seed,
            shuffle = False)
        
        # Retrieve weight maps.
        filelist = glob.glob(path + "/weight/*.npy")
        filelist.sort(key=natural_keys)
        
        # Loads all weight map images in a list.
        weights = [np.load(fname) for fname in filelist]
        weights = np.array(weights)
        weights = weights.reshape((len(weights),256,256,1))
        
        # Creates the weight generator.
        weight_generator = weight_datagen.flow(
                x = weights, 
                y = None,
                batch_size = batch_size,
                seed = seed,
                shuffle = False)
        
        # Builds generator for the test set.
        test_generator = zip(image_generator, label_generator, weight_generator)

        for (img, label, weight) in test_generator:
            img, label = adjustData(img, label = label)
            
            # This is the final generator.
            yield ([img, weight], label)
    
    else:
        raise RuntimeError("Subset name not recognized")
    
# -----------------------------------------------------------------------------
            
def adjustData(image, adjust_lab = True, dist = False, label = None):
    """
    Normalizes the data such that images are in the interval [0,1] and labels 
    are binary values in {0,1}. This step is important as augmentations with
    Keras' ImageDataGenerator() will change the pixel values of the images
    and labels and most notably images will not be normalized anymore and labels
    will not be binary anymore.
    
    The numpy array 'image' represents the input image.
    
    The numpy array 'label' represents the label image. If adjust_lab is set to
    False, label should be set to None. Default value set to None.
    
    The boolean value 'adjust_lab' specifies if we need to process labels or not 
    for training an validation purposes. Default value set to True.
    """
    
    # Checks if the images are already between 0 and 1, otherwise
    # does the normalization.
    if(np.max(image) > 1):
        image = image / 255
    
    if adjust_lab:
        
        # Checks if the labels are already binary, otherwise
        # does the binarization.
        if (np.max(label) > 1):
            label = label / 255
            label[label > 0.5] = 1
            label[label <= 0.5] = 0
        
        return (image, label)
    
    if dist:
        return (image, label)
    
    else:
        return image
    
# -----------------------------------------------------------------------------

def loadGenerator(name_project, model_type, batch_size = 2, target_size = (256,256)):
    """
    Loads generators for the training of a model automatically.
    
    The string 'name_project' represents the name of the DeepPix Worflow project
    given by the user.
    
    The string 'model_type' represents the type of model desired by the user.
    
    The integer value 'batch_size' represents the size of the batches for the
    training. Default value set to 2.
    
    The tuple "target_size" is used to specify the final sizes of the images 
    and labels after augmentation. If the given size does not correspond to
    original size of the images and labels, the data will be resized with the
    given size. Default value is set to (256, 256) (image size of 256x256 pixels).
    """
    
    # Path to data folder.
    data_path = "/content/drive/My Drive/unser_project/data/processed/"
    
    # Paths for train and test folders.
    train_path = data_path + name_project + "/train/"
    test_path = data_path + name_project + "/test/"
    
    # Create generators depending on model type.
    if model_type == "unet_simple":
        trainGen = dataGenerator(path = train_path, batch_size = batch_size, subset = "train", target_size = target_size)
        validGen = dataGenerator(path = test_path, batch_size = batch_size, subset = "validation", target_size = target_size)
    
    elif model_type == "unet_weighted":
        trainGen = weightGen(path = train_path, batch_size = batch_size, subset = "train", target_size = target_size)
        validGen = weightGen(path = test_path, batch_size = batch_size, subset = "validation", target_size = target_size)
    
    else:
        raise RuntimeError("Model not recognised.")
    
    return trainGen, validGen

# -----------------------------------------------------------------------------
    
#def distanceGen(path, n_classes, batch_size = 2, subset = 'train', image_folder = 'image', label_folder = 'distance', 
#                  image_col = "grayscale", label_col = "grayscale", target_size = (256,256), seed = 1):
#    
#    """Builds generators for the weighted U-Net. The generators are built 
#    the same way as the dataGenerator function, only the weights are combined 
#    with the images."""
#    
#    # Builds generator for training set
#    if subset == "train":
#        
#        # Preprocessing arguments
#        aug_arg = dict(rotation_range = 40,
#                    width_shift_range = 0.2,
#                    height_shift_range = 0.2,
#                    shear_range = 0.2,
#                    horizontal_flip = True,
#                    vertical_flip = True,
#                    fill_mode='nearest')
#        
#        # Generates tensor images and labels with augmentations provided above
#        image_datagen = ImageDataGenerator(**aug_arg)
#        label_datagen = ImageDataGenerator(**aug_arg)
#        
#        # Generator for images
#        image_generator = image_datagen.flow_from_directory(
#            path,
#            classes = [image_folder],
#            class_mode = None,
#            color_mode = 'grayscale',
#            target_size = target_size,
#            batch_size = batch_size,
#            save_to_dir = None,
#            seed = seed,
#            shuffle = True)
#        
#        # Generator for labels
#        label_generator = label_datagen.flow_from_directory(
#            path,
#            classes = [label_folder],
#            class_mode = 'categorical',
#            color_mode = 'rgb',
#            target_size = target_size,
#            batch_size = batch_size,
#            save_to_dir = None,
#            seed = seed,
#            shuffle = True)
#        
#        # Builds generator for the training set
#        train_generator = zip(image_generator, label_generator)
#        
#        for (img, label) in train_generator:
#            img, label = adjustData(img, False, True, label = label)
#            
#            print(label)            
#            # This is the final generator
#            yield (img, label)
#            
#    elif subset == "test":
#        
#        # Generates tensor images and labels with no augmnetations and shuffling
#        # (since we are in the test set)
#        image_datagen = ImageDataGenerator()
#        label_datagen = ImageDataGenerator()
#        
#        # Generator for images
#        image_generator = image_datagen.flow_from_directory(
#            path,
#            classes = [image_folder],
#            class_mode = None,
#            color_mode = 'grayscale',
#            target_size = target_size,
#            batch_size = batch_size,
#            save_to_dir = None,
#            seed = seed,
#            shuffle = False)
#        
#        # Generator for labels
#        label_generator = label_datagen.flow_from_directory(
#            path,
#            classes = [label_folder],
#            class_mode = 'categorical',
#            color_mode = 'rgb',
#            target_size = target_size,
#            batch_size = batch_size,
#            save_to_dir = None,
#            seed = seed,
#            shuffle = False)
#        
#        # Builds generator for the test set
#        test_generator = zip(image_generator, label_generator)
#        
#        for (img, label) in test_generator:
#            img, label = adjustData(img, False, True, label = label)
#                        
#            # This is the final generator
#            yield (img, label)
#    
#    else:
#        print("Subset name not recognized")
#        return None
    
# -----------------------------------------------------------------------------
    
#def label_to_cat(label, n_classes):
#    
#    label = np.rint(label / (255 / (n_classes - 1)))
#    
#    n, rows, cols, _ = label.shape
#    output = np.zeros((n, rows, cols, n_classes))
#    
#    for i in range(n_classes):
#        tmp = (label[...,0] == i).astype(int)
#        output[...,i] = tmp
#    
#    output = np.reshape(output, (n, rows*cols, n_classes))
#    
#    return output