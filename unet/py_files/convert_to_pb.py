# Important librairies.

import keras
from keras import backend as K
from keras.models import load_model

import tensorflow as tf
from tensorflow.compat.v1 import graph_util
from tensorflow.python.framework import graph_io

# -----------------------------------------------------------------------------

def convert_to_pb(name_project):
    """
    Converts Keras model into Tensorflow .pb file.
    
    The string 'name_project' represents the name of the DeepPix Worflow project
    given by the user.
    """
    
    # Define paths.
    path_to_model = '/content/drive/My Drive/unser_project/models/{b}.hdf5'.format(b=name_project)
    path_output = '/content/drive/My Drive/unser_project/'
    
    # Load model.
    model = load_model(path_to_model)

    # Get node names.
    node_names = [node.op.name for node in model.outputs]

    # Get Keras session.
    session = K.get_session()
    
    # Convert Keras variables to Tensorflow constants.
    graph_to_constant = graph_util.convert_variables_to_constants(session, session.graph.as_graph_def(), node_names)
    
    # Write graph as .pb file.
    graph_io.write_graph(graph_to_constant, path_output, name_project + ".pb", as_text=False)
    