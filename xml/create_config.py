"""
DeepImageJ

https://deepimagej.github.io/deepimagej/

Conditions of use:

DeepImageJ is an open source software (OSS): you can redistribute it and/or modify it under 
the terms of the BSD 2-Clause License.

In addition, we strongly encourage you to include adequate citations and acknowledgments 
whenever you present or publish results that are based on it.
 
DeepImageJ is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
 
You should have received a copy of the BSD 2-Clause License along with DeepImageJ. 
If not, see <https://opensource.org/licenses/bsd-license.php>.


Reference: 
    
DeepImageJ: A user-friendly plugin to run deep learning models in ImageJ
E. Gomez-de-Mariscal, C. Garcia-Lopez-de-Haro, L. Donati, M. Unser, A. Munoz-Barrutia, D. Sage. 
Submitted 2019.

Bioengineering and Aerospace Engineering Department, Universidad Carlos III de Madrid, Spain
Biomedical Imaging Group, Ecole polytechnique federale de Lausanne (EPFL), Switzerland

Corresponding authors: mamunozb@ing.uc3m.es, daniel.sage@epfl.ch
 
Copyright 2019. Universidad Carlos III, Madrid, Spain and EPFL, Lausanne, Switzerland.

"""

import os
import xml.etree.ElementTree as ET
import time
import numpy as np

import xml.etree.ElementTree as ET

"""
Download the template from this link: 
    
https://raw.githubusercontent.com/esgomezm/python4deepimagej/yaml/yaml/config_template.xml

"""

def write_config(model, test_image, config_path):
    tree = ET.parse('config_template.xml')
    root = tree.getroot()
    
    # ModelInformation
    root[0][0].text = model.Name
    root[0][1].text = model.Authors
    root[0][2].text = model.URL
    root[0][3].text = model.Credits
    root[0][4].text = model.Version
    root[0][5].text = model.Date
    root[0][6].text = model.References
    
    # ModelTest
    root[1][0].text = test_image.Input_shape
    root[1][1].text = test_image.Output_shape
    root[1][2].text = test_image.MemoryPeak
    root[1][3].text = test_image.Runtime
    root[1][4].text = test_image.PixelSize
    
    # ModelTag
    root[2][0].text = 'tf.saved_model.tag_constants.SERVING'
    root[2][1].text = 'tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY'
    root[2][2].text = model.InputTensorDimensions
    root[2][3].text = '1'
    root[2][4].text = 'input'
    root[2][5].text = model.InputOrganization0
    root[2][6].text = '1'
    root[2][7].text = 'output'
    root[2][8].text = model.OutputOrganization0
    root[2][9].text = model.Channels
    root[2][10].text = model.FixedPatch
    root[2][11].text = model.MinimumSize
    root[2][12].text = model.PatchSize
    root[2][13].text = 'true'
    root[2][14].text = model.Padding
    root[2][15].text = 'preprocessing.txt'
    root[2][16].text = 'postprocessing.txt'
    root[2][17].text = '1'
    
    try:
        tree.write(os.path.join(config_path,'config.xml'),encoding="UTF-8",xml_declaration=True, )
    except:
        print("The directory {} does not exist.".format(config_path))

class Model:    
    def __init__(self, tf_model):
        # ModelInformation
        self.Name = 'null'
        self.Authors = 'null'
        self.URL = 'null'
        self.Credits = 'null'
        self.Version = 'null'
        self.References = 'null'
        self.Date = time.ctime()

        # ModelTag
        self.MinimumSize = '32'

        input_dim = tf_model.input_shape
        output_dim = tf_model.output_shape
        if input_dim[2] is None:
            self.FixedPatch = 'false'
            self.PatchSize = self.MinimumSize
            if input_dim[-1] is None:
              self.InputOrganization0 = 'NCHW'
              self.Channels = np.str(input_dim[1])
            else:
              self.InputOrganization0 = 'NHWC'
              self.Channels = np.str(input_dim[-1])
            
            if output_dim[-1] is None:
              self.OutputOrganization0 = 'NCHW'    
            else:
              self.OutputOrganization0 = 'NHWC'
        else:
            self.FixedPatch = 'true'
            self.PatchSize = np.str(input_dim[2])
            if input_dim[-1] < input_dim[-2] and input_dim[-1] < input_dim[-3]:
              self.InputOrganization0 = 'NHWC'
              self.Channels = np.str(input_dim[-1])
            else:
              self.InputOrganization0 = 'NCHW'
              self.Channels = np.str(input_dim[1])

            if output_dim[-1] < output_dim[-2] and output_dim[-1] < output_dim[-3]:
              self.OutputOrganization0 = 'NHWC'
            else:
              self.OutputOrganization0 = 'NCHW'
              
        input_dim = np.str(input_dim)
        input_dim = input_dim.replace('(', ',')
        input_dim = input_dim.replace(')', ',')
        input_dim = input_dim.replace('None', '-1')
        input_dim = input_dim.replace(' ', "")
        self.InputTensorDimensions = input_dim
        self.Padding = np.str(self._pixel_half_receptive_field(tf_model))
        
    def _pixel_half_receptive_field(self, tf_model):
        input_shape = tf_model.input_shape
        
        if self.FixedPatch == 'false':
          min_size = 10*np.int(self.MinimumSize)
          if self.InputOrganization0 == 'NHWC':
            null_im = np.zeros((1, min_size, min_size, input_shape[-1]))
          else:
            null_im = np.zeros((1, input_shape[1], min_size, min_size))
        else:
          null_im = np.zeros((1, input_shape[1:]))
          min_size = np.int(self.MinimumSize)

        point_im = np.zeros_like(null_im)
        min_size = np.int(min_size/2)
        if self.InputOrganization0 == 'NHWC':
            point_im[0,min_size,min_size] = 1
        else:
            point_im[0,:,min_size,min_size] = 1
        result_unit = tf_model.predict(np.concatenate((null_im, point_im)))
        D = np.abs(result_unit[0]-result_unit[1])>0
        if self.InputOrganization0 == 'NHWC':
            D = D[:,:,0]
        else:
            D = D[0,:,:]
        ind = np.where(D[:min_size,:min_size]==1)
        halo = np.min(ind[1])
        halo = min_size-halo+1
        return halo
    
    # ModelInformation    
    def add_name(self, text):
        self.Name = text
        
    def add_author(self, text):
        self.Authors = text
        
    def add_url(self, text):
        self.URL = text
        
    def add_credits(self, text):
        self.Credits = text
        
    def add_version(self, text):
        self.Version = text
        
    def add_date(self, text):
        self.Date = text
        
    def add_references(self, text):
        self.References = text
    
    # ModelTag
    def add_channels(self, text):
        self.Channels = text
    
    def add_min_size(self, text):
        self.MinimumSize = text
        
    
class TestImage:
    
    def __init__(self, input_im, output_im, pixel_size):
        """
        pixel size must be given in microns
        """
        self.Input_shape = '{0}x{1}'.format(input_im.shape[0], input_im.shape[1])
        self.Output_shape = '{0}x{1}'.format(output_im.shape[0], output_im.shape[1])
        self.MemoryPeak = ''
        self.Runtime = ''
        self.PixelSize = '{0}µmx{1}µm'.format(pixel_size, pixel_size)



"""
Example:
    
test_info = TestImage(test_img, output_img,1.6)
model_info = Model(model)
model_info.add_author('E. Gomez-de-Mariscal, C. Garcia-Lopez-de-Haro, L. Donati, M. Unser, A. Munoz-Barrutia, D. Sage.')
model_info.add_name('2D multitask U-Net for cell segmentation')
model_info.add_credits('Copyright 2019. Universidad Carlos III, Madrid, Spain and EPFL, Lausanne, Switzerland.')

write_config(model_info, test_info, "DeepImageJ-model")
    
"""