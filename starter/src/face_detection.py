'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys
import numpy as np
import time


class Model_Face_detection:
    '''
    Class for the Face Detection Model.
    '''

    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_name = model_name
        self.model_weights = model_name+'.bin'
        self.model_structure = model_name+'.xml'
        self.device = device
        try:
            self.model = IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError(
                "Could not Initialise the network. Have you enterred the correct model path?")
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        core = IECore()
        print('in load model')
        self.net = core.load_network(
            network=self.model, device_name=self.device, num_requests=1)
        print('after load model')

    def predict(self, image, initial_dims):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        processed_image = self.preprocess_input(image)

        self.net.start_async(request_id=0, inputs={
                             self.input_name: processed_image})

        while True:
            status = self.net.requests[0].wait(-1)
            if status == 0:
                break
            else:
                time.sleep(1)
        # print('printing result')

        result = self.net.requests[0].outputs[self.output_name]
        # print(result[0])
        # result is of shape 1x1xNx7
        # fopen = open("testout.txt","w")
        # fopen.write(str(result))
        # fopen.write("\n"+str(result.shape))
        # fopen.write("\n"+"result[0][0] \n"+str(result[0][0]))
        # fopen.write("\n"+"result[0][0][0] \n"+str(result[0][0][0]))
        # fopen.close()
        self.threshold = 0.5
        for box in result[0][0]:

            conf = box[2]
            if conf >= self.threshold:

                coords = self.preprocess_output(box, initial_dims)
                image = self.draw_outputs(coords, image)
                cropped_image = self.crop_image(image, coords)
                # cv2.imshow("cropped image", cropped_image)
        return cropped_image

    # def check_model(self): # todo fill this method
        # raise NotImplementedError

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        net_input_shape = self.input_shape
        p_image = cv2.resize(image, (net_input_shape[3], net_input_shape[2]))
        p_image = p_image.transpose((2, 0, 1))
        p_image = p_image.reshape(1, *p_image.shape)
        return p_image

    def preprocess_output(self, box, initial_dims):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        coords = [box[3] * initial_dims[1],
                  box[4] * initial_dims[0],
                  box[5] * initial_dims[1],
                  box[6] * initial_dims[0]]
        return coords
    # raise NotImplementedError

    def draw_outputs(self, coords, image):

        frame = image
        xmin = int(coords[0])
        ymin = int(coords[1])
        xmax = int(coords[2])
        ymax = int(coords[3])

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255, 1))

        return frame

    def crop_image(self, image, coords):
        xmin = int(coords[0])
        ymin = int(coords[1])
        xmax = int(coords[2])
        ymax = int(coords[3])
        cropped_image = image[ymin:ymax, xmin:xmax]
        return cropped_image


    def reshape_after_crop(self,cropped_image, width, height):
        dims = (width, height)
        resized_cropped_image = cv2.resize(cropped_image, dims, interpolation = cv2.INTER_AREA) 
        return resized_cropped_image