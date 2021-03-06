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


class Model_Facial_landmarks_de:
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
        self.net = core.load_network(
            network=self.model, device_name=self.device, num_requests=1)
        print('model loaded')

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

        result = self.net.requests[0].outputs[self.output_name]

        coords = []
        for i in range(0, 10):
            coords.append(result[0][i][0][0])
        real_face_coords = self.preprocess_output(coords, initial_dims)
        # points_drawn_image = self.draw_facial_points(real_face_coords, image)
        return real_face_coords

    def check_model(self):
        raise NotImplementedError

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

    def preprocess_output(self, coords, initial_dims):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        real_face_coords = []
        for i in range(0, 5):
            real_face_coords.append(
                coords[i*2] * initial_dims[1])  # multiply by width
            real_face_coords.append(
                coords[i*2+1] * initial_dims[0])  # multiply by height
        #  print(real_face_coords)
        return real_face_coords

    def draw_facial_points(self, coords, image):
        drawn_image = image
        n = int(len(coords)/2)
        for i in range(0, n):  # draw all points on face
            drawn_image = self.draw_point(
                coords[2*i], coords[2*i+1], drawn_image)
        return drawn_image

    def draw_point(self, x, y, image):
        x = int(x)
        y = int(y)
        draw_image = cv2.circle(image, (x, y), radius=2,
                                color=(0, 0, 255), thickness=2)
        return draw_image
