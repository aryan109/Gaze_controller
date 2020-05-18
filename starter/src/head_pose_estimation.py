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


class Model_Head_pose_estimation:
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
        output_iterator = iter(self.model.outputs)
        self.output_name1 = next(output_iterator)
        self.output_name2 = next(output_iterator)
        self.output_name3 = next(output_iterator)
        self.output_shape = self.model.outputs[self.output_name1].shape

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
        all_result = {}
        all_result[self.output_name1] = self.net.requests[0].outputs[self.output_name1][0][0]
        all_result[self.output_name2] = self.net.requests[0].outputs[self.output_name2][0][0]
        all_result[self.output_name3] = self.net.requests[0].outputs[self.output_name3][0][0]
        
        return all_result

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

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        raise NotImplementedError
    def write_on_video(self,frame,all_result):
    
        # font 
        font = cv2.FONT_HERSHEY_SIMPLEX 

        # org 
        org1 = (0, 20)
        org2 = (0, 40)
        org3 = (0, 60) 

        # fontScale 
        fontScale = 0.4

        # red color in BGR 
        color = (0, 0, 255) 

        # Line thickness of 2 px 
        thickness = 1
        s1 = "all_result[{}]: {}".format(self.output_name1,all_result[self.output_name1])
        s2 = "all_result[{}]: {}".format(self.output_name2,all_result[self.output_name2])
        s3 = "all_result[{}]: {}".format(self.output_name3,all_result[self.output_name3])
            
        # print(s)
        frame = cv2.putText(frame, s1, org1, font,  
                   fontScale, color, thickness, cv2.LINE_AA)
        frame = cv2.putText(frame, s2, org2, font,  
                   fontScale, color, thickness, cv2.LINE_AA)
        frame = cv2.putText(frame, s3, org3, font,  
                   fontScale, color, thickness, cv2.LINE_AA)
        return frame



        
    
