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

class Model_Gaze_estimation:
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
        input_iterator = iter(self.model.inputs)# check inputs 
        self.input_name1 = next(input_iterator) #head_pose_angles
        self.input_name2 = next(input_iterator) #left_eye_image
        self.input_name3 = next(input_iterator) #right_eye_image
        self.input_shape1 = self.model.inputs[self.input_name1].shape # [1,3]
        self.input_shape2 = self.model.inputs[self.input_name2].shape # [1, 3, 60, 60]
        self.input_shape3 = self.model.inputs[self.input_name3].shape # [1, 3, 60, 60]
        output_iterator = iter(self.model.outputs)
        self.output_name = next(output_iterator) #gaze_vector
        self.output_shape = self.model.outputs[self.output_name].shape #[1, 3]

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


    def predict(self, head_pose_angles, left_eye_image, right_eye_image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        processed_left_eye_image =self.preprocess_input(left_eye_image)
        processed_right_eye_image =self.preprocess_input(right_eye_image)
        head_pose_angles_list = [[head_pose_angles['angle_p_fc'], 
                                head_pose_angles['angle_r_fc'],
                                head_pose_angles['angle_y_fc']]]
        angle_shape = np.array(head_pose_angles_list).shape
        net_input = {}

        # self.net.start_async(requst_id=0, inputs=net_input)
        self.net.start_async(request_id=0, inputs={
                             self.input_name1: head_pose_angles_list,
                             self.input_name2: processed_left_eye_image,
                             self.input_name3: processed_right_eye_image})
        while True:
            status = self.net.requests[0].wait(-1)
            if status == 0:
                break
            else:
                time.sleep(1)
        result = self.net.requests[0].outputs[self.output_name]
        return result
        

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        #reshapes input image
        net_input_shape = self.input_shape2
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
