import starter.src.gaze_estimation as GE
import cv2
import os
model_name = '/home/aryan/gaze_controller/models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002'
ge = GE.Model_Gaze_estimation(model_name, 'CPU')
ge.load_model()
