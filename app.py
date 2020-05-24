import starter.src.face_detection as FD
import starter.src.facial_landmarks_detection as FLD
import starter.src.head_pose_estimation as HPE
import starter.src.gaze_estimation as GME
import starter.src.mouse_controller as MC
import cv2
import os
import argparse

def resize_image(image, width, height):
        dims = (width, height)
        resized_image = cv2.resize(image, dims, interpolation = cv2.INTER_AREA) 
        return resized_image

def crop_image(image, coords):
        xmin = int(coords[0])
        ymin = int(coords[1])
        xmax = int(coords[2])
        ymax = int(coords[3])
        cropped_image = image[ymin:ymax, xmin:xmax]
        return cropped_image

def generate_rectangle_coordinates_from_midpoint(x, y, delta, maxlim):
    xmin = max(x - delta,0)
    ymin = max(y - delta,0)
    xmax = min(x + delta, maxlim)
    ymax = min(y + delta, maxlim)
    rect_coords = [xmin, ymin, xmax, ymax]
    return rect_coords

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")

    p_desc = "Add precision of model(FP32, FP16, FP16-INT8)"
    i_desc = "Path to input video file"
    mp_desc = "Precission of mouse controller(high, medium, low)"
    ms_desc = "Speed of mouse controller(high, medium, low)"

    parser._action_groups.pop()

    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-p", help=p_desc, required=False, default = 'FP32')
    optional.add_argument("-i", help=i_desc, default='./starter/bin/demo.mp4')
    optional.add_argument("-mp", help=mp_desc, default='medium')
    optional.add_argument("-ms", help=ms_desc, default='medium')

    args = parser.parse_args()

    return args

def main():
    args = get_args()
    model_name1 = './models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001'
    model_name2 = './models/intel/landmarks-regression-retail-0009/'+args.p+'/landmarks-regression-retail-0009'
    model_name3 = './models/intel/head-pose-estimation-adas-0001/'+args.p+'/head-pose-estimation-adas-0001'
    model_name4 = './models/intel/gaze-estimation-adas-0002/'+args.p+'/gaze-estimation-adas-0002'
    fd = FD.Model_Face_detection(model_name1, 'CPU')
    fld = FLD.Model_Facial_landmarks_de(model_name2, 'CPU')
    hpe = HPE.Model_Head_pose_estimation(model_name3, 'CPU')
    gme = GME.Model_Gaze_estimation(model_name4, 'CPU')
    mc = MC.MouseController(precision= args.mp, speed=args.ms)
    fd.load_model()
    fld.load_model()
    hpe.load_model()
    gme.load_model()
    

if __name__ == "__main__":
    main()