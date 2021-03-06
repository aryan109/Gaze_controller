import starter.src.face_detection as FD
import starter.src.facial_landmarks_detection as FLD
import starter.src.head_pose_estimation as HPE
import starter.src.gaze_estimation as GME
import starter.src.mouse_controller as MC
import cv2
import os
import argparse
import time




def resize_image(image, width, height):
    dims = (width, height)
    resized_image = cv2.resize(image, dims, interpolation=cv2.INTER_AREA)
    return resized_image


def crop_image(image, coords):
    xmin = int(coords[0])
    ymin = int(coords[1])
    xmax = int(coords[2])
    ymax = int(coords[3])
    cropped_image = image[ymin:ymax, xmin:xmax]
    return cropped_image


def generate_rectangle_coordinates_from_midpoint(x, y, delta, maxlim):
    xmin = max(x - delta, 0)
    ymin = max(y - delta, 0)
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
    c_desc = "enter 1 if use webcam else 0, default is 0"
    fd_desc = "path to face detection model"
    lr_desc = "path to facial landmark detecction model"
    hp_desc = "path to face head pose estimation model"
    ge_desc = "path to gaze estimation model"
    cmp_desc = "set to true if want path to custom model"

    parser._action_groups.pop()

    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    optional.add_argument("-p", help=p_desc, required=False, default='FP32')
    optional.add_argument("-i", help=i_desc, default='./starter/bin/demo.mp4')
    optional.add_argument("-mp", help=mp_desc, default='medium')
    optional.add_argument("-ms", help=ms_desc, default='medium')
    optional.add_argument("-c", help=c_desc, default=0)
    optional.add_argument("-fd", help=fd_desc, default=0)
    optional.add_argument("-lr", help=lr_desc, default=0)
    optional.add_argument("-hp", help=hp_desc, default=0)
    optional.add_argument("-ge", help=ge_desc, default=0)
    optional.add_argument("-cmp", help=cmp_desc, default=False)

    args = parser.parse_args()

    return args

def main():
    args = get_args()
    if args.cmp == False:
        model_name1 = './models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001'
        model_name2 = './models/intel/landmarks-regression-retail-0009/' + \
            args.p+'/landmarks-regression-retail-0009'
        model_name3 = './models/intel/head-pose-estimation-adas-0001/' + \
            args.p+'/head-pose-estimation-adas-0001'
        model_name4 = './models/intel/gaze-estimation-adas-0002/' + \
            args.p+'/gaze-estimation-adas-0002'
    else:
        model_name1 = args.fd
        model_name2 = args.lr
        model_name3 = args.hp
        model_name4 = args.ge
    fd = FD.Model_Face_detection(model_name1, 'CPU')
    fld = FLD.Model_Facial_landmarks_de(model_name2, 'CPU')
    hpe = HPE.Model_Head_pose_estimation(model_name3, 'CPU')
    gme = GME.Model_Gaze_estimation(model_name4, 'CPU')
    mc = MC.MouseController(precision=args.mp, speed=args.ms)
    fd_start = time.time()
    fd.load_model()
    fd_end = time.time()
    fd_load_time = fd_end - fd_start
    print('face detection model loading time is :'+str(fd_load_time))
    
    fld_start = time.time()
    fld.load_model()
    fld_end = time.time()
    fld_load_time = fld_end - fld_start
    print('facial landmark detection model loading time is :'+str(fld_load_time))
    
    hpe_start = time.time()
    hpe.load_model()
    hpe_end = time.time()
    hpe_load_time = hpe_end - hpe_start
    print('head pose estimation model loading time is :'+str(hpe_load_time))
        
    gme_start = time.time()
    gme.load_model()
    gme_end = time.time()
    gme_load_time = gme_end - gme_start
    print('gaze model loading time is :'+str(gme_load_time))
    print(f'total model loading time : {gme_load_time+hpe_load_time+fd_load_time+fld_load_time}')
    if(args.c != 1):
        video_file_path = args.i
    else:
        video_file_path = 0
    try:
        cap = cv2.VideoCapture(video_file_path)
        print('captured video')
    except FileNotFoundError:
        print("Cannot locate video file: " + video_file_path)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    initial_dims = [initial_h, initial_w]
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_video_w = int(300)
    output_video_h = int(450)
    delta = 60
    movemnt_time = []
    fd_time = []
    fld_time = []
    hpe_time = []
    ge_time = []
    count = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            infer_start = time.time()
            fd_start = time.time()
            cropped_image,output_image, face_coords = fd.predict(frame, initial_dims)
            fd_end = time.time()
            fd_time.append(fd_end - fd_start)
            resized_cropped_image = fd.reshape_after_crop(cropped_image=cropped_image,
                                                          width=output_video_w,
                                                          height=output_video_h)

            fld_start = time.time()
            real_face_coords = fld.predict(
                resized_cropped_image, [output_video_h, output_video_w])

            fld_end = time.time()
            fld_time.append(fld_end - fld_start)
            left_eye_coords = real_face_coords[:2]
            right_eye_coords = real_face_coords[2:4]

            left_eye_rect_coords = generate_rectangle_coordinates_from_midpoint(
                left_eye_coords[0],
                left_eye_coords[1],
                delta,
                300)
            right_eye_rect_coords = generate_rectangle_coordinates_from_midpoint(
                right_eye_coords[0],
                right_eye_coords[1],
                delta,
                300)

            left_eye_frame = crop_image(
                resized_cropped_image, left_eye_rect_coords)
            right_eye_frame = crop_image(
                resized_cropped_image, right_eye_rect_coords)
            resized_left_eye_frame = resize_image(
                left_eye_frame, 2*delta, 2*delta)
            resized_right_eye_frame = resize_image(
                right_eye_frame, 2*delta, 2*delta)

            hpe_start = time.time()
            head_pose_angles = hpe.predict(resized_cropped_image, [
                                           output_video_h, output_video_w])
            hpe_end = time.time()
            hpe_time.append(hpe_end - hpe_start)

            ge_start = time.time()

            gaze_result = gme.predict(
                head_pose_angles, resized_left_eye_frame, resized_right_eye_frame)
            ge_end = time.time()
            ge_time.append(ge_end - ge_start)

           # mc.move(gaze_result[0][0], gaze_result[0][1])
            print('pointer moved')
            infer_end = time.time()
            print(f'time for 1 total isference is: {infer_end-infer_start}')
            movemnt_time.append(infer_end-infer_start)
            # count += 1
            # if count >= 30:
            #     break
        cap.release()
        cv2.destroyAllWindows()
        sum = 0
        sf = 0
        sfld = 0
        shpe = 0
        sge = 0
        for i,j,k,l,m in zip(movemnt_time,fd_time,fld_time,hpe_time,ge_time):
            sum += i
            sf += j
            sfld += k
            shpe += l
            sge += m
        
        print(f'avg total inference time is: {sum/len(movemnt_time)}')
        print(f'avg face detection model inference time is: {sf/len(fd_time)}')
        print(f'avg face landmark detection model inference time is: {sfld/len(fld_time)}')
        print(f'avg headpose detection model inference time is: {shpe/len(hpe_time)}')
        print(f'avg gaze estimation inference time is: {sge/len(ge_time)}')


    except Exception as e:
        print("Could not run Inference: ", e)

if __name__ == "__main__":
    main()
