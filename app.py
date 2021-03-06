import starter.src.face_detection as FD
import starter.src.facial_landmarks_detection as FLD
import starter.src.head_pose_estimation as HPE
import starter.src.gaze_estimation as GME
import starter.src.mouse_controller as MC
import cv2
import os
import argparse
import logging
logging.basicConfig(filename='app.log', filemode='w',level=logging.INFO) # if you want to log it into file


def denormalize_coordinates(box, initial_w,initial_h):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        coords = [box[3] * initial_w,#width
                  box[4] * initial_h,
                  box[5] * initial_w,
                  box[6] * initial_h]
        return coords

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

def draw_rectangle(xmin, ymin, xmax, ymax,image):

        frame = image
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255, 1))

        return frame

def draw_arrow(image, x_start, y_start, x_end,y_end):
    color = (0,255,0)
    start_point = (int(x_start),int(y_start))
    end_point = (int(x_end),int(y_end))
    thickness = 4
    image = cv2.arrowedLine(image, start_point, end_point, 
                                     color, thickness)
    return image


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
    o_desc = "path to output file (mp4)"
    of_desc = "flag to generate output file by default true, set to False to skip autput file generation"
    d_desc = "Choose hardware on which you have to run(Default is set to 'CPU'"
    

    parser._action_groups.pop()

    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    optional.add_argument("-p", help=p_desc, required=False, default='FP32')
    optional.add_argument("-i", help=i_desc, default='./starter/bin/demo.mp4')
    optional.add_argument("-mp", help=mp_desc, default='medium')
    optional.add_argument("-ms", help=ms_desc, default='medium')
    optional.add_argument("-c", help=c_desc,type=bool ,default=False)
    optional.add_argument("-fd", help=fd_desc, default=0)
    optional.add_argument("-lr", help=lr_desc, default=0)
    optional.add_argument("-hp", help=hp_desc, default=0)
    optional.add_argument("-ge", help=ge_desc, default=0)
    optional.add_argument("-cmp", help=cmp_desc, default=False)
    optional.add_argument("-o", help=o_desc, default='./output/output.mp4')
    optional.add_argument("-of", help=of_desc, default=True)
    optional.add_argument("-d", help =d_desc, default='CPU')


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
    fd = FD.Model_Face_detection(model_name1, args.d)
    fld = FLD.Model_Facial_landmarks_de(model_name2, args.d)
    hpe = HPE.Model_Head_pose_estimation(model_name3, args.d)
    gme = GME.Model_Gaze_estimation(model_name4, args.d)
    mc = MC.MouseController(precision=args.mp, speed=args.ms)
    fd.load_model()
    fld.load_model()
    hpe.load_model()
    gme.load_model()
    if(args.c == False):
        video_file_path = args.i
    else:
        video_file_path = 0
        print("webcam")
        logging.info("video stream from webcam")
    try:
        cap = cv2.VideoCapture(video_file_path)
        print('captured video')
        logging.info("captured video file: "+str(video_file_path))
    except FileNotFoundError:
        print("Cannot locate video file: " + video_file_path)
        logging.info("Cannot locate video file: ")
    except Exception as e:
        print("Something else went wrong with the video file: ", e)
        logging.info("Something else went wrong with the video file: "+str(e))
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    initial_dims = [initial_h, initial_w]
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_video_w = int(300)
    output_video_h = int(450)
    delta = 60
    if args.of :
        output_path = args.o
        print("writing output to:"+str(output_path))
        logging.info("writing output to:"+str(output_path))
        out_video = cv2.VideoWriter(output_path,
                                cv2.VideoWriter_fourcc(*'avc1'),
                                fps,
                                (initial_w, initial_h),
                                True)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            cropped_image, output_image, face_coords = fd.predict(frame, initial_dims)

            resized_cropped_image = fd.reshape_after_crop(cropped_image=cropped_image,
                                                          width=output_video_w,
                                                          height=output_video_h)

            real_face_coords = fld.predict(
                resized_cropped_image, [output_video_h, output_video_w])

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

            xmin_offset = face_coords[0]
            ymin_offset = face_coords[1]
            xmax_offset = face_coords[0] - 50
            ymax_offset = face_coords[1] - 65
            
            # draw left eye box
            output_image = draw_rectangle(xmin_offset + left_eye_rect_coords[0], 
                                        ymin_offset +left_eye_rect_coords[1],
                                        xmax_offset +left_eye_rect_coords[2],
                                        ymax_offset +left_eye_rect_coords[3], 
                                        output_image)

            # draw right eye box
            output_image = draw_rectangle(xmin_offset + right_eye_rect_coords[0], 
                                        ymin_offset +right_eye_rect_coords[1],
                                        xmax_offset +right_eye_rect_coords[2],
                                        ymax_offset +right_eye_rect_coords[3], 
                                        output_image)

            left_eye_frame = crop_image(
                resized_cropped_image, left_eye_rect_coords)
            right_eye_frame = crop_image(
                resized_cropped_image, right_eye_rect_coords)
            resized_left_eye_frame = resize_image(
                left_eye_frame, 2*delta, 2*delta)
            resized_right_eye_frame = resize_image(
                right_eye_frame, 2*delta, 2*delta)

            head_pose_angles = hpe.predict(resized_cropped_image, [
                                           output_video_h, output_video_w])

            gaze_result = gme.predict(
                head_pose_angles, resized_left_eye_frame, resized_right_eye_frame)

            if(args.of == False):
                mc.move(gaze_result[0][0], gaze_result[0][1])
            if(args.of):
                x_move, y_move = mc.get_coord(gaze_result[0][0],gaze_result[0][1])
            output_image = draw_arrow(output_image, 
                                    left_eye_coords[0] + xmax_offset + 25,
                                    left_eye_coords[1] + ymax_offset + 25,
                                    left_eye_coords[0] + x_move + xmin_offset,
                                    left_eye_coords[1] + y_move + ymin_offset)
            output_image = draw_arrow(output_image, 
                                    right_eye_coords[0] + xmax_offset + 25,
                                    right_eye_coords[1] + ymax_offset + 25,
                                    right_eye_coords[0] + x_move + xmin_offset,
                                    right_eye_coords[1] + y_move + ymin_offset)
            print('pointer moved')
            logging.info("pointer moved")
            if args.of :
                out_video.write(output_image)
                print('printed output')
                logging.info("Printed output to output video file")
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference: ", e)
        logging.info("Could not run Inference: "+str(e))

if __name__ == "__main__":
    main()
