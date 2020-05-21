import face_detection as FD
import facial_landmarks_detection as FLD
import head_pose_estimation as HPE
import cv2
import os

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


model_name1 = '/home/aryan/gaze_controller/models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001'
model_name2 = '/home/aryan/gaze_controller/models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009'
model_name3 = '/home/aryan/gaze_controller/models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001'

fd = FD.Model_Face_detection(model_name1, 'CPU')
fld = FLD.Model_Facial_landmarks_de(model_name2, 'CPU')
hpe = HPE.Model_Head_pose_estimation(model_name3, 'CPU')

fd.load_model()
fld.load_model()
hpe.load_model()
video_file_path = '/home/aryan/gaze_controller/starter/bin/demo.mp4'
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
output_path = './output/'
out_video = cv2.VideoWriter(os.path.join(output_path, 'comb_output2.mp4'),
                            cv2.VideoWriter_fourcc(*'avc1'),
                            fps,
                            (output_video_w, output_video_h),
                            True)
left_eye_out_video = cv2.VideoWriter(os.path.join(output_path, 'left_eye_output.mp4'),
                            cv2.VideoWriter_fourcc(*'avc1'),
                            fps,
                            (2*delta, 2*delta),
                            True)
right_eye_out_video = cv2.VideoWriter(os.path.join(output_path, 'right_eye_output.mp4'),
                            cv2.VideoWriter_fourcc(*'avc1'),
                            fps,
                            (2*delta, 2*delta),
                            True)
try:
    # print('inside try')
    while cap.isOpened():
        # print('in while')
        ret, frame = cap.read()
        if not ret:
            break

        # print("frame is "+str(frame.shape))#1080,1920
        # print(initial_dims)#1080,1920
        cropped_image = fd.predict(frame, initial_dims)

        resized_cropped_image = fd.reshape_after_crop(cropped_image=cropped_image,
                                                       width=output_video_w,
                                                       height=output_video_h)

        real_face_coords = fld.predict(
            resized_cropped_image, [output_video_h, output_video_w])
        
        left_eye_coords = real_face_coords[:2]
        right_eye_coords = real_face_coords[2:4]
        face_point_drawn_frame = fld.draw_facial_points(left_eye_coords,resized_cropped_image)
        face_point_drawn_frame = fld.draw_facial_points(right_eye_coords,face_point_drawn_frame)

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

        cropping_coords = [50,50,200,200]#testing
        left_eye_frame = crop_image(resized_cropped_image,left_eye_rect_coords)
        right_eye_frame = crop_image(resized_cropped_image,right_eye_rect_coords)
        resized_left_eye_frame = resize_image(left_eye_frame,2*delta,2*delta)
        resized_right_eye_frame = resize_image(right_eye_frame,2*delta,2*delta)
        
        left_eye_out_video.write(resized_left_eye_frame)
        right_eye_out_video.write(resized_right_eye_frame)

        head_pose_angles = hpe.predict(resized_cropped_image, [output_video_h, output_video_w])
        
        # print("cropped image is "+str(cropped_image.shape))#373, 237
        # break

        
        head_pose_out_frame = hpe.write_on_video(
            face_point_drawn_frame, head_pose_angles)
        out_video.write(head_pose_out_frame)

        

    cap.release()
    cv2.destroyAllWindows()

except Exception as e:
    print("Could not run Inference: ", e)
