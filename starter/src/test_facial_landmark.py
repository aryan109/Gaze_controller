import facial_landmarks_detection as FLD
import cv2 
import os   

model_name = '/home/aryan/gaze_controller/models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009'
fld = FLD.Model_Facial_landmarks_de(model_name, 'CPU')
fld.load_model()
video_file_path = '/home/aryan/gaze_controller/starter/src/output/output_video2.mp4'
try:
    cap = cv2.VideoCapture(video_file_path)
    print('captured video')
except FileNotFoundError:
    print("Cannot locate video file: " + video_file_path)
except Exception as e:
    print("Something else went wrong with the video file: ", e)
initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
initial_dims = [initial_h,initial_w]
video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
output_path = './output2/' #todo fix video output
out_video = cv2.VideoWriter(os.path.join(output_path, 
                            'facial_landmark_output_video.mp4'), 
                            cv2.VideoWriter_fourcc(*'avc1'), 
                            fps, 
                            (initial_w, initial_h), 
                            True)
count = 0
try:
    # print('inside try')
    while cap.isOpened():
        # print('in while')
        ret, frame = cap.read()
        if not ret:
            break

        real_face_coords, face_point_drawn_frame = fld.predict(frame,initial_dims)
        
        print("frame "+str(frame.shape)+"  drawn frame "+str(face_point_drawn_frame.shape))
        break
        # cropped_image = fld.predict(frame, initial_dims)
        # print("cropped image is "+str(cropped_image.shape))
        # resized_cropped_image = fld.reshape_after_crop(cropped_image= cropped_image, width= initial_w,height= initial_h)
        
        # out_video.write(face_point_drawn_frame)

        count += 1
        print(count)


    cap.release()
    cv2.destroyAllWindows()

except Exception as e:
    print("Could not run Inference: ", e)

