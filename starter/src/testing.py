import face_detection as FD
import cv2 
import os   

model_name = '/home/aryan/gaze_controller/models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001'
fd = FD.Model_Face_detection(model_name, 'CPU')
fd.load_model()
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
initial_dims = [initial_h,initial_w]
video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
output_path = './output/'
out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)
try:
    # print('inside try')
    while cap.isOpened():
        # print('in while')
        ret, frame = cap.read()
        if not ret:
            break

        # print("frame is "+str(frame.shape))
        cropped_image = fd.predict(frame, initial_dims)
        # print("cropped image is "+str(cropped_image.shape))
        resized_cropped_image = fd.reshape_after_crop(cropped_image= cropped_image, width= initial_w,height= initial_h)
        out_video.write(resized_cropped_image)


    cap.release()
    cv2.destroyAllWindows()

except Exception as e:
    print("Could not run Inference: ", e)

