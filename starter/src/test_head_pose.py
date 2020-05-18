import head_pose_estimation as HPE
import cv2
import os

model_name = '/home/aryan/gaze_controller/models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001'
hpe = HPE.Model_Head_pose_estimation(model_name, 'CPU')
hpe.load_model()
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
initial_dims = [initial_h, initial_w]
video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
output_path = './output3/'
# out_video = cv2.VideoWriter(os.path.join(output_path,
#                                          'head_pose_output_video.mp4'),
#                             cv2.VideoWriter_fourcc(*'avc1'),
#                             fps,
#                             (initial_w, initial_h),
#                             True)
count = 0
try:
    # print('inside try')
    while cap.isOpened():
        # print('in while')
        ret, frame = cap.read()
        if not ret:
            break

        all_result = hpe.predict(
            frame, initial_dims)
        print(f"result: {all_result}")
        print(f"all_result[{hpe.output_name1}]: {all_result[hpe.output_name1]}")
        print(f"all_result[{hpe.output_name2}]: {all_result[hpe.output_name2]}")
        print(f"all_result[{hpe.output_name3}]: {all_result[hpe.output_name3]}")
        
        
        break
        # out_video.write(face_point_drawn_frame)

    cap.release()
    cv2.destroyAllWindows()

except Exception as e:
    print("Could not run Inference: ", e)
