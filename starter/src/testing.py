import face_detection as FD

model_name = '/home/aryan/gaze_controller/models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001'
fd = FD.Model_Face_detection(model_name, 'CPU')
fd.load_model()
video_file_path = '/home/aryan/gaze_controller/starter/bin/demo.mp4'
try:
    cap = cv2.VideoCapture(video_file_path)
except FileNotFoundError:
    print("Cannot locate video file: " + video_file_path)
except Exception as e:
    print("Something else went wrong with the video file: ", e)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            fd.predict(frame)
            cv2.imshow(frame)

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print("Could not run Inference: ", e)
