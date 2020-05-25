# Computer Pointer Controller
It is a simple python application that moves the mouse
pointer according to the motion of our eyes.

## Project Set Up and Installation

you need to have open vino installed in your device here is the [link](https://docs.openvinotoolkit.org/latest/index.html) to install it.

After this you need to install the dependencies present in _requirements.txt_ .I suggest you to create a virtual environment for runnig this application so the dependencies installed will not interfere with existing packages.

## Demo

run `python app.py -h` to get information about the the commandline arguments.

although there are no required parameters, all are set to defaults for the following scenario.

- default paths to models present in _models_ folder
- run inference on video present in starter/bin/demo.mp4
- default precission is FP32
- paths are set according to linux convention

## Documentation

You can run `python app.py -h` to see description of supported command line arguments.

following is the list of supported commandline arguments:-


  -p P :   Add precision of model(FP32, FP16, FP16-INT8)

  -i I :      Path to input video file

  -mp MP :    Precission of mouse controller(high, medium, low)

  -ms MS :   Speed of mouse controller(high, medium, low)

  -c C   :   enter 1 if use webcam else 0, default is 0

  -fd FD  :  path to face detection model

  -lr LR   : path to facial landmark detecction model

  -hp HP   : path to face head pose estimation model

  -ge GE   : path to gaze estimation model

  -cmp CMP : set to true if want path to custom model

  -o O     : path to output file (mp4)

  -of OF    flag to generate output file by default true, set to False to skip autput file generation

## Benchmarks

face detection model loading time is :0.17 sec

facial landmark detection model loading time is :0.08 sec

head pose estimation model loading time is :0.08 sec
gaze estimation model loading time is :0.10 sec

total model loading time : 0.45 sec

### average pointer movement time with different pression and speed:-

| precission | speed |  avg movement time(sec) |
| --- | --- | --- |
|medium | medium | 5.42 |
| high  | fast   | 1.22 |
| high  | slow   | 10.61|
| high  | medium | 5.40 |
| medium| fast   | 1.24 |
| medium| slow   | 10.61|
| low   | fast   | 1.23 |
| low   | medium | 5.42 |
| low   | slow   | 10.85|

### Inference time :

__FP32:-__ 

average inference time for face detection model = 31.6 ms

average inference time for face landmark detection model = 1.3 ms

average inference time for headpose estimation model = 3.2 ms

average inference time for gaze estimation model = 3.5 ms

average total inference time = 40.7 ms

__FP16:-__

average inference time for face detection model = 27.6 ms

average inference time for face landmark detection model = 1.1 ms

average inference time for headpose estimation model = 2.6 ms

average inference time for gaze estimation model = 2.7 ms

average total inference time = 34.9 ms

__INT8:-__

average inference time for face detection model = 26.9 ms

average inference time for face landmark detection model = 1.1 ms

average inference time for headpose estimation model = 2.2 ms

average inference time for gaze estimation model = 2.2 ms

average total inference time = 33.3 ms

## Results

As we can see  from the benchmarks the lower precission model(FP32 & INT8) takes less inference time as compared to FP32 models. This is because less precission models are lighter models and have quantinnzed weights. They have higher throughput but lower accuraacy.

We can also note that when mouse controller is set to fast it takes least time of about 1.2 sec and when set to slow it takes highest time of about 10.6 sec. When mouse controller speed is set to fast, it quickly moves the pointer and is most responsive.

