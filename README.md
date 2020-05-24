# Computer Pointer Controller
It is a simple python application that moves the mouse
pointer according to the motion of our eyes.

## Project Set Up and Installation
*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.

you need to have open vino installed in your device here is the [link](https://docs.openvinotoolkit.org/latest/index.html) to install it.

After this you need to install the dependencies present in requirements.txt .I suggest you to create a 
virtual environment for runnig this application so the dependencies installed will not interfere with existing packages.

## Demo
*TODO:* Explain how to run a basic demo of your model.

run `python app.py -h` to get information about the the commandline arguments.


although there are no required parameters, all are set to defaults for the following scenario.

- default paths to models present in _models_ folder
- run inference on video present in starter/bin/demo.mp4
- default precission is FP32
- paths are set according to linux convention

## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

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

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

model loading time : 

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
