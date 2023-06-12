
**Chapter 1: Overview**


# Introduction

Computer vision is a resource-intensive task because it involves processing and analyzing large amounts of data in real-time to extract meaningful information from images and video. This requires significant processing power and memory, as well as specialized hardware such as graphics processing units (GPUs) to accelerate the computational workload. Onboard computer vision systems, which are used in a variety of applications including self-driving vehicles and drones, have additional constraints and requirements that can make them even more resource intensive.
Instead of using a costly Convolutional Neural Network (CNN) that processes an entire image at once, we can use a more cost-effective and faster deep neural network (DNN) that detects key points of hand gestures to detect the pose. This method is more efficient and can reduce the computational burden and cost of implementing a CNN. We are using a pre-trained machine learning model based on Keras.
In this project we aimed to make a robot with gesture recognition. Since robots are practically expensive our focus was to make a simulation and for this purpose, we used Turtle Bot on ROS 1. The main challenge was to use ROS-1 as all our workflow was on ROS-2. The major challenge involved porting our python script on to ROS-1 which although does have native Python support requires tinkering around with CMakeFiles to make it all work together. 
The neural network basically detects the hand and its pose and predicts the gesture. Then depending on the type of hand gesture signaled by the user, the robot is programmed to move in that direction.

**Media Pipe**


MediaPipe is an open-source framework for building cross-platform multimodal applied machine learning pipelines. It was developed by Google and is designed to make it easy to build and deploy machine learning models for a variety of media processing tasks such as facial detection, hand tracking, and speech recognition.
The framework consists of a set of reusable components called "calculators" that can be connected to form a pipeline. Each calculator performs a specific task, such as preprocessing data, extracting features, or making predictions. The calculators can be connected in a variety of ways to form a pipeline that meets the needs of the machine-learning task at hand.
MediaPipe also includes tools for labeling and annotation, visualization, and performance evaluation, making it easier to develop, debug, and deploy machine learning models. It is designed to be flexible and extensible, allowing developers to easily add custom calculators or integrate with other machine learning frameworks and libraries. Overall, MediaPipe is a useful tool for developers and researchers working on media processing tasks, as it provides a convenient way to build, test, and deploy machine learning models that can be used in a variety of applications.

**Tensorflow**

TensorFlow is an open-source software library for machine learning and artificial intelligence. It was developed by Google and is widely used in industry and academia for a variety of machine learning tasks, such as training and deploying machine learning models, building neural networks, and performing research.
TensorFlow is based on the concept of data flow graphs, where nodes in the graph represent mathematical operations and the edges represent the data (tensors) that flow between them. This allows TensorFlow to efficiently compute complex mathematical operations on large datasets, as the computations can be parallelized and distributed across multiple devices, such as CPUs and GPUs.

TensorFlow provides a wide range of tools and libraries for building and training machine learning models, including support for deep learning, convolutional neural networks, recurrent neural networks, and gradient-based optimization algorithms. It also includes tools for deploying machine learning models in production environments, such as the TensorFlow Serving library, which allows models to be served over a network for real-time prediction. Overall, TensorFlow is a powerful and widely-used platform for machine learning and artificial intelligence, and is suitable for a wide range of applications in industry and academia.













**Chapter 2: Workflow of Hand Gesture Recognition Model**

Necessary Packages
We used four libraries in our project which are cv2, numpy, MediaPipe and Tensorflow.
**import necessary packages for hand gesture recognition project using Python OpenCV**

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

**Use of Media Pipe**
The Mp.solution.hands module is responsible for performing the hand recognition algorithm. An object is created to store this module, which is called mpHands. The mpHands.Hands method is used to configure the model, with the first argument being the maximum number of hands that the model will detect in a single frame. MediaPipe is capable of detecting multiple hands in a single frame, but this particular project is set up to only detect one hand at a time. The Mp.solutions.drawing_utils module is also used to draw the detected key points of the hand, rather than having to draw them manually. Finally, Tensorflow is initialized to use these features.
# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

**Use of Tensorflow**
To use the TensorFlow pre-trained model for hand gesture recognition, the load_model function is called to load the model. The gesture.names file contains the names of the different gesture classes, and can be opened using the built-in open function in Python. The contents of the file can then be read using the read() function. This allows us to access the names of the gesture classes, which can be used to label the gestures that are detected by the model.

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

**Frame Reading using CV2**
To access the webcam and read the video frames, we create a VideoCapture object and pass an argument of '0', which is the camera ID of the system. In this case, there is only one webcam connected to the system. If you have multiple webcams, you may need to change the argument to the appropriate camera ID. Otherwise, you can leave it as the default value. The cap.read() function is then used to read each frame from the webcam, and the cv2.flip() function is used to flip the frame if desired. The cv2.imshow() function is used to display the frame on a new OpenCV window, and the cv2.waitKey() function keeps the window open until the 'q' key is pressed.
# Initialize the webcam for Hand Gesture Recognition Python project
cap = cv2.VideoCapture(0)

while True:
  # Read each frame from the webcam
  _, frame = cap.read()
x , y, c = frame.shape

  # Flip the frame vertically
  frame = cv2.flip(frame, 1)
  # Show the final output
  cv2.imshow("Output", frame)
  if cv2.waitKey(1) == ord('q'):
    		break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()

**Detection of Hand Key points**
MediaPipe works with RGB images, but OpenCV reads images in the BGR format. Therefore, we use the cv2.cvtCOLOR() function to convert the frame to the RGB format. The process function takes an RGB frame as input and returns a result class. We then use the result.multi_hand_landmarks method to check if any hands have been detected in the frame. If there are any detections, we loop through each detection and store the coordinates in a list called landmarks. The image height and width are multiplied by the result because the model returns a normalized result, with values between 0 and 1. Finally, the mpDraw.draw_landmarks() function is used to draw all of the landmarks on the frame.
framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  # Get hand landmark prediction
  result = hands.process(framergb)

  className = ''

  # post process the result
  if result.multi_hand_landmarks:
    	landmarks = []
    	for handslms in result.multi_hand_landmarks:
        	for lm in handslms.landmark:
            	# print(id, lm)
            	lmx = int(lm.x * x)
            	lmy = int(lm.y * y)

            	landmarks.append([lmx, lmy])

        	# Drawing landmarks on frames
        	mpDraw.draw_landmarks(frame, handslms, 
mpHands.HAND_CONNECTIONS)

**Recognizing the Hand Gestures**
The model.predict() function takes a list of landmarks as input and returns an array containing 10 prediction classes for each landmark. The output of this function will be an array with shape (1, 10), where each element represents the probability that the landmark belongs to each of the 10 classes. To determine the most likely class for each landmark, we can use the np.argmax() function, which returns the index of the maximum value in the list. Once we have the index, we can use it to retrieve the corresponding class name from the classNames list. Finally, we can use the cv2.putText function to display the detected gesture on the frame.
# Predict gesture in Hand Gesture Recognition project
        	prediction = model.predict([landmarks])
print(prediction)
        	classID = np.argmax(prediction)
        	className = classNames[classID]

  # show the prediction on the frame
  cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
               	1, (0,0,255), 2, cv2.LINE_AA)





**Chapter 3: Simulation of Robot**

**Turtlebot 3**

TurtleBot3 is a small, low-cost, open-source robot platform that is designed to be easy to use and learn with. It is powered by the Robot Operating System (ROS) and is equipped with a variety of sensors and actuators, including a laser rangefinder, infrared sensors, and a 360-degree camera. TurtleBot3 is capable of autonomous navigation and mapping and is intended to be used as a platform for learning and experimentation with robotics and artificial intelligence. It is suitable for use in a variety of applications, including education, research, and personal projects.

**Catkin Workspace**

A catkin workspace is a directory on your computer that is used to store and build ROS packages. It is named after the catkin build system, which is used to build ROS packages as part of the catkin build process.
A catkin workspace typically contains one or more ROS packages, along with the necessary resources and configuration files needed to build and install the packages. The top-level directory of a catkin workspace is usually called the "source space", and it contains the source code and other resources for the packages. The workspace also includes a "build space", where the packages are built and compiled, and a "devel space", where the built packages are installed and made available for use.
To create a catkin workspace, you can use the catkin_init_workspace command, which will create the necessary directories and configuration files. You can then use the catkin_make command to build and install the packages in the workspace.
The mkdir command is used to create a new directory on a computer. The command you provided, mkdir catkin_ws2/src, will create a new directory called "catkin_ws2" and a subdirectory called "src" inside it. The mkdir command is usually used in a terminal or command prompt, and it takes a single argument: the name of the directory to be created. 
Catkin workspaces are an important part of the ROS ecosystem, as they provide a structure for organizing and building ROS packages. They are often used by developers and researchers to create and manage complex ROS applications.
For this project, we created our catkin_ws and created 2 packages. One package was for our turtlebot3 and the other for the controlling of our robot.


**Chapter 4: Setting up Turtlebot3**

**Installing ROS**

$ sudo apt-get update
$ sudo apt-get upgrade
$ wget https://raw.githubusercontent.com/ROBOTIS-GIT/robotis_tools/master/install_ros_noetic.sh
$ chmod 755 ./install_ros_kinetic.sh 
$ bash ./install_ros_kinetic.sh

**Installing Dependent ROS Packages**

$ sudo apt-get install ros-kinetic-joy ros-kinetic-teleop-twist-joy \
  ros-kinetic-teleop-twist-keyboard ros-kinetic-laser-proc \
  ros-kinetic-rgbd-launch ros-kinetic-depthimage-to-laserscan \
  ros-kinetic-rosserial-arduino ros-kinetic-rosserial-python \
  ros-kinetic-rosserial-server ros-kinetic-rosserial-client \
  ros-kinetic-rosserial-msgs ros-kinetic-amcl ros-kinetic-map-server \
  ros-kinetic-move-base ros-kinetic-urdf ros-kinetic-xacro \
  ros-kinetic-compressed-image-transport ros-kinetic-rqt* \
  ros-kinetic-gmapping ros-kinetic-navigation ros-kinetic-interactive-markers

**Installing TurtleBot 3 Packages**
To install TurtleBot3 with the Debian packages:
$ sudo apt-get install ros-kinetic-dynamixel-sdk
$ sudo apt-get install ros-kinetic-turtlebot3-msgs
$ sudo apt-get install ros-kinetic-turtlebot3
To install the TurtleBot3 via Waffle Pie
$ echo "export TURTLEBOT3_MODEL=waffle_pi" >> ~/.bashrc

**Installation of Simulation Packages**
To launch the TurtleBot3 Simulation Package, we installed the turtlebot3 and turtlebot3_msgs packages installed on ourr system. Without these prerequisite packages, the Simulation cannot be launched.
$ cd ~/catkin_ws/src/
$ git clone -b kinetic-devel https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git
$ cd ~/catkin_ws && catkin_make

**Chapter 5: Creating Node on ROS**

After installation of Packages, we will now form our node. Following command was used to install the package:
Catkin_create_pkg robo rospy roscpp geometry_msgs
The dependencies were installed along the command which is “rospy roscpp geometry_msgs”.
Geometry_msgs

In ROS 1 (Robot Operating System), geometry_msgs is a package that provides a set of messages for representing geometric data, such as points, vectors, and poses. These messages are used to transmit geometric data between ROS nodes and are an important part of the ROS system.

The geometry_msgs package includes several different message types, including:
•	Point: a point in 3D space, represented by its x, y, and z coordinates
•	PointStamped: a point in 3D space, with a timestamp and a coordinate frame
•	Vector3: a 3D vector, represented by its x, y, and z components
•	Quaternion: a unit quaternion, representing an orientation in 3D space
•	Pose: a pose in 3D space, consisting of a position (a Point) and an orientation (a Quaternion)
These message types are used to represent various types of geometric data in ROS, and are often used in conjunction with other message types, such as sensor_msgs (for sensor data) and nav_msgs (for navigation data). As for our project, we will use these geometry_msgs to move our robot in simulation.
Twist messages
In ROS (Robot Operating System), geometry_msgs/Twist is a message type that represents a velocity command for a mobile robot. It is defined in the geometry_msgs package and consists of two fields: linear and angular.

The linear field represents the linear velocity of the robot, and is a 3D vector (Vector3) with the x, y, and z components of the velocity. The angular field represents the angular velocity of the robot, and is also a 3D vector with the x, y, and z components of the velocity.

The Twist message is often used to control the movement of a mobile robot, such as a differential-drive robot or a robot with holonomic wheels. It can be published to a ROS topic and subscribed to by a node that is responsible for controlling the robot's motors or wheels.
After the package is created, we go in the package folder and create a script folder and place the node1.py. Other files which are to be used in the code were placed in that catkin_ws root directory i.e hand gesture code files and its dependencies. 

**Running Python Code on ROS**

ROS 1 has the ability to run and execute the python script but it runs cpp script by default. To make sure it runs python script, we made a change in“cmakelists.txt” inside the src folder. The changes are as follows:
 
















**Chapter 6: Node Code**

**Subscriber and Publisher**

In ROS (Robot Operating System), a subscriber for geometry_msgs/Twist messages is a node that receives velocity commands for a mobile robot. The Twist message is defined in the geometry_msgs package and consists of two fields: linear and angular, which represent the linear and angular velocities of the robot, respectively. Similarly, the publisher is a node that is used to send velocity commands for the robot.
Implementation of the code
In our node we integrated the Hand gesture recognition and using those hand gestures, we mapped some of those to our node. For example, the thumbs up and down is for the linear movement of the robot and fist and peace signs were used to rotate the robot in the simulation. 

#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils


# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)


# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set up the publisher and subscriber
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)


# Initialize the ROS node
rospy.init_node('node1')

# Define the callback function for the subscriber
def twist_callback(twist_msg):
# Do something with the twist message here
    print(twist_msg)
rospy.Subscriber('/cmd_vel', Twist,twist_callback)
# Define a twist message and set the linear and angular velocities
twist = Twist()


twist.linear.x = 0.5
twist.angular.z = 0.0

# Publish the twist message at a rate of 10 Hz
rate = rospy.Rate(10)
while not rospy.is_shutdown():

    # Read each frame from the webcam
    _, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # print(result)
    
    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
            prediction = model.predict([landmarks])
            # print(prediction)
            classID = np.argmax(prediction)
            className = classNames[classID]

    # show the prediction on the frame
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0,0,255), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame) 

    if (className == "thumbs up"):
        twist.linear.x = 0.5
        twist.angular.z = 0.0

    elif (className == "thumbs down"):
        twist.linear.x = -0.5
        twist.angular.z = 0.0
    elif (className == "fist"):
        twist.linear.x = 0.0
        twist.angular.z = 0.5
    elif (className == "peace"):
        twist.linear.x = 0.0
        twist.angular.z = -0.5
    else:
        twist.linear.x = 0.0
        twist.angular.z = 0.0

    if cv2.waitKey(1) == ord('q'):
        break


    pub.publish(twist)
    rate.sleep()
# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()


**Chapter 7: Block Diagram of our project**


By default, the turtlebot3 subscribes the cmd_val topic to get all movement commands. We, therefore, create our own node that publishes movement commands in form of "twist messages" to this topic, after processing the webcam feed to identify the gesture. These messages then get routed to our turtlebot for linear and angular movements.







![image](https://github.com/usama-qadoos/Gesture-Recognizing-Robot-ROS2-OpenCV/assets/115080912/1020e1f0-1f3f-486f-bfaf-fb55651929cb)










**Chapter 8: Simulation**
Here is the working of our project where a gesture moves our robot in Gazebo. Here are a few case scenarios.

 
Robot moving forward
![image](https://github.com/usama-qadoos/Gesture-Recognizing-Robot-ROS2-OpenCV/assets/115080912/f4026bf6-b7a1-44aa-ba30-5a0605769833)

Robot moving backward
![image](https://github.com/usama-qadoos/Gesture-Recognizing-Robot-ROS2-OpenCV/assets/115080912/8a6a042c-7da1-4f2d-b8dd-1c664693d9cb)

 
Robot rotating anti-clockwise
 ![image](https://github.com/usama-qadoos/Gesture-Recognizing-Robot-ROS2-OpenCV/assets/115080912/7c4ce000-7837-4288-a519-9a3e97e3c1d5)

Robot rotating clockwise
![image](https://github.com/usama-qadoos/Gesture-Recognizing-Robot-ROS2-OpenCV/assets/115080912/0846f443-e68c-4d13-aac7-2bbba3a6eff5)

**Chapter 9: Link to the simulation:**
Here is the link to the video simulation of our project:
https://drive.google.com/file/d/15pjQLsGIpVXpM1sGEwY-d3IhvSxOmfVM/view?usp=sharing
Robotics Final Demo 























**Chapter 10: Conclusion**

In this project, we used OpenCV, TensorFlow, and MediaPipe to detect hand gestures from a webcam input and use these gestures to control a Turtlebot robot through ROS. Specifically, we are using OpenCV to process the video frames from the webcam and identify hand gestures, TensorFlow to apply machine learning techniques to classify the gestures, and MediaPipe to handle the real-time processing of the video stream. We are then using publisher and subscriber nodes in ROS to communicate with the Turtlebot and send it commands based on the identified hand gestures.






















**Chapter 11: References**

1-	Real-time hand gesture recognition using TensorFlow & OpenCV
https://techvidvan.com/tutorials/hand-gesture-recognition-tensorflow-opencv/

2-	Turtlebot3 overview 
https://emanual.robotis.com/docs/en/platform/turtlebot3/overview/

3-	Turtlebot3 quick-start
https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/

4-	Gazebo Simulation
https://emanual.robotis.com/docs/en/platform/turtlebot3/simulation/#gazebo-simulation

5-	Creating a ROS Package
https://www.youtube.com/watch?v=A-1DBhWF_64&t=341s&ab_channel=RoboticsBack-End



