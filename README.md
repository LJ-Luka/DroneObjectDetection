# Drone Object Detection

**Summary**  

I decided to work on a project involving two things I love; drones and software. Hence, I took on this Drone Object Detection Project.   
The objective of this project is to identify and label objects by name in real time. To enable me to gain access to the device camera, utilize a trained neural network model, and manipulate real-time video, I used the abstractions provided by OpenCV (Open-Source Computer Vision).
<br/>
The first step in this project was to import cv2(OpenCV) and other packages needed. I then proceeded to create a variable that holds the device video capture object and set the values for its height and width. This object will later be used to display objects identified in real time.   
Next, I have a file called coco.names, which contains names of popular objects we see daily. The images in that list have been annotated on over 1.5 million object instances. These names are then put into a list so that opencv can choose the object that’s in a video frame.  
My next line of action was to pass in a fixed weight inference graph to the opencv dnn_DetectionModel and a config file. This file contains a pre-trained graph that can’t be trained further, hence the term ‘Frozen’ in the name. The weights of edges between nodes allow the network to make natural deduction inferences on the current video frame.  
The final phases involved looping and reading the current video frame as an image array vector using the video object mentioned earlier. This image is then passed as a parameter to the DNN (deep neural network) detection function, using a confidence threshold of 60%. An object wouldn’t always look 100% like itself due to noise and other factors, so my choice of threshold seems reasonable. This function returns a classId, confidences, and boxes bounding the object detected. The classID is an object ID that allows us to group key points to an object. With the result of the detection, I’m able to select the name of the object using the class ID and successfully display it on the current video frame.  


Completed with guidance from Murtaza's workshop
