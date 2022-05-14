# Live Class Monitoring System [Face emotion Recognition].
DL+ML project

# Introduction

Facial Emotion Recognition (FER) is the technology that analyses facial expressions from both static images and videos in order to reveal information on one's emotional state
The data consists of 48x48 pixel grayscale images of faces. The objective is to classify each face based on the emotion shown in the facial expression into one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). use OpenCV to automatically detect faces in images and draw bounding boxes around them. I trained, saved, and exported the CNN, then directly served the trained model to a web interface and performed real-time facial expression recognition on video and image data. 



# Problem Statement

The Indian education landscape has been undergoing rapid changes for the past 10 years owing to the advancement of web-based learning services, specifically, eLearning platforms. Global E-learning is estimated to witness an 8X over the next 5 years to reach USD 2B in 2021. India is expected to grow with a CAGR of 44% crossing the 10M users mark in 2021. Although the market is growing on a rapid scale, there are major challenges associated with digital learning when compared with brick and mortar classrooms. One of many challenges is how to ensure quality learning for students. Digital platforms might overpower physical classrooms in terms of content quality but when it comes to understanding whether students are able to grasp the content in a live class scenario is yet an open-end challenge. In a physical classroom during a lecturing teacher can see the faces and assess the emotion of the class and tune their lecture accordingly, whether he is going fast or slow. He can identify students who need special attention. Digital classrooms are conducted via video telephony software program (ex- Zoom) where it’s not possible for medium scale class (25-50) to see all students and access the mood. Because of this drawback, students are not focusing on content due to a lack of surveillance. While digital platforms have limitations in terms of physical surveillance but it comes with the power of data and machines which can work for you. It provides data in the form of video, audio, and texts which can be analyzed using deep learning algorithms. Deep learning backed system not only solves the surveillance issue, but it also removes the human bias from the system, and all information is no longer in the teacher’s brain rather translated in numbers that can be analyzed and tracked.

We will solve the above-mentioned challenge by applying deep learning algorithms to live video data.The solution to this problem is by recognizing facial emotions.

# Dataset Information

The data comes from the past Kaggle competition “Challenges in Representation Learning: Facial Expression Recognition Challenge”: we have defined the image size to 48 so each image will be reduced to a size of 48x48.The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. Each image corresponds to a facial expression in one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). The dataset contains approximately 36K images.

Dataset link - www.kaggle.com/jonathanoheix/face-expression-recognition-dataset

![dataset](https://user-images.githubusercontent.com/102009481/168103276-f8f0f27b-c189-408e-b8dd-333030d54575.png)

# Dependencies

Python 3
Tensorflow
Streamlit
Streamlit-Webrtc
OpenCV

# Model Creation

# 1. ResNet-50

             ResNet-50 is a convolutional neural network that is 50 layers deep. You can load a pre-trained version of the network trained on more than a million images from the ImageNet database 

training accuracy and validation accuracy is very low so our pretrained model resnet50 is not performing well

# 2.CNN

         Designing the CNN model for, emotion detection .creating blocks using Conv2D layer,Batch-Normalization, Max-Pooling2D, Dropout, Flatten, and then stacking them together and at the end-use Dense Layer for output

 ![acc](https://user-images.githubusercontent.com/102009481/168103687-2cf25ccb-948c-4804-80c0-e8d4baadc50a.png)
 
 # confusion matrix
 
![confusion](https://user-images.githubusercontent.com/102009481/168104085-260ff2ed-5448-4338-8d61-a5e7133fce64.png)

We trained the neural network and we achieved the highest training accuracy of 74.43%. After using test data to check how well our model 
generalize, we score an astounding 63.17% on the test set.

# Deployment of streamlit webApp in Heroku and Streamlit


We have created front-end using streamlit-webrtc which helped to deal with real time video streams.Image captured from the webcam is sent to VideoTransformer function to detect the emotion .Then this model was deployed on heroku platform

Deployment Link for Heroku -Deployment Link for heroku :- https://face-recognit-apoorva.herokuapp.com/

Deployment Link for Streamlit Share - https://share.streamlit.io/apoorvakr12695/face-emotion-recognition-/main/app.py



 ![Screenshot (5)](https://user-images.githubusercontent.com/102009481/168220751-47a61294-912b-4a91-8b89-559cf2a0ac52.png)
 
 ![Screenshot (7)](https://user-images.githubusercontent.com/102009481/168220847-9fbbe9ec-deaa-4f6f-adfe-dced2ec983b4.png)

![Screenshot (8)](https://user-images.githubusercontent.com/102009481/168220868-54e172b2-8148-40fe-92c9-ada4c681a789.png)

# Run WebApp Locally

Clone the project

        git clone https://github.com/apoorvaKR12695/face-emotion-detection-

  
Install dependencies

          pip install -r requirement.txt
  
Start local webcam

           streamlit run app.py
