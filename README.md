
</p>
<h1 align="center"> Face Emotion Detection </h1>
<h3 align="center"> AlmaBetter Verfied Project - <a href="https://www.almabetter.com/"> AlmaBetter School </a> </h5>



![1_kNmSTa-xG_Ed_xmHPWAzeQ](https://user-images.githubusercontent.com/102009481/177730708-9d7f518e-0553-47b0-8d77-7d3be8159d56.png)

<p>Built an  face emotion detection app that detects the sentiment of the online classroom using live video from the webcam and real-time aggregated feedback to the instructors about the class using CNN model and deployed on Heroku platform.</p>

<h2> :floppy_disk: Project Files Description</h2>


<p>This Project includes 2 executable files, 2 text files ,1 h5 file as well as 2 directories as follows:</p>
<h4>Executable Files:</h4>
<ul>
  
  <li><b>train.py</b> - Includes all functions required for classification operations  and generates the model.h5 file after execution.</li>
  <li><b>test.py</b> -  after execution, evaluation is done on the unseen data as in confusion_matrix.txt.</li>
</ul>

<h4>Output Files:</h4>
<ul>
  <li><b>model.h5</b> - Model contains information about the emotions of the train set, such as the Happy,Sad,Disgust,Calm and so on.</li>
  <li><b>confusion_matrix.txt</b> - Contains information about the classified emotions of the test set.</li>
  <li><b>pics</b> - which contains the output of detecting emotions through live webcam.</li>
</ul>

<h4>Source Directories:</h4>
<ul>
  <li><b>train directory</b> - Includes all emotions  for the training phase of the program.</li>
  <li><b>test directory</b> - Includes all emotions for the testing phase of the program.</li>
</ul>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> :book: CNN </h2>

<p> Convolutional Neural Networks (CNNs) are a type of Neural Network that has excelled in a number of contests involving Computer Vision and Image Processing.Designing the CNN model for, emotion detection .creating blocks using Conv2D layer,Batch-Normalization, Max-Pooling2D, Dropout, Flatten, and then stacking them together and at the end-use Dense Layer for output


![1_CnNorCR4Zdq7pVchdsRGyw](https://user-images.githubusercontent.com/102009481/177744968-d0bb6264-acd9-429e-bc7e-56cd3464574c.png)



![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<h2> :clipboard: Execution Instruction</h2>
<p>The order of execution of the program files is as follows:</p>
<p><b>1) spam_detector.py</b></p>
<p>First, the spam_detector.py file must be executed to define all the functions and variables required for classification operations.</p>
<p><b>2) train.py</b></p>
<p>Then, the train.py file must be executed, which leads to the production of the model.txt file. 
At the beginning of this file, the spam_detector has been imported so that the functions defined in it can be used.</p>
<p><b>3) test.py</b></p>
<p>Finally, the test.py file must be executed to create the result.txt and evaluation.txt files.
Just like the train.py file, at the beginning of this file, the spam_detector has been imported so that the functions defined in it can be used.</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- CREDITS -->
<h2 id="credits"> :scroll: Credits</h2>

< Your Name > | Avid Learner | Data Scientist | Machine Learning Engineer | Deep Learning enthusiast

<p> <i> Contact me for Data Science Project Collaborations</i></p>


[![LinkedIn Badge](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/company/almabetter/mycompany/)
[![GitHub Badge](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/orgs/AlmaBetter-School/)
[![Medium Badge](https://img.shields.io/badge/Medium-1DA1F2?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/almabetter)
[![Resume Badge](https://img.shields.io/badge/resume-0077B5?style=for-the-badge&logo=resume&logoColor=white)](https://docs.google.com/document/d/1oqq7SOX-VfSNAwPcCo4rS5dtf5fm57ZNVGBg0nDRIcI/edit?usp=sharing)


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)
<h2> :books: References</h2>
<ul>
  <li><p>Jonathan Lee, 'Notes on Naive Bayes Classifiers for Spam Filtering'. [Online].</p>
      <p>Available: https://courses.cs.washington.edu/courses/cse312/18sp/lectures/naive-bayes/naivebayesnotes.pdf</p>
  </li>
  <li><p>Wikipedia.org, 'Naive Bayes Classifier'. [Online].</p>
      <p>Available: https://en.wikipedia.org/wiki/Naive_Bayes_classifier</p>
  </li>
  <li><p>Youtube.com, 'Naive Bayes for Spam Detection'. [Online].</p>
      <p>Available: https://www.youtube.com/watch?v=8aZNAmWKGfs</p>
  </li>
  <li><p>Youtube.com, 'Text Classification Using Naive Bayes'. [Online].</p>
      <p>Available: https://www.youtube.com/watch?v=EGKeC2S44Rs</p>
  </li>
  <li><p>Manisha-sirsat.blogspot.com, 'What is Confusion Matrix and Advanced Classification Metrics?'. [Online].</p>
      <p>Available: https://manisha-sirsat.blogspot.com/2019/04/confusion-matrix.html</p>
  </li>
  <li><p>Pythonforengineers.com, 'Build a Spam Filter'. [Online].</p>
      <p>Available: https://www.pythonforengineers.com/build-a-spam-filter/</p>
  </li>
</ul>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)











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

        git clone https://github.com/apoorvaKR12695/face-emotion-recognition-

  
Install dependencies

          pip install -r requirement.txt
  
Start local webcam

           streamlit run app.py
