# Bird vs Drone Classification using Deep Learning

## Project Overview
In this project, I have built a deep learning model to classify aerial images into two categories: Bird and Drone. This problem is important in areas like airport safety, surveillance, and wildlife monitoring.

## Objective
The main goal was to correctly identify whether an object in an aerial image is a bird or a drone using image classification techniques.

## Dataset
The dataset contains images of birds and drones divided into:
- Training set
- Validation set
- Test set

## Steps Performed
1. Data preprocessing (resizing images, normalization)
2. Data augmentation (rotation, flipping, zoom)
3. Built a Custom CNN model
4. Applied Transfer Learning using MobileNetV2
5. Trained and evaluated both models
6. Compared performance of both models
7. Created a Streamlit app for prediction

## Model Performance
- Custom CNN Accuracy: around 80–85%
- MobileNetV2 Accuracy: around 97%

MobileNetV2 performed better and was selected as the final model.

## Deployment
A simple Streamlit app was created where users can upload an image and get prediction results.

Live App Link:
[Open Streamlit App](https://bird-drone-classifier-tmrdwqrgrnqyg3kb7yybbs.streamlit.app/)

## Tools & Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Streamlit

## Conclusion
This project helped me understand how deep learning models work for image classification. I also learned how to deploy a model using Streamlit.

## Experience
So far, I had mostly worked with structured data in the form of rows and columns. This is the first time I worked with an image dataset.
It was a valuable experience with lots of learning, new challenges, and small achievements. I feel more confident and upgraded in my skills after 
completing this project. 

## Future Improvements
- Improve dataset size
- Add real-time detection using YOLO
- Improve deployment with full model integration

