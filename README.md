# Digit_Recognition_System

Digit Recognition Model
This project presents a Digit Recognition Model utilizing a Convolutional Neural Network (CNN) built with TensorFlow and Keras to classify handwritten digits. The model is trained on the MNIST dataset and is capable of recognizing digits from 0 to 9.

Key Features & Technologies:

Convolutional Neural Network (CNN): The core of the model is a CNN, which is highly effective for image classification tasks.

TensorFlow & Keras: These powerful deep learning libraries are used to build, train, and manage the model.

OpenCV (cv2): The project includes an interactive drawing canvas created with OpenCV, allowing users to draw a digit in real time.

Real-time Prediction: The application can predict the drawn digit in real-time by pressing the spacebar.

Data Preprocessing: The drawn digit is resized to a 28x28 pixel grayscale image and normalized before being fed into the model for prediction.

How to Use:

To interact with the model, run the Jupyter Notebook. The user interface provides simple controls:

Spacebar: Press to get a real-time prediction of your drawn digit.

'c': Press to clear the canvas.

ESC: Press to exit the application window.

2. Email Spam Classifier
This repository contains an Email Spam Classifier that uses a classical machine learning approach to differentiate between "spam" and "ham" (non-spam) emails.

Key Features & Technologies:

Multinomial Naive Bayes Model: The classifier is based on the MultinomialNB algorithm from scikit-learn, a probabilistic classifier well-suited for text classification.

Bag-of-Words Vectorization: The CountVectorizer from scikit-learn is used to transform the raw email text into a numerical "Bag-of-Words" representation. This process counts the occurrences of words, which serves as input for the model.

Training Data: The model is trained on a dataset from a CSV file (emails.csv), which contains the email text and a corresponding spam label (1 for spam, 0 for ham).

Evaluation Metrics: The project uses classification_report, confusion_matrix, and accuracy_score from scikit-learn to evaluate the model's performance on a test set.

How to Use:

The notebook includes a function predictMessage that takes a message as input, vectorizes it using the trained CountVectorizer, and then uses the model to predict if it is spam or ham. The output will clearly state the classification.
