import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

def train_return_model():

    data = pd.read_csv("digit-recognizer/train.csv").to_numpy()
    clf = DecisionTreeClassifier()

    #training model dataset

    x_train = data[0:, 1:]
    y_train = data[0:, 0]

    clf.fit(x_train, y_train)

    return clf

#testing model dataset

def test_model(data, clf):
    x_test = data[21000:, 1:]
    actual_label = data[21000:, 0]

    # INFERENCE

    d = x_test[8]
    d.shape = (28,28)

    print( clf.predict([x_test[8] ]) )
    plt.show()

def predict_image(image, model):

    print('Converting to Grayscale')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (28,28))
    plt.imshow(255-gray_image, cmap='gray')
    flat_image = np.array(gray_image).flatten()
    number = model.predict([flat_image])
    return number[0]

