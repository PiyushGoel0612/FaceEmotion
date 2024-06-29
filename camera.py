import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU, Activation
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator


faceCascade = cv2.CascadeClassifier('haarcascade_frontalface.xml')
mm = tf.saved_model.load('newNew')
infer = mm.signatures["serving_default"]
k = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []

    for (x,y,w,h) in features:
        input_data = tf.convert_to_tensor(cv2.resize(np.array([i[x:x+w] for i in gray_img[y:y+h]]), (48,48)).reshape(1,48,48,1)/255, dtype=tf.float32)
        predictions = infer(input_data)
        predicted_values = predictions['out_layer'].numpy()
        cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
        cv2.putText(img, k[np.argmax(predicted_values[0])], (x,y-4), cv2.FONT_HERSHEY_SIMPLEX, .8, color, 1, cv2.LINE_AA)
        coords.append((x,y,w,h))

    return coords, img 

def open_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        coords, frame = draw_boundary(frame, faceCascade, 1.1, 10, (255,255,255))
        cv2.imshow('Camera Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    open_camera()
