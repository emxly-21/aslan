
import tensorflow as tf
import numpy as np
import cv2 as cv
from constants import *


def handle_frame(frame): 
    frame = cv.flip(frame, 1)

    # get the center 28 by 28 pixels
    height, width = frame.shape[:2]
    frame_x_start = width // 2 - (sample_dimensions // 2)
    frame_x_end = frame_x_start + sample_dimensions
    frame_y_start = height // 2 - (sample_dimensions // 2)
    frame_y_end = frame_y_start + sample_dimensions

    #  print(frame.shape)
    cropped_frame = frame[frame_x_start:frame_x_end, frame_y_start:frame_y_end, :]
    gray = cv.cvtColor(cropped_frame, cv.COLOR_BGR2GRAY)
    gray = cv.resize(gray, (mnist_dimensions, mnist_dimensions))
    gray = np.reshape(gray, (1, mnist_dimensions * mnist_dimensions))

    # run gray through the recognizer
    logits = aslan_model.predict(gray)
    output = tf.argmax(logits, 1)
    #  output = str(output.numpy()[0])
    output = output.numpy()[0]
    #  print(logits)
    #  print("logits.shape: ", logits.shape)
    print(output)

    letter = LETTERS[output + 1]
    cv.rectangle(frame,(frame_x_start,frame_y_start),(frame_x_end,frame_y_end),(0,255,0),3)
    cv.putText(frame, letter, (50,500), cv.FONT_HERSHEY_SIMPLEX, 4, (0,0,255), 2, cv.LINE_AA)

    # Display the resulting frame
    cv.imshow('frame', frame)


if __name__ == '__main__':
    aslan_model = tf.keras.models.load_model('../model/')

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        handle_frame(frame)

        if cv.waitKey(200) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

