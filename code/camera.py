
import tensorflow as tf
import numpy as np
import cv2 as cv
from constants import *




def handle_frame(frame): 
    frame = cv.flip(frame, 1)

    # get the center 28 by 28 pixels
    height, width = frame.shape[:2]

    offset = 0
    frame_x_start = width // 2 - (sample_dimensions // 2) + offset
    frame_x_end = frame_x_start + sample_dimensions + offset
    frame_y_start = height // 2 - (sample_dimensions // 2) + offset
    frame_y_end = frame_y_start + sample_dimensions + offset

    cropped_frame = frame[frame_y_start:frame_y_end, frame_x_start:frame_x_end, :]
    gray = cv.cvtColor(cropped_frame, cv.COLOR_BGR2GRAY)
    gray = cv.resize(gray, (mnist_dimensions, mnist_dimensions))
    grayPixelated = cv.resize(gray, (sample_dimensions, sample_dimensions))
    cv.imshow('grayPixelated', grayPixelated)
    gray = np.reshape(gray, (1, mnist_dimensions, mnist_dimensions))
    gray = gray / 255

    # run gray through the recognizer
    logits = aslan_model.predict(gray)
    output = tf.nn.top_k(logits, 5, sorted=True)
    outputProbabilities = output[0].numpy()[0]
    outputValues = output[1].numpy()[0]

    #  print("outputProbabilities: ", outputProbabilities)
    #  print("outputValues: ", outputValues)

    text = ""
    if outputProbabilities.max() < MIN_ACCURACY_THRESHOLD: 
        text = "Letter Unrecognized"
    else: 
        for i in range(5): 
            prob = outputProbabilities[i]
            if (prob < MIN_ACCURACY_THRESHOLD): 
                continue
            value = outputValues[i]
            letter = LETTERS[value + 1]
            percentage = "{:.0%}".format(prob)
            text += "(" + letter +  ", " + percentage + ")  "

    cv.rectangle(frame,(frame_x_start,frame_y_start),(frame_x_end,frame_y_end),(0,255,0),3)
    cv.putText(frame, text, (50,500), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv.LINE_AA)

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

        #  if cv.waitKey(200) == ord('q'):
        if cv.waitKey(sample_interval) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

