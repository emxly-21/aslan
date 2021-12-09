
#  import tensorflow as tf
import numpy as np
import cv2 as cv
from constants import *
import os
import uuid

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
    small_cropped_frame = cv.resize(gray, (mnist_dimensions, mnist_dimensions))

    cv.imshow('small_cropped_frame', small_cropped_frame)

    filename = "color_" + cur_letter + "_" + str(uuid.uuid4()) + ".png"
    cv.imwrite(filename, small_cropped_frame)

    cv.rectangle(frame,(frame_x_start,frame_y_start),(frame_x_end,frame_y_end),(0,255,0),3)
    #  cv.putText(frame, text, (50,500), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv.LINE_AA)

    # Display the resulting frame
    cv.imshow('frame', frame)


if __name__ == '__main__':

    cap = cv.VideoCapture(0)
    absolute_path = project_dir + "custom_dataset/A/" + cur_letter + "/"

    if not os.path.exists(absolute_path): 
        os.makedirs(absolute_path)

    os.chdir(absolute_path)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        handle_frame(frame)

        if cv.waitKey(sample_interval) == ord('q'):
        #  if cv.waitKey(1000) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

