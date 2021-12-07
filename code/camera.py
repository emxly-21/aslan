
import tensorflow as tf
import numpy as np
import cv2 as cv

aslan_model = tf.keras.models.load_model('../model/')
mnist_dimensions = 28
sample_dimensions = mnist_dimensions * 10

LETTERS = {1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f", 7: "g", 8: "h",
9: "i", 11: "k", 12: "l", 13: "m", 14: "n", 15: "o", 16: "p", 17: "q", 18: "r",
19: "s", 20: "t", 21: "u", 22: "v", 23: "w", 24: "x", 25: "y"}

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

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
    gray = np.reshape(gray, (1,mnist_dimensions * mnist_dimensions))

    # run gray through the recognizer
    logits = aslan_model.predict(gray)
    output = tf.argmax(logits, 1)
    #  output = str(output.numpy()[0])
    output = output.numpy()[0]
    #  print(logits)
    #  print("logits.shape: ", logits.shape)
    print(output)

    letter = LETTERS[output]
    cv.rectangle(frame,(frame_x_start,frame_y_start),(frame_x_end,frame_y_end),(0,255,0),3)
    cv.putText(frame, letter, (50,500), cv.FONT_HERSHEY_SIMPLEX, 4, (0,0,255), 2, cv.LINE_AA)

    # Display the resulting frame
    cv.imshow('frame', frame)
    if cv.waitKey(200) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

