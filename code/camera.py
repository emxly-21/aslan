
import tensorflow as tf
import numpy as np
import cv2 as cv
aslan_model = tf.keras.models.load_model('../model/')

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
    # Our operations on the frame come here

    print(frame.shape)
    cropped_frame = frame[0:28, 0:28, :]
    gray = cv.cvtColor(cropped_frame, cv.COLOR_BGR2GRAY)
    gray = np.reshape(gray, (1,784))

    # run gray through the recognizer
    logits = aslan_model.predict(gray)
    output = tf.argmax(logits, 1).numpy()[0]
    print("logits.shape: ", logits.shape)
    print(output)

    cv.putText(frame, str(output), (10,500), cv.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 2, cv.LINE_AA)


    # Display the resulting frame
    cv.imshow('frame', frame)
    if cv.waitKey(1000) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

