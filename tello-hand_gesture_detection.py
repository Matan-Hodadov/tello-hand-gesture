# import necessary packages
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

from djitellopy import Tello

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

w, h = 360, 240
startCounter = 0  # for no Flight 1   - for flight 0


def move(className: str, myDrone):
    udv = 0  # up down velocity
    fbv = 0  # for back velocity
    speed = 20
    if className == "okay":
        myDrone.land()
        return
    elif className == "thumbs up" or className == "call me":
        udv = speed
    elif className == "thumbs down":
        udv = -speed
    elif className == "fist":
        fbv = speed
    elif className == "stop" or className == "live long":
        fbv = -speed
    elif className == "peace":
        myDrone.flip('r')
        return

    myDrone.up_down_velocity = udv
    myDrone.for_back_velocity = fbv
    if myDrone.send_rc_control:
        myDrone.send_rc_control(myDrone.left_right_velocity,
                                myDrone.for_back_velocity,
                                myDrone.up_down_velocity,
                                myDrone.yaw_velocity)
    return


def initializeTello():
    myDrone = Tello()
    myDrone.connect()
    myDrone.for_back_velocity = 0
    myDrone.left_right_velocity = 0
    myDrone.up_down_velocity = 0
    myDrone.yaw_velocity = 0
    myDrone.speed = 0
    myDrone.streamoff()
    myDrone.streamon()
    print(myDrone.get_battery())
    return myDrone


def telloGetFrame(myDrone, w=360, h=240):
    myFrame = myDrone.get_frame_read()
    myFrame = myFrame.frame
    img = cv2.resize(myFrame, (w, h))
    return img


myDrone = initializeTello()

while True:
    # Flight
    if startCounter == 0:
        myDrone.takeoff()
        myDrone.move_up(20)
        startCounter = 1
    # Step 1
    frame = telloGetFrame(myDrone, w, h)
    # if myDrone.get_height() < 15:
    #     myDrone.land()
    #     break

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # print(result)
    
    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

        # Predict gesture
        prediction = model.predict([landmarks])
        # print(prediction)
        classID = np.argmax(prediction)
        className = classNames[classID]
    move(className, myDrone)

    # show the prediction on the frame
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Battery: " + str(myDrone.get_battery()), (10, 200), cv2.FONT_HERSHEY_SIMPLEX,
                0.33, (255, 0, 0), 1, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame)

    if cv2.waitKey(1) == ord('q'):
        myDrone.land()
        print("battery: " + str(myDrone.get_battery()))
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()