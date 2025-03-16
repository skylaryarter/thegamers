import cv2
import mediapipe as mp
import math

# Import the tasks API for gesture recognition
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions
from mediapipe.tasks.python import BaseOptions
from mediapipe.framework.formats import landmark_pb2

import webbrowser
import time
import pyautogui

# Path to the gesture recognition model
model_path = "gesture_recognizer.task"  # Update this to the correct path where the model is saved, if not in current directory

# Initialize the Gesture Recognizer
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=1
)
gesture_recognizer = GestureRecognizer.create_from_options(options)


## DEFINE THUMB LEFT
# Initialize Mediapipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Define a function to calculate the distance between two points
def calculate_distance(point1, point2):
    return math.hypot(point2[0]-point1[0], point2[1]-point1[1])

def fingers_y(y1, y2, y3, y4, y5):
    if (abs((y1 + y2 + y3 + y4 + y5)/5 - y1) < 0.009):
        return True
    return False

def recognize_thumb_left(hand_landmarks):
    # Extract necessary landmarks
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # Calculate y distance between thumb tip and other finger tips
    distance_y = fingers_y(thumb_tip.y, index_tip.y, middle_tip.y, ring_tip.y, pinky_tip.y)
    other_dist = calculate_distance((thumb_tip.y, thumb_tip.x), (index_tip.y, index_tip.x))

    # Check if thumb and index are close and other fingers are open
    if distance_y:
        if (other_dist > 0.1):
            return "Left Thumb Gesture"
    return "Unknown"


def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # 0 is the default webcam

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the image horizontally and convert the BGR image to RGB.
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)

            # Convert the image to a Mediapipe Image object for the gesture recognizer
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Recognize gesture
                    # gesture = recognize_palm(hand_landmarks)
                    gesture = recognize_thumb_left(hand_landmarks)

            # Perform gesture recognition on the image
            result = gesture_recognizer.recognize(mp_image)

            # Draw the gesture recognition results on the image
            if result.gestures:
                recognized_gesture = result.gestures[0][0].category_name
                confidence = result.gestures[0][0].score

                if recognized_gesture == "Open_Palm":
                    webbrowser.open('https://www.quaxio.com/2048/', new=2)
                    # Make sure to allow for time between recognized gestures so only one window is opened
                    time.sleep(5)

                # Example of pressing keys with pyautogui based on recognized gesture
                if recognized_gesture == "Thumb_Up":
                    pyautogui.press("w")
                    time.sleep(1)
                elif recognized_gesture == "Thumb_Down":
                    pyautogui.press("s")
                    time.sleep(1)
                elif recognized_gesture == "ILoveYou":
                    pyautogui.press("a")
                    time.sleep(1)
                elif recognized_gesture == "Closed_Fist":
                    pyautogui.press("d")
                    time.sleep(1)
                elif recognized_gesture == "Victory":
                    pyautogui.press("space")
                    time.sleep(1)
                elif recognized_gesture == "Pointing_Up":
                    pyautogui.press("z")
                    time.sleep(1)

                # Display recognized gesture and confidence 
                cv2.putText(image, f"Gesture: {recognized_gesture} ({confidence:.2f})", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display the resulting image (can comment this out for better performance later on)
            cv2.imshow('Gesture Recognition', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
