import cv2
import mediapipe as mp
import math

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

            # Draw the hand annotations on the image.
            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Recognize gesture
                    # gesture = recognize_palm(hand_landmarks)
                    gesture = recognize_thumb_left(hand_landmarks)
                    
                    # Display gesture near hand location
                    cv2.putText(image, gesture, 
                                (int(hand_landmarks.landmark[0].x * image.shape[1]), 
                                 int(hand_landmarks.landmark[0].y * image.shape[0]) - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Display the resulting image
            cv2.imshow('Gesture Recognition', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()