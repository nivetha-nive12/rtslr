import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import copy
import itertools
import string

MODEL_PATH = "dataset_download/isl-sign-recognition/model.h5"

model = tf.keras.models.load_model(MODEL_PATH)

alphabet = ['1','2','3','4','5','6','7','8','9'] + list(string.ascii_uppercase)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for landmark in landmarks.landmark:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp = copy.deepcopy(landmark_list)

    base_x, base_y = temp[0]

    for point in temp:
        point[0] -= base_x
        point[1] -= base_y

    temp = list(itertools.chain.from_iterable(temp))

    max_value = max(map(abs, temp))

    if max_value == 0:
        max_value = 1

    return [x / max_value for x in temp]


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print("Camera started.")
print("Show an ISL alphabet/digit sign.")
print("Press Q to quit.")

with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()

        if not ret:
            print("ERROR: Could not read camera frame.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        prediction_text = "No hand"

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                landmark_list = calc_landmark_list(
                    frame,
                    hand_landmarks
                )

                features = pre_process_landmark(landmark_list)

                input_data = np.array(
                    [features],
                    dtype=np.float32
                )

                predictions = model(input_data, training=False).numpy()

                class_index = int(np.argmax(predictions[0]))
                confidence = float(np.max(predictions[0])) * 100

                prediction_text = (
                    f"{alphabet[class_index]} "
                    f"{confidence:.1f}%"
                )

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        cv2.putText(
            frame,
            prediction_text,
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )

        cv2.imshow("ISL Alphabet Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()