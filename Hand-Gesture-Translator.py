!pip install mediapipe opencv-python-headless numpy scikit-learn


from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
from google.colab import files
from google.colab.patches import cv2_imshow
import mediapipe as mp
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# (sign meanings)
gesture_meanings = {
    'A': 'Hello / Attention',
    'B': 'Stop / Calm down',
    'C': 'Cup / Hold',
    'D': 'Point / Indicate',
    'E': 'Enough',
    'F': 'OK / Perfect',
    'G': 'Small / Pinch',
    'H': 'Here',
    'I': 'Me / Myself',
    'J': 'You',
    'K': 'Go',
    'L': 'Love / Like',
    'M': 'Mother / Mom',
    'N': 'No',
    'O': 'Circle / Round',
    'P': 'Please',
    'Q': 'Question',
    'R': 'Ready',
    'S': 'Stop / Silence',
    'T': 'Thanks',
    'U': 'You / Together',
    'V': 'Victory / Peace',
    'W': 'Wait',
    'X': 'Cancel / Wrong',
    'Y': 'Yes / Affirm',
    'Z': 'Zigzag / Sign'
}


print("📁 Upload a hand gesture image showing a letter (A–Z)...")
uploaded = files.upload()


image_path = list(uploaded.keys())[0]
img = cv2.imread(image_path)
if img is None:
    raise ValueError(f"Image not found: {image_path}")


img = cv2.resize(img, (640, 480))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)


X_train = [
    [0.1]*42,  # A
    [0.2]*42,  # B
    [0.3]*42,  # C
    [0.4]*42,  # D
    [0.5]*42   # E
]
y_train = ['A', 'B', 'C', 'D', 'E']


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)


results = hands.process(img_rgb)

if results.multi_hand_landmarks:
    print("✅ Hand detected!")

    for hand_landmarks in results.multi_hand_landmarks:
       
        mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

       
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append(lm.x)
            landmarks.append(lm.y)

        landmarks = landmarks[:42] 

         
        prediction = knn.predict([landmarks])[0]
        meaning = gesture_meanings.get(prediction, "Unknown")

        
        cv2.putText(img, f"Letter: {prediction}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, f"Meaning: {meaning}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 0), 2)

        print(f"🔤 Predicted Letter: {prediction}")
        print(f"📖 Meaning: {meaning}")
else:
    print("❌ No hand detected. Try again with a clearer image.")


cv2_imshow(img)
hands.close()
