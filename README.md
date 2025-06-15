# Hand-Gesture-Translator

 🖐️ Hand Gesture to English Letter Translator using MediaPipe

This project is a **Python-based Hand Gesture Recognition system** that uses a webcam or uploaded images to detect hand signs (A–Z) and translates them into corresponding English letters. It is designed to assist people with speech or hearing impairments in communicating more easily.

 📸 Features

- ✅ Detects real hand gestures from images or webcam
- 🔠 Classifies hand signs into English alphabets (A–Z)
- 🤝 Built to support communication for deaf and mute individuals
- 📦 Uses **MediaPipe**, **OpenCV**, and **scikit-learn**

🚀 Technologies Used

| Tool    | Purpose |
|------   |---------|
| [MediaPipe](https://google.github.io/mediapipe/) | Hand tracking and landmark detection |
| OpenCV  | Image capture and drawing |
| scikit-learn| Classification using KNN |
| Colab / Jupyter Notebook| For development and demonstration |

---

🧠 Model

- Currently uses a **K-Nearest Neighbors (KNN)** classifier
- Training data consists of **hand landmark coordinates** for each letter (A–Z)
- You can expand it using your own dataset (see `data_collection.ipynb`)



