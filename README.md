# 🧠 Multimodal Depression Detection System

A machine learning-based system designed to detect signs of depression by analyzing both **audio** and **textual** responses from participants. This project leverages multimodal data from the **DAIC-WOZ dataset** to improve detection accuracy and provide a better understanding of mental health indicators.

---

## 📌 Features

- Extracts **audio features** like pitch, energy, and MFCC using Librosa.
- Extracts **textual features** like polarity, subjectivity, and noun ratio using NLTK and spaCy.
- Combines both audio and text features for **multimodal classification**.
- Achieves up to **85% accuracy**, outperforming unimodal baselines.
- Modular ML pipeline with preprocessing, training, and evaluation scripts.

---

## 🛠️ Tools & Technologies

- Python
- Librosa
- NLTK
- Scikit-learn
- spaCy / TextBlob
- PyAudioAnalysis
- DAIC-WOZ Dataset
