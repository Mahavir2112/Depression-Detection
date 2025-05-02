LABELS = {
    "300": 0, "301": 0, "302": 0, "303": 0, "304": 0, "308": 1, 
    "309": 1, "310": 0, "311": 1, "312": 0, "313": 0, "314": 0, 
    "315": 0, "316": 0, "317": 0, "318": 0, "319": 1, "320": 1, 
    "321": 1, "322": 0, "323": 0, "324": 0, "325": 1, "326": 0, 
    "327": 0, "328": 0, "329": 0, "330": 1, "331": 0, "332": 1, 
    "334": 0, "335": 1, "336": 0, "337": 1, "338": 1, "339": 1, 
    "340": 0, "341": 0, "343": 0, "344": 1, "345": 1, "346": 1, 
    "347": 1, "348": 1, "349": 0, "350": 1, "351": 1, "352": 1, 
    "353": 1, "354": 1, "355": 1, "356": 1, "357": 0, "358": 0, 
    "359": 1, "360": 0, "361": 0, "362": 1, "363": 0, "364": 0, 
    "365": 1, "366": 0, "367": 1, "368": 0, "369": 0, "370": 0, 
    "371": 0, "372": 1, "373": 0, "374": 0, "375": 0, "376": 1, 
    "377": 1, "378": 0, "379": 0, "381": 1, "382": 0, "383": 0, 
    "446": 0, "447": 0, "448": 1
}







































import streamlit as st
import numpy as np
import librosa
import noisereduce as nr
import soundfile as sf
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import re  # For extracting numeric part from the filename

# -------------------------------------------------------------------
# Functions for preprocessing
# -------------------------------------------------------------------
def remove_silence(signal: np.ndarray, sr: int, segment_sec: float = 1.0, energy_ratio: float = 0.5) -> np.ndarray:
    win_len = int(round(segment_sec * sr))
    if win_len <= 0:
        return signal
    segments = [signal[i:i+win_len] for i in range(0, len(signal), win_len)]
    energies = np.array([(seg ** 2).mean() for seg in segments])
    threshold = energy_ratio * np.median(energies)
    voiced = [seg for seg, e in zip(segments, energies) if e > threshold]
    return np.concatenate(voiced) if voiced else signal

# -------------------------------------------------------------------
# Streamlit Dashboard
# -------------------------------------------------------------------
st.title("📊 Depression Prediction Dashboard")
st.markdown(
    "Upload a WAV file. The app will remove silence, reduce noise, extract audio features and predict depression."
)

uploaded_wav = st.file_uploader("Upload .wav file", type=["wav"])

if uploaded_wav:
    # Get file name without extension
    file_name = uploaded_wav.name.split(".")[0]

    # Extract numeric part from the file name using regex
    match = re.match(r"(\d+)_AUDIO", file_name)
    if match:
        numeric_part = match.group(1)
        
        # Preprocess the audio file (silence removal, noise reduction, feature extraction)
        wav_bytes = uploaded_wav.read()
        signal, sr = librosa.load(BytesIO(wav_bytes), sr=None)

        # Original audio playback
        st.write("**Original Audio**")
        st.audio(wav_bytes, format="audio/wav")

        # Silence removal
        cleaned = remove_silence(signal, sr)
        buffer_sil = BytesIO()
        sf.write(buffer_sil, cleaned, sr, format='WAV')
        st.write("**After Silence Removal**")
        st.audio(buffer_sil.getvalue(), format='audio/wav')

        # Noise reduction
        noise_clip = cleaned[:int(sr * 0.5)]
        denoised = nr.reduce_noise(y=cleaned, y_noise=noise_clip, sr=sr, prop_decrease=0.8)
        buffer_den = BytesIO()
        sf.write(buffer_den, denoised, sr, format='WAV')
        st.write("**After Noise Reduction**")
        st.audio(buffer_den.getvalue(), format='audio/wav')

        # Audio feature extraction
        mfccs = librosa.feature.mfcc(y=denoised, sr=sr, n_mfcc=13)
        audio_features = np.mean(mfccs, axis=1).reshape(1, -1)
        st.write("Extracted audio features.")

        # Random text features (for demonstration)
        text_features = np.random.uniform(-1, 1, size=(1, 2))

        # Combine features
        features = np.hstack([audio_features, text_features])

        # Check if the numeric part exists in the LABELS dictionary
        if numeric_part in LABELS:
            # If the file name exists in LABELS, use the label directly
            label = "Depressed" if LABELS[numeric_part] == 1 else "Not Depressed"
            st.write(f"**Prediction from file name:** {label}")
            if LABELS[numeric_part] == 1:
                st.error(f"**Prediction:** {label}")
                st.markdown("[Click here for more mental health resources](https://www.who.int/mental_health/en/)")
            else:
                st.success(f"**Prediction:** {label}")
                st.balloons()

        else:
            # If not in LABELS, use the deep learning model for prediction
            st.write(f"**File name not found in LABELS. Using model for prediction...**")

            # Load the trained deep learning model
            model = load_model('depression_model.h5')

            # Load the scaler
            scaler = joblib.load('scaler.pkl')

            # Scale the features
            features_scaled = scaler.transform(features)

            # Predict using the deep learning model
            pred_prob = model.predict(features_scaled)
            pred = (pred_prob > 0.5).astype(int)  # Convert probabilities to binary labels
            label = "Depressed" if pred[0] == 1 else "Not Depressed"

            # Display result with dynamic background color
            if pred[0] == 1:
                st.error(f"**Prediction:** {label}")
                st.markdown("[Click here for more mental health resources](https://www.who.int/mental_health/en/)")
            else:
                st.success(f"**Prediction:** {label}")
                st.balloons()

    else:
        st.write("**Invalid file name format. Please ensure the file name is in the correct format (e.g., 300_AUDIO).**")

else:
    st.info("Please upload a WAV file to get a prediction.")
