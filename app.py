import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Try to import tensorflow
TF_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras import initializers
except Exception:
    TF_AVAILABLE = False

st.set_page_config(page_title="Face Mask Detector", layout="wide")

st.title("🟢 Face Mask Detector (Webcam - Streamlit Cloud)")
st.write("Use your webcam to capture a live image. The app will detect faces and classify them as **Mask / No Mask**.")

# -----------------------------
#  Model Loading (if available)
# -----------------------------
if TF_AVAILABLE:
    # Custom initializer fixes for model loading
    class GlorotUniform(initializers.GlorotUniform): pass
    class Zeros(initializers.Zeros): pass
    class Ones(initializers.Ones): pass
    class Constant(initializers.Constant): pass
    class RandomNormal(initializers.RandomNormal): pass
    class HeNormal(initializers.HeNormal): pass
    class RandomUniform(initializers.RandomUniform): pass
    class VarianceScaling(initializers.VarianceScaling): pass

    custom_inits = {
        'GlorotUniform': GlorotUniform,
        'Zeros': Zeros,
        'Ones': Ones,
        'Constant': Constant,
        'RandomNormal': RandomNormal,
        'HeNormal': HeNormal,
        'RandomUniform': RandomUniform,
        'VarianceScaling': VarianceScaling,
    }

    @st.cache_resource
    def load_mask_model(path='mask_detector_model.h5'):
        return load_model(path, custom_objects=custom_inits)

    model = None
    try:
        with st.spinner("🔄 Loading model..."):
            model = load_mask_model()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"⚠️ Failed to load model: {e}")
else:
    model = None
    st.warning("⚠️ TensorFlow not available in this environment. Model inference disabled.")

# Haarcascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

st.write("### 📷 Capture an image from your webcam")
img_file = st.camera_input("Take a photo")

if img_file is not None:
    if model is None:
        st.error("Model not loaded. Cannot perform detection.")
    else:
        # Read image
        image = Image.open(img_file)
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        orig = img.copy()

        # Convert to grayscale and detect faces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        labels = ['Mask', 'No Mask']
        colors = [(0, 255, 0), (0, 0, 255)]

        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face) / 255.0
            face = np.expand_dims(face, axis=0)

            (mask, withoutMask) = model.predict(face)[0]
            label = labels[0] if mask > withoutMask else labels[1]
            color = colors[0] if mask > withoutMask else colors[1]

            cv2.rectangle(orig, (x, y), (x+w, y+h), color, 2)
            cv2.putText(orig, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Display output
        output_img = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        st.image(output_img, caption="Detection Result", use_container_width=True)

st.info("ℹ️ On Streamlit Cloud, you capture one frame at a time. For continuous real-time webcam detection, run this app locally with `cv2.VideoCapture(0)`.")

