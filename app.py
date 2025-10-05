import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import initializers
from PIL import Image

st.set_page_config(page_title="Face Mask Detector", layout="wide")

st.title("Face Mask Detector Demo")
st.write("Upload an image and the app will detect faces and predict mask/no-mask.")

# Initializer compatibility wrappers (same approach as GUI script)
class GlorotUniform(initializers.GlorotUniform):
    def __init__(self, seed=None, dtype=None):
        super().__init__(seed=seed)

class Zeros(initializers.Zeros):
    def __init__(self, dtype=None):
        super().__init__()

class Ones(initializers.Ones):
    def __init__(self, dtype=None):
        super().__init__()

class Constant(initializers.Constant):
    def __init__(self, value=0.0, dtype=None):
        super().__init__(value=value)

class RandomNormal(initializers.RandomNormal):
    def __init__(self, mean=0.0, stddev=1.0, seed=None, dtype=None):
        super().__init__(mean=mean, stddev=stddev, seed=seed)

class HeNormal(initializers.HeNormal):
    def __init__(self, seed=None, dtype=None):
        super().__init__(seed=seed)

class RandomUniform(initializers.RandomUniform):
    def __init__(self, minval=-0.05, maxval=0.05, seed=None, dtype=None):
        super().__init__(minval=minval, maxval=maxval, seed=seed)

class VarianceScaling(initializers.VarianceScaling):
    def __init__(self, scale=1.0, mode='fan_in', distribution='truncated_normal', seed=None, dtype=None):
        super().__init__(scale=scale, mode=mode, distribution=distribution, seed=seed)

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
    # Load model with custom_objects to be robust across Keras versions
    model = load_model(path, custom_objects=custom_inits)
    return model

model = None
try:
    with st.spinner('Loading model...'):
        model = load_mask_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
if uploaded_file is not None and model is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    orig = img.copy()
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

    # Convert BGR -> RGB for display
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    st.image(orig, channels='RGB', use_column_width=True)

st.write("---")
st.write("Notes: For live webcam support you'd need Streamlit's experimental camera input or run locally.")
