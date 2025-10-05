import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import initializers
import tkinter as tk
from PIL import Image, ImageTk

# Load the pretrained face detector and mask detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Compatibility wrappers for initializer deserialization issues
# Some HDF5 models saved with older/newer Keras may include a 'dtype' key
# in initializer configs which causes TypeError during load on different
# Keras versions. These thin wrappers accept the dtype kwarg and forward
# the call to the real initializer.
class GlorotUniform(initializers.GlorotUniform):
    def __init__(self, seed=None, dtype=None):
        # older/newer Keras may pass dtype; ignore it and forward seed
        super().__init__(seed=seed)

class Zeros(initializers.Zeros):
    def __init__(self, dtype=None):
        super().__init__()

class Ones(initializers.Ones):
    def __init__(self, dtype=None):
        # Ones() in some Keras versions doesn't accept args; ignore dtype
        super().__init__()

class Constant(initializers.Constant):
    def __init__(self, value=0.0, dtype=None):
        # pass value through, ignore dtype
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

# Map class names used in saved configs to these wrappers so deserialization
# ignores unexpected kwargs like 'dtype'. Add more mappings if new errors show up.
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

# Load model with custom_objects to handle initializer compatibility
model = load_model('mask_detector_model.h5', custom_objects=custom_inits)

# Define labels and colors
labels = ['Mask', 'No Mask']
colors = [(0, 255, 0), (0, 0, 255)]

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create GUI window
window = tk.Tk()
window.title("Face Mask Detection")
window.geometry("800x600")

# Video frame in the UI
video_label = tk.Label(window)
video_label.pack()

def detect_and_display():
    ret, frame = cap.read()
    if not ret:
        window.after(10, detect_and_display)
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face) / 255.0
        face = np.expand_dims(face, axis=0)

        # Prediction
        (mask, withoutMask) = model.predict(face)[0]
        label = labels[0] if mask > withoutMask else labels[1]
        color = colors[0] if mask > withoutMask else colors[1]

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Convert frame for Tkinter
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    window.after(10, detect_and_display)

detect_and_display()

def on_closing():
    cap.release()
    cv2.destroyAllWindows()
    window.destroy()

window.protocol("WM_DELETE_WINDOW", on_closing)
window.mainloop()
