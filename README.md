# Face Mask Detector - Streamlit Demo

This repository contains a simple Streamlit demo that uses a pre-trained Keras H5 model (`mask_detector_model.h5`) to detect faces and predict whether a person is wearing a mask.

Files:
- `mask_mask_gui.py.py` - original GUI script using Tkinter + OpenCV.
- `app.py` - Streamlit demo (upload an image and it will detect faces and label them).
- `mask_detector_model.h5` - the pretrained model (not modified).
- `requirements.txt` - Python dependencies for deployment.

Run locally

1. Create a Python virtual environment and activate it.
2. Install dependencies:

   pip install -r requirements.txt

3. Run Streamlit:

   streamlit run app.py

Deploy to Streamlit Cloud

- Create a new repository on GitHub and push this project.
- On Streamlit Cloud, create a new app and point it to the GitHub repo; Streamlit Cloud will install dependencies from `requirements.txt` and start `streamlit run app.py` by default.

Deploy to Render (Web Service)

- On Render, create a new Web Service.
- Use the command `streamlit run app.py --server.port $PORT` as the start command.
- Ensure `requirements.txt` is present in the repo. Render will install the dependencies.

Notes and troubleshooting

- Model compatibility: If you see errors when loading the H5 model due to Keras version differences, `app.py` includes compatibility wrappers for several common initializers. If the model uses other initializers or custom layers, add them to the `custom_inits` or `custom_objects` mapping in `app.py`.
- For live webcam support you may need to run locally and use Streamlit's camera input or a separate front-end.
