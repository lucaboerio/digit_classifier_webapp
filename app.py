import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas

from PIL import Image
import torch
import torch.nn as nn

import torch.nn.functional as F
import psycopg2
from datetime import datetime
import io
import base64

# Neural Network Model Definition
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Database Connection
def get_db_connection():
    return psycopg2.connect(
        dbname='digits_db',
        user='postgres',
        password='postgres',
        host='db'
    )

# Save prediction to database
def save_prediction(digit, probability):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO predictions (digit, probability, timestamp) VALUES (%s, %s, %s)",
        (digit, probability, datetime.now())
    )
    conn.commit()
    cur.close()
    conn.close()

# Get all predictions from database
def get_predictions():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT digit, probability, timestamp FROM predictions ORDER BY timestamp DESC")
    predictions = cur.fetchall()
    cur.close()
    conn.close()
    return predictions

# Initialize the model
model = DigitClassifier()
# Load the pre-trained weights
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()  # Set the model to evaluation mode

# Streamlit UI
st.set_page_config(layout="centered")  # oppure layout="wide"
st.title('Digit Classifier App')

# Create a canvas for drawing
drawing_mode = st.checkbox("Drawing Mode", True)
stroke_width = st.slider("Spessore del tratto: ", 1, 25, 8)
stroke_color = st.color_picker("Colore del tratto: ", "#000000")

# Create two columns with adjusted ratios
col1, col2 = st.columns([1, 1])

with col1:
    # Create a canvas component with responsive width
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="#000000",
        width=280,
        height=280,
        drawing_mode="freedraw" if drawing_mode else "transform",
        key="canvas",
        display_toolbar=True,
    )

with col2:
    st.markdown("### Predizione")
    # Add a button to perform prediction
    if st.button('Predici', key='predict_button', use_container_width=True):
        if canvas_result.image_data is not None:
            # Process the image
            image = Image.fromarray(canvas_result.image_data.astype('uint8'))
            image = image.convert('L')  # Convert to grayscale
            image = image.resize((28, 28))  # Resize to MNIST format
            
            # Convert to tensor and normalize
            tensor = torch.FloatTensor(np.array(image)).unsqueeze(0).unsqueeze(0) / 255.0
            
            # Make prediction
            with torch.no_grad():
                output = model(tensor)
                prediction = output.argmax(dim=1).item()
                probability = torch.exp(output.max()).item()
            
            # Save to database
            save_prediction(prediction, probability)
            
            # Display results
            st.write(f'Predicted Digit: {prediction}')
            st.write(f"Confidence: {probability:.2f}")

# Display predictions history
st.markdown('---')
st.subheader('Cronologia Previsioni')

# Create a container with improved styling
with st.container():
    predictions_container = st.empty()
    with predictions_container.container():
        predictions = get_predictions()
        for digit, prob, timestamp in predictions:
            with st.container():
                cols = st.columns([1, 1, 2])
                with cols[0]:
                    st.write(f'Numero: {digit}')
                with cols[1]:
                    st.write(f'Probabilità: {prob:.2f}')
                with cols[2]:
                    st.write(f'Data: {timestamp}')
                st.markdown('---')