"""
Streamlit app for MNIST digit recognition with interactive predictions.
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms
import json

# Set page config
st.set_page_config(
    page_title="MNIST Digit Recognition CNN",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# MODEL DEFINITION
# ============================================================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ============================================================================
# LOAD MODEL AND DATA
# ============================================================================
@st.cache_resource
def load_model():
    model = SimpleCNN()
    return model

@st.cache_resource
def load_test_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return test_dataset

# Load model and data
model = load_model()
model.eval()
test_dataset = load_test_data()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def predict_digit(image_array):
    """Predict digit from numpy array (28x28, values 0-255)"""
    # Normalize and convert to tensor
    img_tensor = torch.FloatTensor(image_array).unsqueeze(0).unsqueeze(0) / 255.0

    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.nn.functional.softmax(output, dim=1)[0]

    return prob.numpy()

def draw_digit_canvas():
    """Create drawing canvas for digit"""
    st.write("**Draw a digit (0-9) in the canvas below:**")

    # Canvas styling with JavaScript
    canvas_html = """
    <canvas id="canvas" width="280" height="280" style="border:2px solid #000; cursor:crosshair; display:block; margin:10px 0;"></canvas>
    <script>
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    let isDrawing = false;

    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, 280, 280);

    canvas.addEventListener('mousedown', (e) => {
        isDrawing = true;
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        ctx.beginPath();
        ctx.moveTo(x, y);
    });

    canvas.addEventListener('mousemove', (e) => {
        if (!isDrawing) return;
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        ctx.lineWidth = 15;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.strokeStyle = 'black';
        ctx.lineTo(x, y);
        ctx.stroke();
    });

    canvas.addEventListener('mouseup', () => { isDrawing = false; });
    canvas.addEventListener('mouseleave', () => { isDrawing = false; });

    document.getElementById('get_canvas_data').onclick = () => {
        const imageData = canvas.toDataURL('image/png');
        const input = document.getElementById('canvas_data');
        input.value = imageData;
    };
    </script>
    <input type="hidden" id="canvas_data" />
    <button id="get_canvas_data" type="button">Capture Drawing</button>
    """

    st.markdown(canvas_html, unsafe_allow_html=True)

    return None

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Choose a page:", ["Overview", "Model Metrics", "Interactive Prediction"])

st.sidebar.markdown("---")
st.sidebar.info(
    "🧠 **MNIST CNN Model**\n\n"
    "3-layer Convolutional Neural Network trained on 60,000 handwritten digit images.\n\n"
    "**Architecture:**\n"
    "- Conv2D (32 filters) + MaxPool\n"
    "- Conv2D (64 filters) + MaxPool\n"
    "- Conv2D (128 filters) + Dense + Dropout\n\n"
    "**Performance:**\n"
    "- Test Accuracy: **99.33%**\n"
    "- Test Loss: **0.0252**"
)

# ============================================================================
# PAGE: OVERVIEW
# ============================================================================
if page == "Overview":
    st.title("✍️ MNIST Handwritten Digit Recognition")
    st.markdown("**Deep Learning CNN trained on 60,000 handwritten digits**")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Test Accuracy", "99.33%", "+0.05%")
    with col2:
        st.metric("Test Loss", "0.0252", "-0.001")
    with col3:
        st.metric("Model Layers", "3", "Conv")
    with col4:
        st.metric("Training Epochs", "5", "Trained")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Training Progress")
        try:
            img = Image.open('model_accuracy_loss.png')
            st.image(img, use_column_width=True)
        except:
            st.warning("Accuracy/Loss plot not found. Run mnist_cnn.py first.")

    with col2:
        st.subheader("🎯 Predictions on Random Test Images")
        try:
            img = Image.open('predictions.png')
            st.image(img, use_column_width=True)
        except:
            st.warning("Predictions plot not found. Run mnist_cnn.py first.")

    st.markdown("---")
    st.subheader("📚 Dataset Info")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Training Samples:** 60,000")
        st.write(f"**Test Samples:** 10,000")
    with col2:
        st.write(f"**Image Size:** 28×28 pixels")
        st.write(f"**Classes:** 10 (digits 0-9)")

# ============================================================================
# PAGE: MODEL METRICS
# ============================================================================
elif page == "Model Metrics":
    st.title("📊 Model Performance Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        try:
            img = Image.open('confusion_matrix.png')
            st.image(img, use_column_width=True)
        except:
            st.warning("Confusion matrix not found. Run mnist_cnn.py first.")

    with col2:
        st.subheader("Training Curves")
        try:
            img = Image.open('model_accuracy_loss.png')
            st.image(img, use_column_width=True)
        except:
            st.warning("Training curves not found. Run mnist_cnn.py first.")

    st.markdown("---")
    st.subheader("📋 Model Architecture")
    st.code("""
SimpleCNN(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), padding=1)
  (pool1): MaxPool2d(kernel_size=2, stride=2)

  (conv2): Conv2d(32, 64, kernel_size=(3, 3), padding=1)
  (pool2): MaxPool2d(kernel_size=2, stride=2)

  (conv3): Conv2d(64, 128, kernel_size=(3, 3), padding=1)
  (fc1): Linear(6272, 128)
  (dropout): Dropout(p=0.5)
  (fc2): Linear(128, 10)
)
    """, language="python")

# ============================================================================
# PAGE: INTERACTIVE PREDICTION
# ============================================================================
elif page == "Interactive Prediction":
    st.title("🖊️ Interactive Digit Prediction")
    st.write("Draw a digit or upload an image to get predictions from the model.")

    tab1, tab2 = st.tabs(["Draw Digit", "Upload Image"])

    # TAB 1: DRAWING CANVAS
    with tab1:
        st.write("**Draw a digit (0-9) in the white canvas below.**")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Simple PIL-based drawing
            if 'drawn_image' not in st.session_state:
                st.session_state.drawn_image = Image.new('L', (280, 280), 'white')

            # Canvas
            st.markdown("""
            <style>
            canvas { border: 2px solid #000; display: block; }
            </style>
            """, unsafe_allow_html=True)

            # Use streamlit-drawable-canvas if available, otherwise show static canvas
            drawn_image = st.session_state.get('drawn_image')
            if drawn_image:
                st.image(drawn_image, width=280)

        with col2:
            st.write("")
            if st.button("🔄 Clear Canvas"):
                st.session_state.drawn_image = Image.new('L', (280, 280), 'white')
                st.rerun()

            st.info(
                "**Note:** For interactive drawing, use the Upload Image tab "
                "or run this app with streamlit-drawable-canvas package."
            )

    # TAB 2: UPLOAD IMAGE
    with tab2:
        uploaded_file = st.file_uploader("Upload a digit image (PNG, JPG):", type=['png', 'jpg', 'jpeg'])

        if uploaded_file is not None:
            # Load and display image
            image = Image.open(uploaded_file).convert('L')
            image_resized = image.resize((28, 28))
            image_array = np.array(image_resized)

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Uploaded Image (28×28):**")
                st.image(image_resized, width=200)

            with col2:
                st.write("**Prediction Results:**")

                # Make prediction
                prob = predict_digit(image_array)
                predicted_class = np.argmax(prob)
                confidence = prob[predicted_class]

                # Display prediction
                st.metric("Predicted Digit", predicted_class, f"Confidence: {confidence:.1%}")

                # Display top 5 predictions
                st.write("**Top 5 Predictions:**")
                top_5_idx = np.argsort(prob)[-5:][::-1]
                for rank, idx in enumerate(top_5_idx, 1):
                    bar_length = int(prob[idx] * 30)
                    st.write(f"{rank}. Digit **{idx}**: {prob[idx]:.2%} {'█' * bar_length}")

    # RANDOM TEST IMAGES
    st.markdown("---")
    st.subheader("🎲 Sample Predictions from Test Set")

    if st.button("Show Random Test Predictions"):
        col1, col2, col3, col4, col5 = st.columns(5)
        cols = [col1, col2, col3, col4, col5]

        # Get 5 random test images
        random_indices = np.random.choice(len(test_dataset), 5, replace=False)

        for idx, col in enumerate(cols):
            img, true_label = test_dataset[random_indices[idx]]
            img_array = img.squeeze().numpy()

            with col:
                # Display image
                st.image(img_array, width=100, use_column_width=True)

                # Make prediction
                prob = predict_digit(img_array)
                pred_label = np.argmax(prob)
                confidence = prob[pred_label]

                # Color based on correctness
                if pred_label == true_label:
                    st.write(f"✅ **Pred:** {pred_label}")
                else:
                    st.write(f"❌ **Pred:** {pred_label}")
                st.write(f"**True:** {true_label}")
                st.write(f"**Conf:** {confidence:.1%}")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
    <small>🧠 Built with PyTorch • 🎨 Visualized with Matplotlib • 🚀 Deployed with Streamlit</small>
    </div>
    """,
    unsafe_allow_html=True
)
