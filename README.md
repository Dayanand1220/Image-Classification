🎯 AI Image Classification - Professional Deep Learning Web App

This project is a modern, professional Streamlit-based Image Classification Web App built using three powerful TensorFlow pretrained models — MobileNetV2, InceptionResNetV2, and EfficientNetB0.

✨ NEW: Enhanced Professional UI - Features a completely redesigned interface with modern styling, gradient backgrounds, interactive elements, and mobile-responsive design.

The app allows users to upload images and instantly view top-3 predicted classes with confidence scores, processing times, and detailed model information in a beautiful, user-friendly interface.

The goal is to demonstrate state-of-the-art deep learning architectures within a professional, production-ready web interface that rivals commercial AI applications.

🚀 Project Overview

This app combines the strength of modern CNN architectures from TensorFlow’s tf.keras.applications module, trained on the ImageNet dataset (1.2 million images, 1,000 categories). Users can switch between three pretrained models and observe differences in prediction accuracy and speed.

⚙️ Models Used Model Type Accuracy (Top-1) Key Feature 🧩 MobileNetV2 Lightweight CNN ~71% Optimized for mobile and real-time applications 🧠 InceptionResNetV2 Deep Hybrid Network ~80% Combines Inception and ResNet for high accuracy ⚡ EfficientNetB0 Scaled CNN ~77% Balances speed and accuracy efficiently

All three models are available free and pre-trained inside TensorFlow — no extra downloads required.

✨ Key Features

🔍 Three Model Options: Switch easily between MobileNetV2, InceptionResNetV2, and EfficientNetB0 using the sidebar.

📤 Upload & Predict: Upload any .jpg, .jpeg, or .png image for instant classification.

📊 Top-3 Predictions: Displays class labels with confidence percentages.

⚡ Real-Time Results: Models are optimized using Streamlit’s caching feature for faster predictions.

🎨 Elegant UI: Custom-styled interface with light background, buttons, and smooth layout.

🆕 NEW: Enhanced Professional UI Features
🎨 Modern Visual Design:

Beautiful gradient backgrounds and color schemes
Interactive hover effects and smooth animations
Mobile-responsive design for all screen sizes
Professional typography with Inter font family
📊 Advanced Interface Elements:

Tabbed navigation for better content organization
Real-time statistics dashboard with model metrics
Interactive prediction cards with confidence bars
Color-coded confidence levels (high/medium/low)
🚀 Improved User Experience:

Drag-and-drop file upload with visual feedback
Image metadata display (dimensions, size, format)
Processing time tracking and performance metrics
Interactive classify button with loading states
✨ Professional Styling:

Custom CSS with modern design patterns
Smooth transitions and micro-interactions
Enhanced sidebar with model information cards
Responsive layout that works on mobile devices
🧰 Technologies Used Component Technology Framework Streamlit Deep Learning Library TensorFlow / Keras Programming Language Python Models MobileNetV2, InceptionResNetV2, EfficientNetB0 Visualization Streamlit Components (Custom CSS) Image Handling Pillow (PIL) Environment Virtual Environment (venv) 🧩 How It Works

The user uploads an image.

The app loads the selected pretrained model (from TensorFlow).

The image is preprocessed to the correct size for that model:

MobileNetV2 → 224×224

EfficientNetB0 → 224×224

InceptionResNetV2 → 299×299

The model predicts the top 3 possible classes.

The results are displayed with confidence scores in real time.

🧠 Installation Guide

2️⃣ Create and Activate Virtual Environment python -m venv venv .\venv\Scripts\activate # On Windows

OR
source venv/bin/activate # On Mac/Linux

3️⃣ Install Dependencies pip install --upgrade pip pip install tensorflow streamlit pillow numpy

💡 If you have no GPU or want a lightweight version:

pip install tensorflow-cpu

4️⃣ Test the Installation (Optional) python test_app.py

This will verify all components are working correctly.

5️⃣ Run the Application streamlit run app.py

Then open your browser at 👉 http://localhost:8501

🎉 You'll now see the enhanced professional UI with modern styling!

🖼️ Usage Instructions

Select Model → Choose from the sidebar (MobileNetV2, InceptionResNetV2, or EfficientNetB0).

Upload Image → Upload any .jpg, .jpeg, or .png.

View Predictions → The app displays the top-3 predictions with confidence scores.

Compare Models → Try the same image with all three models to compare their outputs.

🎨 UI/UX Improvements
The application now features a completely redesigned professional interface with:

Visual Enhancements
Modern gradient backgrounds with smooth color transitions
Interactive elements with hover effects and animations
Professional typography using Inter font family
Mobile-responsive design that works on all devices
User Experience
Tabbed interface for better content organization
Real-time metrics showing model accuracy and parameters
Enhanced file upload with drag-and-drop functionality
Processing time tracking for performance transparency
Prediction Display
Confidence bars with color-coded levels
Ranking badges (🥇🥈🥉) for top predictions
Smooth animations for result presentation
Professional styling with modern card layouts
Technical Features
Custom CSS with modern design patterns
Responsive layout for mobile compatibility
Performance optimizations for faster loading
Accessibility improvements for better usability
The new interface provides a production-ready, professional experience that rivals commercial AI applications while maintaining the powerful functionality of the original deep learning models.
