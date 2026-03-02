import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import time

st.set_page_config(
    page_title="AI Image Classifier | Deep Learning Models",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/image-classifier',
        'Report a bug': 'https://github.com/your-repo/image-classifier/issues',
        'About': "# AI Image Classifier\nPowered by TensorFlow and Streamlit"
    }
)

# Enhanced CSS for professional and modern UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global App Styling */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #111827 100%); /* Dark charcoal/navy gradient */
        min-height: 100vh;
        color: #e2e8f0; /* Light gray for default text */
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 3.5rem;
        padding-bottom: 3.5rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        background: rgba(15, 23, 42, 0.9); /* Darker, slightly transparent */
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5); /* Stronger, softer shadow */
        backdrop-filter: blur(25px); /* Enhanced blur for futuristic feel */
        margin: 2rem auto;
        max-width: 1300px;
        border: 1px solid rgba(99, 102, 241, 0.3); /* Violet accent border */
    }
    
    /* Sidebar styling */
    .css-1d391kg { /* This targets the Streamlit sidebar directly */
        background: linear-gradient(180deg, #111827 0%, #1e293b 100%); /* Deep navy to slate gradient */
        border-radius: 0 25px 25px 0;
        border-right: 2px solid rgba(34, 211, 238, 0.4); /* Aqua cyan accent border */
        box-shadow: 8px 0 20px rgba(0,0,0,0.3); /* Deeper shadow */
    }
    
    .sidebar .sidebar-content {
        background: transparent;
        padding: 3.5rem 2rem;
    }
    
    /* Title styling */
    h1 {
        color: #ffffff !important; /* White for main title */
        text-align: center;
        font-weight: 900 !important;
        font-size: 3rem !important; /* Adjusted font size */
        margin-bottom: 0.8rem;
        text-shadow: none !important;
        z-index: 10;
        position: relative;
    }
    
    /* Subtitle styling (H2 equivalent for consistency) */
    .subtitle {
        text-align: center;
        color: #94a3b8 !important; /* Silver-gray accent */
        font-size: 1.8rem; /* Adjusted font size */
        font-weight: 600 !important; /* Semi-bold */
        margin-bottom: 3.5rem;
        text-shadow: none;
        opacity: 0.95;
    }
    
    /* Section Titles (H3) */
    h3 {
        color: #6366f1; /* Violet for headers */
        font-weight: 700;
        margin-top: 2.5rem;
        margin-bottom: 1.2rem;
        text-shadow: none;
        text-align: left; /* Left-aligned */
        font-size: 1.3rem; /* Adjusted font size */
    }
    
    /* Body Text */
    p {
        color: #e2e8f0; /* Light gray for readability */
        font-weight: 400;
        line-height: 1.7;
        font-size: 1rem; /* Regular font size */
        text-align: left; /* Left-aligned */
    }
    
    /* Radio button styling */
    .stRadio > label {
        background: rgba(30, 41, 59, 0.7); /* Darker slate background */
        padding: 1.4rem;
        border-radius: 18px;
        margin: 0.8rem 0;
        border: 2px solid rgba(99, 102, 241, 0.3); /* Subtle violet border */
        transition: all 0.3s ease-in-out;
        cursor: pointer;
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.3); /* Soft shadow */
    }
    
    .stRadio > label:hover {
        background: rgba(47, 57, 76, 0.9); /* Slightly lighter on hover */
        border: 2px solid #22d3ee; /* Aqua cyan accent on hover */
        transform: translateY(-5px);
        box-shadow: 0 0 20px rgba(34, 211, 238, 0.6), 0 12px 25px rgba(0, 0, 0, 0.4); /* Glowing effect */
    }
    
    /* All radio button options - light text */
    .stRadio div[role="radiogroup"] label {
        color: #e2e8f0 !important; /* Light gray text */
    }
    
    .stRadio div[role="radiogroup"] label p {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
    }
    
    /* Selected radio button */
    .stRadio div[role="radiogroup"] label[aria-checked="true"] {
        background: rgba(99, 102, 241, 0.2) !important; /* Violet background for selected */
        border: 2px solid #22d3ee !important; /* Aqua cyan accent for selected */
        box-shadow: 0 0 15px rgba(34, 211, 238, 0.5) !important; /* Aqua cyan glow */
    }
    
    .stRadio div[role="radiogroup"] label[aria-checked="true"] p {
        color: #ffffff !important; /* Pure white for selected text */
        font-weight: 600 !important;
    }
    
    /* "Select Model" label should be light */
    .stRadio > div > label[data-testid="stWidgetLabel"] {
        color: #22d3ee !important; /* Aqua cyan for label */
        font-weight: 700 !important;
        font-size: 1.3rem !important;
    }
    
    .stRadio > div > label[data-testid="stWidgetLabel"] p {
        color: #22d3ee !important;
        font-weight: 700 !important;
        font-size: 1.3rem !important;
    }
    
    /* All model option text - light text */
    .stRadio > div > label > div {
        color: #e2e8f0 !important;
        font-weight: 500;
        font-size: 1.1rem;
    }
    
    /* File uploader styling */
    .stFileUploader > div > div {
        background: rgba(15, 23, 42, 0.8); /* Dark charcoal background */
        border: 3px dashed rgba(34, 211, 238, 0.5); /* Aqua cyan dashed border */
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.4);
    }
    
    .stFileUploader > div > div:hover {
        border-color: rgba(34, 211, 238, 0.8);
        transform: scale(1.01);
        background: rgba(22, 33, 53, 0.9); /* Slightly darker on hover */
        box-shadow: 0 0 25px rgba(34, 211, 238, 0.7), 0 15px 30px rgba(0, 0, 0, 0.5); /* Glowing effect */
    }
    
    .stFileUploader label {
        color: #22d3ee !important; /* Aqua cyan */
        font-weight: 700;
        font-size: 1.3rem;
    }
    
    /* Prediction box styling */
    .prediction-box {
        background: rgba(15, 23, 42, 0.85); /* Dark charcoal background */
        color: #e2e8f0;
        padding: 2rem;
        margin: 1.5rem 0;
        border-radius: 20px;
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.4); /* Stronger shadow */
        border: 1px solid rgba(99, 102, 241, 0.4); /* Violet border */
        font-size: 1.2rem;
        font-weight: 500;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease-in-out;
    }
    
    .prediction-box:hover {
        transform: translateY(-8px);
        box-shadow: 0 0 30px rgba(99, 102, 241, 0.7), 0 25px 50px rgba(0, 0, 0, 0.6); /* Enhanced glowing effect */
        background: rgba(22, 33, 53, 0.95); /* Darker on hover */
    }
    
    .prediction-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 6px; /* Thicker accent line */
        background: linear-gradient(90deg, #22d3ee, #6366f1); /* Cyan-violet gradient */
        background-size: 400% 400%;
        animation: gradient 5s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .prediction-rank {
        display: inline-block;
        background: rgba(34, 211, 238, 0.2); /* Aqua cyan subtle background */
        padding: 0.5rem 1.2rem;
        border-radius: 30px;
        font-size: 1rem;
        font-weight: 600;
        margin-right: 1.5rem;
        color: #22d3ee; /* Aqua cyan text */
        border: 1px solid rgba(34, 211, 238, 0.4);
    }
    
    /* Button styling */
    .stButton > button {
        background: #000000; /* Solid black background */
        color: #ffffff; /* White text */
        border: none;
        border-radius: 35px;
        padding: 1.4rem 3.5rem;
        font-size: 1.3rem;
        font-weight: 800;
        cursor: pointer;
        transition: all 0.4s ease-in-out;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5); /* Soft shadow */
        text-transform: uppercase;
        letter-spacing: 2px;
        position: relative;
        overflow: hidden;
        border: 2px solid #ffffff; /* White border */
        background-clip: padding-box, border-box;
        background-origin: padding-box, border-box;
        animation: none; /* Removed glow animation */
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.7);
        background: #333333; /* Darker gray on hover */
        animation: none;
        border-color: #94a3b8; /* Silver-gray border on hover */
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(0.99);
    }

    @keyframes glow {
        from { box-shadow: none; }
        to { box-shadow: none; }
    }
    
    /* Spinner styling */
    .stSpinner > div > div {
        border-top-color: #22d3ee !important; /* Aqua cyan spinner */
        border-right-color: #22d3ee !important;
    }
    
    /* Info box styling */
    .stInfo {
        background: rgba(30, 41, 59, 0.7) !important; /* Dark slate background */
        border: 2px solid rgba(99, 102, 241, 0.5) !important; /* Violet border */
        border-radius: 18px;
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4) !important;
        padding: 1.8rem !important;
        margin: 1.8rem 0 !important;
    }
    
    .stInfo > div {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }
    
    .stInfo [data-testid="stMarkdownContainer"] p {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
        font-size: 1.2rem !important;
        text-shadow: 0 0 5px rgba(99, 102, 241, 0.2) !important;
    }
    
    .stInfo div[data-testid="stMarkdownContainer"] {
        color: #e2e8f0 !important;
    }
    
    /* Image styling */
    .stImage > img {
        border-radius: 20px;
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease-in-out;
        border: 1px solid rgba(34, 211, 238, 0.3); /* Aqua cyan subtle border */
    }
    
    .stImage:hover > img {
        transform: scale(1.01);
        box-shadow: 0 0 20px rgba(34, 211, 238, 0.7), 0 25px 50px rgba(0, 0, 0, 0.4); /* Glowing effect */
    }
    
    /* Sidebar title */
    .sidebar .sidebar-content h1 {
        color: #94a3b8; /* Silver-gray */
        font-size: 2rem;
        margin-bottom: 1.8rem;
        text-align: center;
        font-weight: 800;
        text-shadow: 0 0 10px rgba(148, 163, 184, 0.3);
    }
    
    .sidebar .sidebar-content p {
        color: #94a3b8; /* Silver-gray */
        text-align: center;
        margin-bottom: 3rem;
        font-size: 1rem;
    }
    
    /* Model info cards */
    .model-info {
        background: rgba(30, 41, 59, 0.6); /* Dark slate background */
        padding: 1.5rem;
        border-radius: 18px;
        margin: 1.2rem 0;
        color: #e2e8f0;
        font-size: 1rem;
        line-height: 1.6;
        border: 1px solid rgba(34, 211, 238, 0.3); /* Aqua cyan border */
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.3);
    }

    .model-info strong {
        color: #6366f1; /* Violet for strong text */
    }
    
    /* Results section */
    .results-header {
        background: linear-gradient(135deg, #6366f1 0%, #22d3ee 100%); /* Violet to Aqua Cyan gradient */
        color: #ffffff;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 2.5rem 0;
        font-weight: 800;
        font-size: 1.6rem;
        box-shadow: 0 12px 30px rgba(99, 102, 241, 0.6); /* Violet shadow */
        border: 2px solid rgba(34, 211, 238, 0.5); /* Aqua cyan border */
        position: relative;
        overflow: hidden;
    }
    
    .results-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    /* Loading animation */
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.7; transform: scale(1.08); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    .loading {
        animation: pulse 2s infinite;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1.8rem;
            margin: 1rem;
        }
        
        h1 {
            font-size: 2.5rem !important; /* Adjusted font size */
        }
        
        .subtitle {
            font-size: 1.4rem;
        }
        
        .prediction-box {
            padding: 1.5rem;
            font-size: 1.1rem;
        }

        .stButton > button {
            padding: 1.2rem 2.5rem;
            font-size: 1.1rem;
        }
    }
    
    /* Smooth transitions for all interactive elements */
    * {
        transition: all 0.3s ease-in-out;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0f172a;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #22d3ee, #6366f1); /* Cyan-violet gradient */
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #6366f1, #22d3ee); /* Violet-cyan on hover */
    }
    
    /* Enhanced metric styling */
    .metric-container {
        background: rgba(30, 41, 59, 0.7); /* Dark slate background */
        padding: 1.5rem;
        border-radius: 18px;
        text-align: center;
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(34, 211, 238, 0.3); /* Aqua cyan border */
        color: #e2e8f0; /* Light gray text */
        margin-bottom: 1.5rem;
        display: flex; /* Use flexbox for internal alignment */
        flex-direction: column; /* Stack content vertically */
        justify-content: center; /* Center content vertically */
        align-items: center; /* Center horizontally */
        min-height: 120px; /* Ensure consistent height */
        gap: 0.5rem; /* Space between value and label */
    }
    
    .metric-container:hover {
        transform: translateY(-5px); /* More pronounced lift on hover */
        box-shadow: 0 0 25px rgba(34, 211, 238, 0.6), 0 15px 30px rgba(0, 0, 0, 0.5); /* Enhanced glowing effect */
        background: rgba(47, 57, 76, 0.95); /* Slightly darker on hover */
    }

    /* Streamlit metric value (e.g., "3") */
    [data-testid="stMetricValue"] {
        color: #6366f1 !important; /* Violet accent for metric values */
        font-size: 2.7rem !important; /* Slightly larger font size */
        font-weight: 800 !important;
        text-shadow: 0 0 10px rgba(99, 102, 241, 0.6);
        margin-bottom: 0.2rem; /* Reduced space between value and label, now handled by gap */
    }

    /* Streamlit metric label (e.g., "Models Available") */
    [data-testid="stMetricLabel"] {
        color: #22d3ee !important; /* Aqua cyan for labels */
        font-size: 1.25rem !important; /* Slightly larger font size */
        font-weight: 700 !important; /* Bolder font weight */
        text-transform: uppercase; /* Uppercase for labels */
        letter-spacing: 0.08em; /* Increased letter spacing */
    }

    /* Streamlit metric delta (e.g., "+1") */
    [data-testid="stMetricDelta"] {
        color: #10b981 !important; /* Emerald green for positive */
    }
    
    /* Tab styling - make text light with glow */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        color: #ffffff !important; /* White for tab headers */
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        text-shadow: 0 0 8px rgba(148, 163, 184, 0.3);
    }
    
    .stTabs [data-baseweb="tab-list"] button {
        color: #ffffff !important; /* White for tabs */
        background: rgba(30, 41, 59, 0.6); /* Dark slate background */
        border-radius: 12px 12px 0 0;
        margin-right: 0.8rem;
        transition: all 0.3s ease-in-out;
        border: 1px solid rgba(22, 33, 53, 0.4);
        box-shadow: 0 -5px 15px rgba(0, 0, 0, 0.2);
    }

    .stTabs [data-baseweb="tab-list"] button:hover {
        background: rgba(47, 57, 76, 0.8); /* Slightly darker on hover */
        transform: translateY(-3px);
        box-shadow: 0 -8px 20px rgba(34, 211, 238, 0.4), 0 -5px 15px rgba(0, 0, 0, 0.3);
    }

    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background: #1e1b4b !important; /* Deep navy for selected tab */
        border-bottom: 3px solid #6366f1 !important; /* Violet accent line */
        color: #ffffff !important; /* White for selected text */
        transform: translateY(-2px);
        box-shadow: 0 -5px 15px rgba(99, 102, 241, 0.5), 0 -8px 20px rgba(0, 0, 0, 0.4) !important;
    }

    /* General Markdown elements */
    [data-testid="stMarkdownContainer"] {
        color: #e2e8f0; /* Default markdown text color */
    }

    /* Adjust Streamlit specific elements for better padding/margin */
    .stHorizontalBlock {
        margin-bottom: 2rem;
    }

    .stVerticalBlock {
        margin-bottom: 1.5rem;
    }
    /* Updated prediction colors for better contrast and visual appeal */
    /* These colors are for the progress bars in the prediction boxes */
    .prediction-box .high-confidence {
        background-color: #10b981; /* Emerald green */
    }
    .prediction-box .medium-confidence {
        background-color: #fbbf24; /* Amber yellow */
    }
    .prediction-box .low-confidence {
        background-color: #ef4444; /* Strong red */
    }

    /* Welcome section styling */
    .welcome-section {
        text-align: center;
        padding: 3.5rem;
        background: linear-gradient(135deg, #1e1b4b 0%, #111827 100%); /* Deep navy gradient */
        border-radius: 20px;
        color: #e2e8f0; /* Light gray */
        margin: 2rem 0;
        border: 1px solid rgba(99, 102, 241, 0.4);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4), 0 0 20px rgba(99, 102, 241, 0.3);
    }

    .welcome-section h3 {
        color: #6366f1; /* Violet */
        font-size: 2rem;
        margin-bottom: 1.2rem;
        text-shadow: 0 0 10px rgba(99, 102, 241, 0.5);
    }

    .welcome-section p {
        color: #94a3b8; /* Silver-gray */
        font-size: 1.2rem;
        margin: 1rem 0;
    }

    .welcome-section small {
        color: #94a3b8;
        opacity: 0.9;
    }

    /* Footer styling */
    .app-footer {
        text-align: center;
        padding: 2.5rem;
        background: linear-gradient(135deg, #1e1b4b 0%, #111827 100%); /* Deep navy gradient */
        border-radius: 20px;
        color: #e2e8f0;
        margin-top: 2.5rem;
        border: 1px solid rgba(34, 211, 238, 0.4);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4), 0 0 20px rgba(34, 211, 238, 0.3);
    }

    .app-footer h4 {
        color: #22d3ee; /* Aqua cyan */
        font-size: 1.5rem;
        margin-bottom: 1rem;
        text-shadow: 0 0 8px rgba(34, 211, 238, 0.4);
    }

    .app-footer p {
        color: #6366f1; /* Violet */
        margin-bottom: 0.8rem;
    }

    .app-footer small {
        color: #6366f1;
        opacity: 0.8;
    }

    /* Adjust Streamlit columns for equal height and consistent spacing */
    div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stHorizontalBlock"]) {
        align-items: stretch;
        display: flex;
        flex-direction: column; /* Ensure vertical stacking for inner content */
    }

    /* Individual columns within the main classification area */
    div[data-testid="stHorizontalBlock"] > div[data-testid="stVerticalBlock"] {
        padding: 1.5rem; /* Consistent padding inside columns */
        background: rgba(15, 23, 42, 0.7); /* Slightly lighter background for columns */
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(99, 102, 241, 0.2); /* Subtle border */
    }

    /* Image column specific styling */
    div[data-testid="stHorizontalBlock"] > div:first-child {
        margin-right: 0.75rem; /* Space between image and results column */
    }

    /* Results column specific styling */
    div[data-testid="stHorizontalBlock"] > div:last-child {
        margin-left: 0.75rem; /* Space between image and results column */
    }

    /* Ensure images fill their column width and maintain aspect ratio */
    .stImage > img {
        width: 100%;
        height: auto;
        object-fit: contain; /* Prevent cropping and maintain aspect ratio */
        max-height: 400px; /* Limit height to prevent excessive scrolling */
    }

    /* Adjust prediction box spacing within results column */
    .prediction-box {
        margin: 0.8rem 0; /* Reduced vertical margin */
        padding: 1.5rem; /* Adjusted padding */
    }

    /* Center the button */
    div[data-testid="stVerticalBlock"] > div > div > button {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }

    /* Mobile responsiveness for columns */
    @media (max-width: 768px) {
        div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stHorizontalBlock"]) {
            flex-direction: column; /* Stack columns vertically */
            align-items: center; /* Center horizontally when stacked */
        }

        div[data-testid="stHorizontalBlock"] > div[data-testid="stVerticalBlock"] {
            width: 100% !important; /* Full width on small screens */
            margin-left: 0 !important;
            margin-right: 0 !important;
            margin-bottom: 1.5rem; /* Space between stacked columns */
            padding: 1rem; /* Adjusted padding for mobile */
        }

        .stImage > img {
            max-height: 300px; /* Further limit image height on mobile */
        }

        .prediction-box {
            padding: 1rem; /* Adjusted padding for mobile */
        }
    }
    
    /* Adjust overall app statistics section */
    .st-emotion-cache-nahz7x { /* Target for st.columns in main stat section */
        margin-bottom: 2rem; 
    }

    /* Ensure general markdown elements have appropriate spacing */
    [data-testid="stMarkdownContainer"] {
        margin-bottom: 1rem; 
    }

    /* Adjust for Streamlit tabs to have better spacing */
    .stTabs {
        margin-top: 2rem;
        margin-bottom: 2rem;
    }

</style>
""", unsafe_allow_html=True)

# Unified classification function for better code organization
def classify_with_model(container, image, model_name, model_class, preprocessing_module, input_size):
    """
    Unified function to classify images with any of the three models
    """
    try:
        # Load the model
        model = model_class(weights='imagenet')
        
        # Preprocess the image
        img = image.resize(input_size)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocessing_module.preprocess_input(img_array)
        
        # Make predictions
        predictions = model.predict(img_array, verbose=0)
        decoded_predictions = preprocessing_module.decode_predictions(predictions, top=3)[0]
        
        # Display results with enhanced styling
        confidence_colors = ['#e74c3c', '#f39c12', '#27ae60']  # Red, Orange, Green
        rank_emojis = ['#1', '#2', '#3']
        
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            confidence_percentage = score * 100
            
            # Determine confidence level for styling
            if confidence_percentage >= 70:
                confidence_level = "high"
                bar_color = "#10b981"
            elif confidence_percentage >= 40:
                confidence_level = "medium" 
                bar_color = "#fbbf24"
            else:
                confidence_level = "low"
                bar_color = "#ef4444"
            
            # Create prediction box with progress bar
            container.markdown(f"""
            <div class='prediction-box'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;'>
                    <div>
                        <span class='prediction-rank'>{rank_emojis[i]} #{i+1}</span>
                        <strong style='font-size: 1.2rem;'>{label.replace('_', ' ').title()}</strong>
                    </div>
                    <div style='font-size: 1.3rem; font-weight: 700;'>
                        {confidence_percentage:.1f}%
                    </div>
                </div>
                <div style='background: rgba(255,255,255,0.2); border-radius: 10px; height: 8px; overflow: hidden;'>
                    <div style='background: {bar_color}; height: 100%; width: {confidence_percentage}%; 
                               border-radius: 10px; transition: width 0.5s ease;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Add model performance info
        container.markdown("---")
        container.markdown(f"""
        <div style='text-align: center; padding: 1rem; background: rgba(99, 102, 241, 0.1); 
                    border-radius: 10px; margin-top: 1rem;'>
            <strong>Classification completed with {model_name}</strong><br>
            <small style='opacity: 0.8; color: #94a3b8;'>Processed with {input_size[0]}×{input_size[1]} input resolution</small>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        container.error(f"Error during classification: {str(e)}")
        container.info("Please try uploading a different image or refresh the page.")

# Main function to control the navigation
def main():
    # Sidebar with model selection
    st.sidebar.title("AI Models")
    st.sidebar.markdown("Choose a deep learning model to classify your images with state-of-the-art accuracy.")
    
    # Model information
    model_info = {
        "MobileNetV2 (ImageNet)": {
            "description": "Lightweight and efficient model optimized for mobile devices",
            "accuracy": "71.8% Top-1",
            "params": "3.4M"
        },
        "InceptionResNetV2 (ImageNet)": {
            "description": "High-accuracy model combining Inception and ResNet architectures",
            "accuracy": "80.3% Top-1", 
            "params": "55.8M"
        },
        "EfficientNetB0 (ImageNet)": {
            "description": "Balanced model optimizing accuracy and efficiency",
            "accuracy": "77.1% Top-1",
            "params": "5.3M"
        }
    }
    
    choice = st.sidebar.radio(
        "",
        list(model_info.keys()),
        format_func=lambda x: x.split(' ')[0],
        label_visibility="collapsed"
    )
 
    
    # Display model info in sidebar
    selected_info = model_info[choice]
    st.sidebar.markdown(f"""
    <div class='model-info'>
        <strong>Model Details</strong><br>
        Accuracy: {selected_info['accuracy']}<br>
        Parameters: {selected_info['params']}<br>
        {selected_info['description']}
    </div>
    """, unsafe_allow_html=True)
    
    # Main content area with enhanced header
    st.markdown("""
    <h1 style='color: white; font-weight: 800; font-size: 3.5rem; text-align: center; 
               text-shadow: 2px 2px 4px rgba(0,0,0,0.3); margin-bottom: 0.5rem;'>
        🤖 AI Powered Image Classification
    </h1>
    """, unsafe_allow_html=True)
    st.markdown(f"<div class='subtitle' style='color: white; font-weight: 600; margin-bottom: 2rem;'>Powered by {choice.split(' ')[0]} Neural Network</div>", unsafe_allow_html=True)
    
    # Add app statistics in a nice layout
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Models Available", "3", help="MobileNetV2, InceptionResNetV2, EfficientNetB0")
    with col2:
        st.metric("Categories", "1000+", help="ImageNet dataset categories")
    with col3:
        st.metric("Accuracy", f"{model_info[choice]['accuracy'].replace(' Top-1', '')}", help=f"Accuracy for {choice.split(' ')[0]}")
    with col4:
        st.metric("Parameters", f"{model_info[choice]['params']}", help="Model size and complexity")
    
    # Create tabs for better organization
    tab1, tab2 = st.tabs(["Upload & Classify", "About"])
    
    with tab1:
        st.markdown("### Upload Your Image")
        st.markdown("Supported formats: JPG, PNG, JPEG • Max size: 200MB")
        
        uploaded_file = st.file_uploader(
            "Drag and drop an image here or click to browse",
            type=["jpg", "png", "jpeg"],
            help="Upload an image to get AI-powered classification results"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            # Display image info
            st.markdown("---")
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("Dimensions", f"{image.size[0]} × {image.size[1]}")
            with col_info2:
                st.metric("Mode", image.mode)
            with col_info3:
                st.metric("Size", f"{len(uploaded_file.getvalue()) / 1024:.1f} KB")
            
            st.markdown("---")
            
            # Main classification area
            col1, col2 = st.columns([1, 1], gap="medium") # Added gap="medium"
            
            with col1:
                st.markdown("### Your Image")
                st.image(image, caption='', use_column_width=True)
            
            with col2:
                st.markdown("<div class='results-header'>AI Classification Results</div>", unsafe_allow_html=True)
                
                results_display_area = st.container() # Container for classification results

                # Centering the button using Streamlit's internal layout
                st.markdown("<div style='text-align: center; margin-top: 1.5rem;'>", unsafe_allow_html=True)
                if st.button("Classify Image", type="primary", use_container_width=False, key="classify_button"): # Changed to use_container_width=False
                    start_time = time.time()
                    
                    with st.spinner("🔄 Analyzing image with AI model..."):
                        if choice == "MobileNetV2 (ImageNet)":
                            classify_with_model(results_display_area, image, "MobileNetV2", tf.keras.applications.MobileNetV2, 
                                              tf.keras.applications.mobilenet_v2, (224, 224))
                        elif choice == "InceptionResNetV2 (ImageNet)":
                            classify_with_model(results_display_area, image, "InceptionResNetV2", tf.keras.applications.InceptionResNetV2,
                                              tf.keras.applications.inception_resnet_v2, (299, 299))
                        elif choice == "EfficientNetB0 (ImageNet)":
                            classify_with_model(results_display_area, image, "EfficientNetB0", tf.keras.applications.EfficientNetB0,
                                              tf.keras.applications.efficientnet, (224, 224))
                    
                    processing_time = time.time() - start_time
                    results_display_area.success(f"Classification completed in {processing_time:.2f} seconds!")
                else:
                    st.info("Click the button above to start classification")
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='welcome-section'>
                <h3>Ready to Start?</h3>
                <p>Upload an image above to see AI classification in action!</p>
                <p>Our models can identify thousands of objects, animals, and scenes.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### About This Application")
        st.markdown("""
        This application uses state-of-the-art deep learning models trained on ImageNet dataset 
        to classify images into 1000+ categories with high accuracy.
        
        **Features:**
        - Three different neural network architectures
        - Real-time image classification
        - Top-3 predictions with confidence scores
        - Professional and intuitive interface
        
        **Models Available:**
        """)
        
        for model_name, info in model_info.items():
            st.markdown(f"""
            **{model_name}**
            - {info['description']}
            - Accuracy: {info['accuracy']}
            - Parameters: {info['params']}
            """)
        
        st.markdown("""
        **Technology Stack:**
        - TensorFlow/Keras for deep learning
        - Streamlit for web interface
        - PIL for image processing
        - Pre-trained ImageNet weights
        """)
    
    # Add footer
    st.markdown("---")
    st.markdown("""
    <div class='app-footer'>
        <h4>Ready to Explore AI?</h4>
        <p>This application demonstrates the power of deep learning for image recognition. 
           Try different images and models to see how AI can understand and classify visual content!</p>
        <small>Built with ❤️ using TensorFlow, Streamlit, and modern web technologies</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
