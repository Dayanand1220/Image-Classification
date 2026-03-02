#!/usr/bin/env python3
"""
Demo script to showcase the enhanced UI features
"""

import streamlit as st

def main():
    st.set_page_config(
        page_title="UI Features Demo",
        page_icon="🎨",
        layout="wide"
    )
    
    st.title("🎨 Enhanced UI Features Demo")
    st.markdown("This demo showcases the new professional UI improvements")
    
    # Show the CSS features
    st.markdown("""
    <style>
        .demo-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            margin: 1rem 0;
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
            transition: all 0.3s ease;
        }
        .demo-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(102, 126, 234, 0.4);
        }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='demo-card'>
            <h3>🎯 Modern Design</h3>
            <p>Beautiful gradients, smooth animations, and professional styling</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='demo-card'>
            <h3>📱 Mobile Responsive</h3>
            <p>Works perfectly on all screen sizes and devices</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='demo-card'>
            <h3>⚡ Interactive Elements</h3>
            <p>Hover effects, smooth transitions, and engaging UI</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='demo-card'>
            <h3>🎨 Professional Look</h3>
            <p>Production-ready interface that rivals commercial apps</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("🚀 Run `streamlit run app.py` to see the full enhanced image classification app!")

if __name__ == "__main__":
    main()