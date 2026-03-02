#!/usr/bin/env python3
"""
Simple test script to verify the app components work correctly
"""

import sys
import importlib.util

def test_imports():
    """Test if all required modules can be imported"""
    try:
        import streamlit as st
        import numpy as np
        import tensorflow as tf
        from PIL import Image
        import time
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_tensorflow_models():
    """Test if TensorFlow models can be loaded"""
    try:
        import tensorflow as tf
        
        # Test model loading (without weights to save time)
        models = [
            ("MobileNetV2", tf.keras.applications.MobileNetV2),
            ("InceptionResNetV2", tf.keras.applications.InceptionResNetV2),
            ("EfficientNetB0", tf.keras.applications.EfficientNetB0)
        ]
        
        for name, model_class in models:
            try:
                # Just test if the model class exists and can be instantiated
                model_class(weights=None, include_top=False, input_shape=(224, 224, 3))
                print(f"✅ {name} model class accessible")
            except Exception as e:
                print(f"❌ {name} model error: {e}")
                return False
        
        print("✅ All TensorFlow models accessible!")
        return True
    except Exception as e:
        print(f"❌ TensorFlow models test failed: {e}")
        return False

def test_app_structure():
    """Test if the main app file has correct structure"""
    try:
        # Read the app.py file and check for main function
        with open("app.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Check if main function exists
        if "def main():" in content:
            print("✅ Main function found in app.py")
        else:
            print("❌ Main function not found in app.py")
            return False
        
        # Check if the main execution block exists
        if 'if __name__ == "__main__":' in content:
            print("✅ Main execution block found")
        else:
            print("❌ Main execution block not found")
            return False
            
        print("✅ App structure looks good!")
        return True
    except Exception as e:
        print(f"❌ App structure test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Image Classification App Components...\n")
    
    tests = [
        ("Import Test", test_imports),
        ("TensorFlow Models Test", test_tensorflow_models),
        ("App Structure Test", test_app_structure)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        result = test_func()
        results.append(result)
    
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 All tests passed! ({passed}/{total})")
        print("✅ Your app is ready to run!")
        print("\n💡 To start the app, run: streamlit run app.py")
    else:
        print(f"⚠️  {passed}/{total} tests passed")
        print("❌ Please fix the issues above before running the app")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)