"""
app.py - FIXED VERSION
Streamlit Dashboard with CORRECTED Interpretation Logic
"""

import streamlit as st
import cv2
import numpy as np
from tensorflow import keras
from PIL import Image
import os

# Page config
st.set_page_config(
    page_title="Defect Detector - MobileNetV2",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_model():
    return keras.models.load_model('models/mobilenet_model.keras')

def preprocess_image(image):
    """Preprocess uploaded image"""
    img = np.array(image)
    
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    
    return img

def predict(model, image):
    """Make prediction with detailed output"""
    try:
        img_array = np.expand_dims(image, axis=0)
        predictions = model.predict(img_array, verbose=0)
        
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        class_name = "GOOD" if class_idx == 0 else "DEFECT"
        
        good_prob = float(predictions[0][0])
        defect_prob = float(predictions[0][1])
        
        return {
            'class': class_name,
            'confidence': float(confidence),
            'good_prob': good_prob,
            'defect_prob': defect_prob
        }
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# Custom CSS
st.markdown("""
<style>
.big-prediction {
    font-size: 48px;
    font-weight: bold;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    margin: 20px 0;
}
.good-box {
    background-color: #d4edda;
    color: #155724;
    border: 3px solid #28a745;
}
.defect-box {
    background-color: #f8d7da;
    color: #721c24;
    border: 3px solid #dc3545;
}
.warning-box {
    background-color: #fff3cd;
    color: #856404;
    border: 3px solid #ffc107;
}
.metric-card {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #007bff;
}
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.title("🔍 Metal Nut Defect Detector")
    st.markdown("**Powered by MobileNetV2 Transfer Learning**")
    
    # Model info
    with st.expander("📊 Model Performance"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Accuracy", "88.2%", "↑ from 73.5%")
        with col2:
            st.metric("GOOD Detection", "94.6%", "37/39 correct")
        with col3:
            st.metric("DEFECT Detection", "71.4%", "10/14 caught")
        
        st.markdown("""
        **Note:** Model achieves 71.4% defect recall, meaning it catches ~7 out of 10 defects.
        Some minor defects may be missed.
        """)
    
    st.markdown("---")
    
    # Load model
    with st.spinner("Loading MobileNetV2 model..."):
        model = load_model()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📤 Upload Metal Nut Image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a 224x224 image of a metal nut"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Input Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Analysis Results")
            
            with st.spinner("Analyzing with MobileNetV2..."):
                processed_img = preprocess_image(image)
                result = predict(model, processed_img)
            
            if result:
                # Main prediction box
                if result['class'] == "GOOD":
                    if result['confidence'] > 0.85:
                        box_class = "good-box"
                        icon = "✅"
                        message = "HIGH CONFIDENCE"
                    else:
                        box_class = "warning-box"
                        icon = "⚠️"
                        message = "LOW CONFIDENCE - REVIEW RECOMMENDED"
                else:
                    box_class = "defect-box"
                    icon = "❌"
                    message = "DEFECT DETECTED"
                
                st.markdown(f"""
                <div class="big-prediction {box_class}">
                    {icon} {result['class']}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"**Status:** {message}")
                
                # Confidence meter
                st.markdown("### Confidence Score")
                confidence_pct = result['confidence'] * 100
                
                if confidence_pct > 85:
                    conf_color = "green"
                elif confidence_pct > 70:
                    conf_color = "orange"
                else:
                    conf_color = "red"
                
                st.markdown(f"<h2 style='color: {conf_color};'>{confidence_pct:.1f}%</h2>", 
                           unsafe_allow_html=True)
                
                st.progress(result['confidence'])
                
                # Probability breakdown
                st.markdown("### Class Probabilities")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("**GOOD**")
                    st.progress(result['good_prob'])
                    st.markdown(f"`{result['good_prob']*100:.1f}%`")
                
                with col_b:
                    st.markdown("**DEFECT**")
                    st.progress(result['defect_prob'])
                    st.markdown(f"`{result['defect_prob']*100:.1f}%`")
                
                # FIXED INTERPRETATION LOGIC
                st.markdown("---")
                st.markdown("### 💡 Interpretation")
                
                # Case 1: GOOD prediction with high defect probability (borderline)
                if result['class'] == "GOOD" and result['defect_prob'] > 0.15:
                    st.warning(f"""
                    **⚠️ BORDERLINE GOOD PART**
                    
                    Model predicts GOOD but detects {result['defect_prob']*100:.1f}% chance of defect.
                    
                    **Possible reasons:**
                    - Minor surface imperfection within tolerance
                    - Lighting/angle variations
                    - Borderline acceptable quality
                    
                    **Recommendation:** Manual inspection recommended for quality assurance.
                    """)
                
                # Case 2: GOOD prediction with high confidence
                elif result['class'] == "GOOD" and result['confidence'] > 0.85:
                    st.success("""
                    **✅ HIGH CONFIDENCE GOOD**
                    
                    Part appears to be in excellent condition with no visible defects.
                    
                    **Recommendation:** Part passes quality inspection.
                    """)
                
                # Case 3: GOOD prediction with low confidence
                elif result['class'] == "GOOD" and result['confidence'] <= 0.85:
                    st.info(f"""
                    **ℹ️ ACCEPTABLE QUALITY**
                    
                    Part classified as GOOD with {result['confidence']*100:.1f}% confidence.
                    
                    **Recommendation:** Part passes but consider spot-check inspection.
                    """)
                
                # Case 4: DEFECT prediction with low confidence
                elif result['class'] == "DEFECT" and result['confidence'] < 0.75:
                    st.warning(f"""
                    **⚠️ POSSIBLE DEFECT DETECTED**
                    
                    Model detects potential defect with {result['confidence']*100:.1f}% confidence.
                    
                    **Recommendation:** Manual inspection required to confirm defect type.
                    """)
                
                # Case 5: DEFECT prediction with high confidence
                else:  # DEFECT with high confidence
                    st.error(f"""
                    **❌ DEFECT DETECTED**
                    
                    Critical defect identified with {result['confidence']*100:.1f}% confidence.
                    
                    **Recommendation:** Part should be rejected or sent for rework.
                    """)
        
        # Detailed metrics
        with st.expander("📈 Detailed Metrics"):
            st.json({
                "Predicted Class": result['class'],
                "Confidence": f"{result['confidence']*100:.2f}%",
                "GOOD Probability": f"{result['good_prob']*100:.2f}%",
                "DEFECT Probability": f"{result['defect_prob']*100:.2f}%",
                "Model": "MobileNetV2",
                "Input Size": "224x224x3",
                "Inference Time": "~50ms"
            })
    
    else:
        # Instructions
        st.info("👆 Upload an image to get started")
        
        st.markdown("---")
        st.markdown("### 📁 Sample Test Images")
        
        st.markdown("""
        Try these sample images from the test set:
        - **GOOD samples:** `data/raw/metal_nut/test/good/`
        - **DEFECT samples:** 
          - Bent: `data/raw/metal_nut/test/bent/`
          - Scratch: `data/raw/metal_nut/test/scratch/`
          - Color: `data/raw/metal_nut/test/color/`
          - Flip: `data/raw/metal_nut/test/flip/`
        """)
        
        # Quick stats
        st.markdown("---")
        st.markdown("### 📊 Model Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Training Details:**
            - Architecture: MobileNetV2 (pretrained)
            - Dataset: MVTec Metal Nut (335 images)
            - Training: 241 images
            - Validation: 43 images
            - Test: 51 images
            """)
        
        with col2:
            st.markdown("""
            **Performance:**
            - Test Accuracy: 88.2%
            - GOOD Precision: 89.7%
            - DEFECT Precision: 83.3%
            - F1-Score: 87.9%
            """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
    <b>Transfer Learning Project</b> | MobileNetV2 + Custom Classifier<br>
    Demonstrates: Class Imbalance Handling, Transfer Learning, Industrial ML<br>
    <i>Antra Tiwari</i>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
