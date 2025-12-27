# src/inference.py
# Inference system for defect detection

import numpy as np
import cv2
from tensorflow import keras
import matplotlib.pyplot as plt
import os

class DefectDetector:
    """
    Inference system for defect detection.
    Load trained model and predict on new images.
    """
    
    def __init__(self, model_path='models/final_model.keras'):
        """
        Load trained model.
        
        Parameters:
        -----------
        model_path : str
            Path to saved model
        """
        self.model = keras.models.load_model(model_path)
        self.class_names = ['GOOD', 'DEFECT']
        
        print(f"✅ Model loaded from: {model_path}")
        print(f"   Model size: {os.path.getsize(model_path) / (1024**2):.1f} MB")
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for prediction (same as training).
        
        Parameters:
        -----------
        image_path : str
            Path to image file
            
        Returns:
        --------
        numpy.ndarray : Preprocessed image (224x224x3, normalized)
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Resize to 224x224
        resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        return normalized
    
    def predict(self, image_path):
        """
        Predict defect for a single image.
        
        Parameters:
        -----------
        image_path : str
            Path to image
            
        Returns:
        --------
        dict : Prediction results
            - class: 'GOOD' or 'DEFECT'
            - class_index: 0 (GOOD) or 1 (DEFECT)
            - confidence: probability of predicted class
            - probabilities: dict with prob for each class
        """
        # Preprocess
        img = self.preprocess_image(image_path)
        
        # Add batch dimension (model expects batch)
        img_batch = np.expand_dims(img, axis=0)
        
        # Predict
        predictions = self.model.predict(img_batch, verbose=0)
        
        # Get predicted class and confidence
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        
        result = {
            'class': self.class_names[class_idx],
            'class_index': int(class_idx),
            'confidence': float(confidence),
            'probabilities': {
                'GOOD': float(predictions[0][0]),
                'DEFECT': float(predictions[0][1])
            }
        }
        
        return result
    
    def predict_batch(self, image_paths):
        """
        Predict multiple images at once.
        
        Parameters:
        -----------
        image_paths : list
            List of image paths
            
        Returns:
        --------
        list : List of prediction results
        """
        results = []
        for path in image_paths:
            try:
                result = self.predict(path)
                result['image_path'] = path
                results.append(result)
            except Exception as e:
                print(f"❌ Error processing {path}: {e}")
        
        return results
    
    def visualize_prediction(self, image_path, save_path=None):
        """
        Visualize prediction with confidence bars.
        
        Parameters:
        -----------
        image_path : str
            Path to image
        save_path : str, optional
            Where to save visualization
        """
        # Get prediction
        result = self.predict(image_path)
        
        # Load original image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Show image with prediction
        ax1.imshow(img_rgb)
        ax1.axis('off')
        
        # Color based on prediction
        color = 'green' if result['class'] == 'GOOD' else 'red'
        ax1.set_title(
            f"Prediction: {result['class']}\nConfidence: {result['confidence']*100:.1f}%",
            fontsize=14, fontweight='bold', color=color
        )
        
        # Show probability bars
        classes = list(result['probabilities'].keys())
        probs = list(result['probabilities'].values())
        colors = ['green' if c == 'GOOD' else 'red' for c in classes]
        
        bars = ax2.barh(classes, probs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax2.set_xlim([0, 1])
        ax2.set_xlabel('Probability', fontsize=12)
        ax2.set_title('Class Probabilities', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # Add percentage labels on bars
        for i, (cls, prob) in enumerate(zip(classes, probs)):
            ax2.text(prob + 0.02, i, f'{prob*100:.1f}%', 
                    va='center', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Visualization saved to: {save_path}")
        
        plt.show()
        
        return result

# ============================================================
# TEST INFERENCE
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING DEFECT DETECTOR")
    print("="*70)
    
    # Initialize detector
    detector = DefectDetector('models/final_model.keras')
    
    # Test on sample images
    print("\n📋 Testing predictions on sample images:")
    print("-" * 70)
    
    test_images = [
        ('data/raw/metal_nut/test/good/000.png', 'GOOD'),
        ('data/raw/metal_nut/test/bent/000.png', 'BENT DEFECT'),
        ('data/raw/metal_nut/test/scratch/000.png', 'SCRATCH DEFECT'),
        ('data/raw/metal_nut/test/color/000.png', 'COLOR DEFECT')
    ]
    
    correct = 0
    for img_path, true_label in test_images:
        result = detector.predict(img_path)
        is_correct = (result['class'] == 'GOOD' and 'GOOD' in true_label) or \
                     (result['class'] == 'DEFECT' and 'DEFECT' in true_label)
        
        status = "✅" if is_correct else "❌"
        if is_correct:
            correct += 1
        
        print(f"\n{status} Image: {img_path}")
        print(f"   True Label:  {true_label}")
        print(f"   Prediction:  {result['class']} ({result['confidence']*100:.1f}% confident)")
        print(f"   Probabilities: GOOD={result['probabilities']['GOOD']*100:.1f}%, "
              f"DEFECT={result['probabilities']['DEFECT']*100:.1f}%")
    
    print("\n" + "="*70)
    print(f"Accuracy: {correct}/{len(test_images)} = {correct/len(test_images)*100:.0f}%")
    print("="*70)
    
    # Visualize one prediction
    print("\n📊 Creating visualization for bent defect...")
    detector.visualize_prediction(
        'data/raw/metal_nut/test/bent/000.png',
        'outputs/prediction_example.png'
    )
    
    print("\n✅ Inference system test complete!")
