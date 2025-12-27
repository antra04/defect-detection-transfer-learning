# evaluate_model.py
# Comprehensive model evaluation

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow import keras
import os

def load_test_data():
    """Load the actual test set used during training."""
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.utils import to_categorical
    
    print("Loading test data...")
    
    data_path = 'data/processed_simple/metal_nut'
    images = []
    labels = []
    filenames = []
    
    for split in ['train', 'test']:
        split_path = os.path.join(data_path, split)
        for category in os.listdir(split_path):
            category_path = os.path.join(split_path, category)
            if not os.path.isdir(category_path):
                continue
            
            label = 0 if category == 'good' else 1
            
            for file in os.listdir(category_path):
                if file.endswith('.npy'):
                    img = np.load(os.path.join(category_path, file))
                    images.append(img)
                    labels.append(label)
                    filenames.append(f"{split}/{category}/{file}")
    
    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    
    # Same split as training
    X_temp, X_test, y_temp, y_test, _, files_test = train_test_split(
        X, y, filenames, test_size=0.1, random_state=42, stratify=y
    )
    
    print(f"Test set: {len(X_test)} images")
    print(f"  GOOD: {np.sum(y_test==0)}")
    print(f"  DEFECT: {np.sum(y_test==1)}")
    
    return X_test, y_test, files_test

def evaluate_model():
    """Comprehensive model evaluation."""
    
    print("="*70)
    print("MODEL EVALUATION - COMPREHENSIVE ANALYSIS")
    print("="*70)
    
    # Load model
    print("\nLoading model...")
    model = keras.models.load_model('models/final_model.keras')
    
    # Load test data
    X_test, y_test, files_test = load_test_data()
    
    # Predict
    print("\nGenerating predictions...")
    predictions = model.predict(X_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    
    # Overall accuracy
    accuracy = np.mean(y_pred == y_test)
    
    print("\n" + "="*70)
    print("OVERALL RESULTS")
    print("="*70)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"Correct: {np.sum(y_pred == y_test)}/{len(y_test)}")
    
    # Per-class accuracy
    print("\n" + "="*70)
    print("PER-CLASS PERFORMANCE")
    print("="*70)
    
    good_mask = y_test == 0
    defect_mask = y_test == 1
    
    good_acc = np.mean(y_pred[good_mask] == y_test[good_mask])
    defect_acc = np.mean(y_pred[defect_mask] == y_test[defect_mask])
    
    print(f"\nGOOD samples:")
    print(f"  Total: {np.sum(good_mask)}")
    print(f"  Correct: {np.sum(y_pred[good_mask] == 0)}")
    print(f"  Accuracy: {good_acc*100:.1f}%")
    
    print(f"\nDEFECT samples:")
    print(f"  Total: {np.sum(defect_mask)}")
    print(f"  Correct: {np.sum(y_pred[defect_mask] == 1)}")
    print(f"  Accuracy: {defect_acc*100:.1f}%")
    
    # Classification report
    print("\n" + "="*70)
    print("DETAILED METRICS")
    print("="*70)
    print("\n" + classification_report(
        y_test, y_pred, 
        target_names=['GOOD', 'DEFECT'],
        digits=3
    ))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n" + "="*70)
    print("CONFUSION MATRIX")
    print("="*70)
    print("\n          Predicted")
    print("          GOOD  DEFECT")
    print(f"Actual GOOD    {cm[0,0]:3d}    {cm[0,1]:3d}")
    print(f"       DEFECT  {cm[1,0]:3d}    {cm[1,1]:3d}")
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['GOOD', 'DEFECT'],
                yticklabels=['GOOD', 'DEFECT'],
                cbar_kws={'label': 'Count'})
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=150)
    print("\n✅ Confusion matrix saved: outputs/confusion_matrix.png")
    plt.show()
    
    # Confidence analysis
    print("\n" + "="*70)
    print("CONFIDENCE ANALYSIS")
    print("="*70)
    
    confidences = np.max(predictions, axis=1)
    
    print(f"\nAverage confidence: {np.mean(confidences)*100:.1f}%")
    print(f"Min confidence: {np.min(confidences)*100:.1f}%")
    print(f"Max confidence: {np.max(confidences)*100:.1f}%")
    
    # Correct vs incorrect confidence
    correct_mask = y_pred == y_test
    
    print(f"\nCorrect predictions:")
    print(f"  Average confidence: {np.mean(confidences[correct_mask])*100:.1f}%")
    
    print(f"\nIncorrect predictions:")
    print(f"  Average confidence: {np.mean(confidences[~correct_mask])*100:.1f}%")
    
    # Show most confident mistakes
    mistakes = np.where(~correct_mask)[0]
    if len(mistakes) > 0:
        mistake_confidences = confidences[mistakes]
        top_mistakes = mistakes[np.argsort(mistake_confidences)[-5:]][::-1]
        
        print("\n" + "="*70)
        print("TOP 5 MOST CONFIDENT MISTAKES")
        print("="*70)
        
        for i, idx in enumerate(top_mistakes, 1):
            true_label = 'GOOD' if y_test[idx] == 0 else 'DEFECT'
            pred_label = 'GOOD' if y_pred[idx] == 0 else 'DEFECT'
            conf = confidences[idx]
            
            print(f"\n{i}. {files_test[idx]}")
            print(f"   True: {true_label}, Predicted: {pred_label}")
            print(f"   Confidence: {conf*100:.1f}%")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    evaluate_model()
