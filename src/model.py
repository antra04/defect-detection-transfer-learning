# src/model.py
# SIMPLIFIED CNN Model for Defect Detection (FIXED for small datasets)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np

class DefectClassifier:
    """Lightweight CNN for binary defect classification with small datasets."""
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
        print(f"✅ DefectClassifier initialized (Lightweight Version)")
        print(f"   Input shape: {input_shape}")
        print(f"   Output classes: {num_classes}")
    
    def build_model(self):
        """
        Build SIMPLIFIED CNN architecture for small datasets.
        
        KEY CHANGES:
        - Fewer filters (16, 32, 64 instead of 32, 64, 128)
        - Smaller dense layer (64 instead of 128)
        - More aggressive dropout (0.6 instead of 0.5)
        - Global Average Pooling instead of Flatten
        
        Result: ~400K parameters instead of 12.9M!
        """
        
        model = models.Sequential([
            # ==================== BLOCK 1 ====================
            layers.Conv2D(16, (3, 3), activation='relu', padding='same',
                         input_shape=self.input_shape, name='conv1'),
            layers.MaxPooling2D((2, 2), name='pool1'),
            layers.BatchNormalization(name='bn1'),
            layers.Dropout(0.3, name='drop1'),  # Early dropout
            
            # ==================== BLOCK 2 ====================
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2'),
            layers.MaxPooling2D((2, 2), name='pool2'),
            layers.BatchNormalization(name='bn2'),
            layers.Dropout(0.4, name='drop2'),
            
            # ==================== BLOCK 3 ====================
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv3'),
            layers.MaxPooling2D((2, 2), name='pool3'),
            layers.BatchNormalization(name='bn3'),
            layers.Dropout(0.5, name='drop3'),
            
            # ==================== GLOBAL POOLING ====================
            # This reduces params massively! (replaces Flatten)
            layers.GlobalAveragePooling2D(name='global_pool'),
            
            # ==================== CLASSIFICATION HEAD ====================
            layers.Dense(64, activation='relu', name='dense1'),
            layers.Dropout(0.6, name='dropout'),  # Heavy dropout
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ])
        
        self.model = model
        print("\n✅ Lightweight model built successfully!")
        return model
    
    def compile_model(self, learning_rate=0.0001):  # Lower learning rate
        """Compile with lower learning rate for stability."""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"\n✅ Model compiled with:")
        print(f"   Optimizer: Adam (lr={learning_rate})")
        print(f"   Loss: Categorical Crossentropy")
        print(f"   Metrics: Accuracy")
    
    def get_summary(self):
        """Print model summary."""
        print("\n" + "="*70)
        print("MODEL ARCHITECTURE SUMMARY (LIGHTWEIGHT)")
        print("="*70)
        self.model.summary()
        
        total_params = self.model.count_params()
        print("\n" + "="*70)
        print(f"Total Parameters: {total_params:,}")
        print(f"Reduction: {(1 - total_params/12939586)*100:.1f}% smaller than original!")
        print("="*70)
    
    def save_model(self, filepath='models/defect_classifier.keras'):
        """Save trained model."""
        self.model.save(filepath)
        print(f"\n✅ Model saved to: {filepath}")
    
    def load_model(self, filepath='models/defect_classifier.keras'):
        """Load pre-trained model."""
        self.model = keras.models.load_model(filepath)
        print(f"\n✅ Model loaded from: {filepath}")

if __name__ == "__main__":
    print("\n🔬 TESTING LIGHTWEIGHT CNN MODEL\n")
    
    classifier = DefectClassifier(input_shape=(224, 224, 3), num_classes=2)
    classifier.build_model()
    classifier.compile_model(learning_rate=0.0001)
    classifier.get_summary()
    
    # Test with random data
    test_batch = np.random.rand(4, 224, 224, 3).astype(np.float32)
    predictions = classifier.model.predict(test_batch, verbose=0)
    
    print(f"\nTest prediction: {predictions[0]}")
    print(f"  Class 0 (Good): {predictions[0][0]*100:.2f}%")
    print(f"  Class 1 (Defect): {predictions[0][1]*100:.2f}%")
    
    print("\n✅ Lightweight model test complete!")
