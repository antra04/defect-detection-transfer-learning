"""
train_mobilenet.py
Transfer Learning with MobileNetV2 - INDUSTRY STANDARD
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

print("="*70)
print("TRANSFER LEARNING WITH MOBILENETV2")
print("="*70)

def load_data():
    print("\n📂 Loading data...")
    data_path = 'data/processed_simple/metal_nut'
    images = []
    labels = []
    
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
    
    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    
    print(f"   Total: {len(X)}")
    print(f"   GOOD: {np.sum(y==0)} ({np.sum(y==0)/len(y)*100:.1f}%)")
    print(f"   DEFECT: {np.sum(y==1)} ({np.sum(y==1)/len(y)*100:.1f}%)")
    
    return X, y

def build_mobilenet_model():
    """Build MobileNetV2 with custom head"""
    print("\n🏗️ Building MobileNetV2 transfer learning model...")
    
    # Load pretrained MobileNetV2 (without top classifier)
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'  # Pretrained on 1.4M images!
    )
    
    # Freeze base model (use pretrained features)
    base_model.trainable = False
    
    print(f"   ✅ Loaded MobileNetV2 pretrained on ImageNet")
    print(f"   🔒 Frozen {len(base_model.layers)} layers (using pretrained weights)")
    
    # Build model
    model = keras.Sequential([
        base_model,
        
        # Custom classifier head
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')
    ])
    
    return model

def main():
    # Load
    X, y = load_data()
    
    # Split
    print("\n✂️ Splitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp
    )
    
    print(f"   Train: {len(X_train)} (GOOD: {np.sum(y_train==0)}, DEFECT: {np.sum(y_train==1)})")
    print(f"   Val:   {len(X_val)} (GOOD: {np.sum(y_val==0)}, DEFECT: {np.sum(y_val==1)})")
    print(f"   Test:  {len(X_test)} (GOOD: {np.sum(y_test==0)}, DEFECT: {np.sum(y_test==1)})")
    
    # Moderate class weights (MobileNet is smarter, doesn't need extreme)
    print("\n⚖️ Setting class weights...")
    class_weights = {
        0: 1.0,
        1: 2.0  # 2x penalty for defects
    }
    
    print(f"   GOOD: {class_weights[0]:.1f}x")
    print(f"   DEFECT: {class_weights[1]:.1f}x")
    
    # Convert
    from tensorflow.keras.utils import to_categorical
    y_train_cat = to_categorical(y_train, 2)
    y_val_cat = to_categorical(y_val, 2)
    y_test_cat = to_categorical(y_test, 2)
    
    # Build
    model = build_mobilenet_model()
    
    # Compile
    print("\n⚙️ Compiling...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Higher LR for transfer learning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n📋 Model Summary:")
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train
    print("\n" + "="*70)
    print("🚀 TRAINING WITH TRANSFER LEARNING")
    print("="*70)
    print("   Strategy: Use MobileNetV2's pretrained features")
    print("   Only training: Final classifier layers")
    print("   This should work MUCH better!")
    print("="*70)
    
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=30,
        batch_size=16,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n" + "="*70)
    print("📊 EVALUATION")
    print("="*70)
    
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")
    
    predictions = model.predict(X_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    
    good_mask = y_test == 0
    defect_mask = y_test == 1
    
    good_acc = np.mean(y_pred[good_mask] == 0) if np.sum(good_mask) > 0 else 0
    defect_acc = np.mean(y_pred[defect_mask] == 1) if np.sum(defect_mask) > 0 else 0
    
    print(f"\n📈 Per-Class Performance:")
    print(f"   GOOD:   {good_acc*100:.1f}% ({np.sum(y_pred[good_mask]==0)}/{np.sum(good_mask)})")
    print(f"   DEFECT: {defect_acc*100:.1f}% ({np.sum(y_pred[defect_mask]==1)}/{np.sum(defect_mask)})")
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n🔍 Confusion Matrix:")
    print(f"          Predicted")
    print(f"          GOOD  DEFECT")
    print(f"Actual GOOD    {cm[0,0]:3d}    {cm[0,1]:3d}")
    print(f"       DEFECT  {cm[1,0]:3d}    {cm[1,1]:3d}")
    
    print(f"\n📊 Detailed Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['GOOD', 'DEFECT'],
                                digits=3))
    
    # Save
    print("\n💾 Saving model...")
    model_path = 'models/mobilenet_model.keras'
    model.save(model_path)
    print(f"   ✅ Saved: {model_path}")
    
    # Plot
    print("\n📈 Creating plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history['accuracy'], label='Train', linewidth=2, marker='o')
    axes[0].plot(history.history['val_accuracy'], label='Val', linewidth=2, marker='s')
    axes[0].set_title('MobileNetV2 Transfer Learning - Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['loss'], label='Train', linewidth=2, marker='o')
    axes[1].plot(history.history['val_loss'], label='Val', linewidth=2, marker='s')
    axes[1].set_title('MobileNetV2 Transfer Learning - Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = 'outputs/mobilenet_training.png'
    plt.savefig(plot_path, dpi=150)
    print(f"   ✅ Saved: {plot_path}")
    plt.show()
    
    # Summary
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📊 FINAL RESULTS:")
    print(f"   Overall Accuracy: {test_acc*100:.1f}%")
    print(f"   GOOD Detection:   {good_acc*100:.1f}%")
    print(f"   DEFECT Detection: {defect_acc*100:.1f}%")
    
    if defect_acc > 0.3:
        print("\n🎉 SUCCESS! Model is detecting defects!")
    else:
        print("\n⚠️ Still struggling with defects...")
    
    print("\n💾 Saved Files:")
    print(f"   Model: {model_path}")
    print(f"   Plot:  {plot_path}")
    print("\n🎯 Next: Update app.py to use mobilenet_model.keras")
    print("="*70)

if __name__ == "__main__":
    main()
