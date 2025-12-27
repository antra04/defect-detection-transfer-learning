# src/data_loader.py
# Load and prepare data for training

import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataLoader:
    """
    Load preprocessed data and prepare for training.
    """
    
    def __init__(self, data_path='data/processed/metal_nut'):
        """
        Initialize data loader.
        
        Parameters:
        -----------
        data_path : str
            Path to preprocessed .npy files
        """
        self.data_path = data_path
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        
        print(f"✅ DataLoader initialized")
        print(f"   Data path: {data_path}")
    
    def load_data(self):
        """
        Load all preprocessed images and create labels.
        
        Returns:
        --------
        tuple : (X, y) where X is images and y is labels
        """
        print("\n" + "="*70)
        print("LOADING PREPROCESSED DATA")
        print("="*70)
        
        images = []
        labels = []
        
        # Load training data (all are GOOD - label 0)
        print("\n📚 Loading training images (GOOD)...")
        train_good_path = os.path.join(self.data_path, 'train', 'good')
        train_files = [f for f in os.listdir(train_good_path) if f.endswith('.npy')]
        
        for file in train_files:
            img = np.load(os.path.join(train_good_path, file))
            images.append(img)
            labels.append(0)  # 0 = GOOD
        
        print(f"   Loaded {len(train_files)} GOOD images")
        
        # Load test data
        test_base = os.path.join(self.data_path, 'test')
        test_categories = [d for d in os.listdir(test_base) 
                          if os.path.isdir(os.path.join(test_base, d))]
        
        print("\n📚 Loading test images...")
        for category in sorted(test_categories):
            category_path = os.path.join(test_base, category)
            category_files = [f for f in os.listdir(category_path) if f.endswith('.npy')]
            
            # Determine label (0=good, 1=defect)
            label = 0 if category == 'good' else 1
            label_name = "GOOD" if label == 0 else "DEFECT"
            
            for file in category_files:
                img = np.load(os.path.join(category_path, file))
                images.append(img)
                labels.append(label)
            
            print(f"   {category:15s}: {len(category_files):3d} images (label={label}, {label_name})")
        
        # Convert to numpy arrays
        X = np.array(images, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)
        
        print("\n" + "="*70)
        print("DATA LOADING COMPLETE")
        print("="*70)
        print(f"Total images: {X.shape[0]}")
        print(f"Image shape: {X.shape[1:]}")
        print(f"Labels shape: {y.shape}")
        print(f"\nClass distribution:")
        print(f"  GOOD (0):   {np.sum(y == 0):3d} images ({np.sum(y == 0)/len(y)*100:.1f}%)")
        print(f"  DEFECT (1): {np.sum(y == 1):3d} images ({np.sum(y == 1)/len(y)*100:.1f}%)")
        
        return X, y
    
    def split_data(self, X, y, val_split=0.15, test_split=0.15, random_state=42):
        """
        Split data into train/validation/test sets.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Images
        y : numpy.ndarray
            Labels
        val_split : float
            Fraction for validation (0.15 = 15%)
        test_split : float
            Fraction for testing (0.15 = 15%)
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        tuple : (X_train, y_train), (X_val, y_val), (X_test, y_test)
        """
        print("\n" + "="*70)
        print("SPLITTING DATA")
        print("="*70)
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_split,
            random_state=random_state,
            stratify=y  # Maintain class distribution
        )
        
        # Second split: separate train and validation
        val_size = val_split / (1 - test_split)  # Adjust val size
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=random_state,
            stratify=y_temp
        )
        
        # Convert labels to one-hot encoding
        # Example: 0 → [1, 0], 1 → [0, 1]
        y_train = to_categorical(y_train, num_classes=2)
        y_val = to_categorical(y_val, num_classes=2)
        y_test = to_categorical(y_test, num_classes=2)
        
        # Store in instance variables
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        
        print(f"\nTraining set:   {X_train.shape[0]} images")
        print(f"  GOOD:   {np.sum(np.argmax(y_train, axis=1) == 0)}")
        print(f"  DEFECT: {np.sum(np.argmax(y_train, axis=1) == 1)}")
        
        print(f"\nValidation set: {X_val.shape[0]} images")
        print(f"  GOOD:   {np.sum(np.argmax(y_val, axis=1) == 0)}")
        print(f"  DEFECT: {np.sum(np.argmax(y_val, axis=1) == 1)}")
        
        print(f"\nTest set:       {X_test.shape[0]} images")
        print(f"  GOOD:   {np.sum(np.argmax(y_test, axis=1) == 0)}")
        print(f"  DEFECT: {np.sum(np.argmax(y_test, axis=1) == 1)}")
        
        print("\n" + "="*70)
        print("✅ DATA SPLIT COMPLETE")
        print("="*70)
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def get_data_generators(self, batch_size=16, augment_train=True):
        """
        Create data generators for training.
        
        Data Augmentation (for training only):
        - Random rotations (±15°)
        - Random zoom (±10%)
        - Random horizontal/vertical flips
        - Random shifts
        
        Why augmentation?
        - Creates more training variety
        - Prevents overfitting
        - Model learns to be rotation/position invariant
        
        Parameters:
        -----------
        batch_size : int
            Number of images per batch
        augment_train : bool
            Whether to apply data augmentation to training set
            
        Returns:
        --------
        tuple : (train_gen, val_gen)
        """
        print("\n" + "="*70)
        print("CREATING DATA GENERATORS")
        print("="*70)
        
        if augment_train:
            print("\n📈 Training augmentation enabled:")
            print("   - Rotation: ±15°")
            print("   - Zoom: ±10%")
            print("   - Horizontal flip: Yes")
            print("   - Vertical flip: Yes")
            print("   - Width shift: ±10%")
            print("   - Height shift: ±10%")
            
            train_datagen = ImageDataGenerator(
                rotation_range=15,
                zoom_range=0.1,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator()
        
        # Validation: no augmentation (test on original images)
        val_datagen = ImageDataGenerator()
        
        # Create generators
        train_generator = train_datagen.flow(
            self.X_train, self.y_train,
            batch_size=batch_size,
            shuffle=True
        )
        
        val_generator = val_datagen.flow(
            self.X_val, self.y_val,
            batch_size=batch_size,
            shuffle=False
        )
        
        print(f"\n✅ Generators created:")
        print(f"   Batch size: {batch_size}")
        print(f"   Training batches per epoch: {len(train_generator)}")
        print(f"   Validation batches per epoch: {len(val_generator)}")
        print("="*70)
        
        return train_generator, val_generator

# ============================================================
# TEST DATA LOADING
# ============================================================

if __name__ == "__main__":
    print("\n" + "🔬 TESTING DATA LOADER" + "\n")
    
    # Initialize loader
    loader = DataLoader(data_path='data/processed/metal_nut')
    
    # Load data
    X, y = loader.load_data()
    
    # Split data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.split_data(X, y)
    
    # Create generators
    train_gen, val_gen = loader.get_data_generators(batch_size=16, augment_train=True)
    
    # Test generator
    print("\n" + "="*70)
    print("TESTING DATA GENERATOR")
    print("="*70)
    batch_X, batch_y = next(train_gen)
    print(f"Batch images shape: {batch_X.shape}")
    print(f"Batch labels shape: {batch_y.shape}")
    print(f"Sample label (one-hot): {batch_y[0]}")
    print(f"Sample label (class): {np.argmax(batch_y[0])}")
    
    print("\n" + "="*70)
    print("✅ DATA LOADER TEST COMPLETE")
    print("="*70)
    print("\nReady for training!")
