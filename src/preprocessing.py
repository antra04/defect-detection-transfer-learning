# src/preprocessing.py
# Day 2 - Image Preprocessing Pipeline

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class ImagePreprocessor:
    """
    Handles all image preprocessing operations for defect detection.
    Each method transforms the image to improve AI model performance.
    """
    
    def __init__(self, target_size=(224, 224)):
        """
        Initialize preprocessor with target image size.
        
        Parameters:
        -----------
        target_size : tuple
            (width, height) for resized images
            224x224 is standard for many CNN architectures
        """
        self.target_size = target_size
        print(f"✅ ImagePreprocessor initialized")
        print(f"   Target size: {target_size}")
    
    def load_image(self, image_path):
        """
        Load image from file path.
        
        Parameters:
        -----------
        image_path : str
            Path to image file
            
        Returns:
        --------
        numpy.ndarray
            Image in BGR format (OpenCV default)
            
        Concept:
        --------
        OpenCV reads images as numpy arrays with shape (H, W, 3)
        Color order is BGR (Blue, Green, Red) - not RGB!
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        img = cv2.imread(image_path)
        
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"✅ Loaded image: {image_path}")
        print(f"   Shape: {img.shape}")
        
        return img
    
    def resize_image(self, img):
        """
        Resize image to target dimensions.
        
        Why:
        ----
        - Neural networks need fixed-size inputs
        - Standardizes all images to same dimensions
        - Reduces computation time
        
        How:
        ----
        Uses INTER_AREA interpolation (best for shrinking images)
        Preserves image quality while reducing size
        
        Parameters:
        -----------
        img : numpy.ndarray
            Input image
            
        Returns:
        --------
        numpy.ndarray
            Resized image
        """
        resized = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        print(f"✅ Resized: {img.shape} → {resized.shape}")
        return resized
    
    def normalize(self, img):
        """
        Normalize pixel values from [0, 255] to [0, 1].
        
        Why:
        ----
        - Neural networks train better with smaller values
        - Prevents gradient explosion/vanishing
        - Standardizes input range
        
        How:
        ----
        Divide each pixel by 255
        Convert to float32 (required for division)
        
        Example:
        --------
        Pixel value 127 becomes 127/255 = 0.498
        
        Parameters:
        -----------
        img : numpy.ndarray
            Input image (0-255 range)
            
        Returns:
        --------
        numpy.ndarray
            Normalized image (0-1 range), dtype=float32
        """
        normalized = img.astype(np.float32) / 255.0
        print(f"✅ Normalized: [{img.min()}, {img.max()}] → [{normalized.min():.3f}, {normalized.max():.3f}]")
        return normalized
    
    def denoise(self, img):
        """
        Apply Gaussian blur to reduce noise.
        
        Why:
        ----
        - Camera sensors introduce random noise
        - Lighting variations create unwanted patterns
        - Smoothing helps AI focus on real features, not noise
        
        How:
        ----
        Gaussian blur = weighted average of nearby pixels
        Center pixel has highest weight, neighbors have lower weight
        Creates smooth, natural-looking blur
        
        Parameters:
        -----------
        img : numpy.ndarray
            Input image
            
        kernel_size : (5, 5)
            Size of blur window (larger = more blur)
            Must be odd numbers
            
        sigmaX : 0
            Gaussian kernel standard deviation
            0 = auto-calculate from kernel size
            
        Returns:
        --------
        numpy.ndarray
            Denoised (blurred) image
        """
        denoised = cv2.GaussianBlur(img, (5, 5), 0)
        print(f"✅ Applied Gaussian blur (5x5 kernel)")
        return denoised
    
    def convert_to_grayscale(self, img):
        """
        Convert color image to grayscale.
        
        Why:
        ----
        - Reduces data from 3 channels to 1 (faster processing)
        - Many defects show better in grayscale
        - Simplifies edge detection
        
        How:
        ----
        Weighted average: Gray = 0.299*R + 0.587*G + 0.114*B
        Weights based on human eye sensitivity
        
        Parameters:
        -----------
        img : numpy.ndarray
            Input BGR image
            
        Returns:
        --------
        numpy.ndarray
            Grayscale image (single channel)
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"✅ Converted to grayscale: {img.shape} → {gray.shape}")
        return gray
    
    def adaptive_threshold(self, gray_img):
        """
        Convert grayscale to binary (black & white only).
        
        Why:
        ----
        - Separates foreground from background
        - Highlights edges and defects
        - Simplifies shape detection
        
        How:
        ----
        Adaptive = threshold varies across image
        Handles uneven lighting better than global threshold
        Each pixel compared to local neighborhood
        
        Parameters:
        -----------
        gray_img : numpy.ndarray
            Grayscale image
            
        maxValue : 255
            Value for "white" pixels
            
        adaptiveMethod : ADAPTIVE_THRESH_GAUSSIAN_C
            Use Gaussian-weighted average
            
        thresholdType : THRESH_BINARY
            Output is binary (0 or 255)
            
        blockSize : 11
            Size of local neighborhood (must be odd)
            
        C : 2
            Constant subtracted from weighted mean
            
        Returns:
        --------
        numpy.ndarray
            Binary image (0 or 255 only)
        """
        binary = cv2.adaptiveThreshold(
            gray_img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # Block size
            2    # Constant C
        )
        print(f"✅ Applied adaptive thresholding")
        return binary
    
    def morphological_operations(self, binary_img):
        """
        Clean up binary image using morphology.
        
        Why:
        ----
        - Remove small noise dots
        - Fill small holes in objects
        - Smooth object boundaries
        
        How:
        ----
        Opening = Erosion followed by Dilation
          - Removes small white noise
        Closing = Dilation followed by Erosion
          - Fills small black holes
        
        Parameters:
        -----------
        binary_img : numpy.ndarray
            Binary image
            
        Returns:
        --------
        numpy.ndarray
            Cleaned binary image
        """
        # Create structuring element (kernel)
        kernel = np.ones((3, 3), np.uint8)
        
        # Opening: removes small white noise
        opened = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
        
        # Closing: fills small black holes
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        print(f"✅ Applied morphological operations (opening + closing)")
        return closed
    
    def detect_edges(self, gray_img):
        """
        Detect edges using Canny edge detector.
        
        Why:
        ----
        - Edges indicate boundaries and defects
        - Highlights scratches, cracks, deformations
        - Essential for defect localization
        
        How:
        ----
        Canny algorithm:
        1. Gaussian blur to reduce noise
        2. Calculate intensity gradients
        3. Non-maximum suppression (thin edges)
        4. Double thresholding
        5. Edge tracking by hysteresis
        
        Parameters:
        -----------
        gray_img : numpy.ndarray
            Grayscale image
            
        threshold1 : 50
            Lower threshold for edge detection
            
        threshold2 : 150
            Upper threshold for edge detection
            
        Returns:
        --------
        numpy.ndarray
            Binary edge map
        """
        edges = cv2.Canny(gray_img, 50, 150)
        print(f"✅ Detected edges using Canny")
        return edges
    
    def preprocess_pipeline(self, image_path, show_steps=False):
        """
        Complete preprocessing pipeline.
        Applies all operations in sequence.
        
        Parameters:
        -----------
        image_path : str
            Path to input image
        show_steps : bool
            If True, display intermediate steps
            
        Returns:
        --------
        numpy.ndarray
            Preprocessed image ready for model
        """
        print("\n" + "="*60)
        print("PREPROCESSING PIPELINE")
        print("="*60)
        
        # Step 1: Load
        img = self.load_image(image_path)
        original = img.copy()
        
        # Step 2: Resize
        img = self.resize_image(img)
        
        # Step 3: Denoise
        img = self.denoise(img)
        
        # Step 4: Normalize
        img = self.normalize(img)
        
        print("="*60)
        print("✅ PREPROCESSING COMPLETE")
        print("="*60)
        
        if show_steps:
            self.visualize_pipeline(original, image_path)
        
        return img
    
    def visualize_pipeline(self, original_img, image_path):
        """
        Visualize all preprocessing steps.
        
        Parameters:
        -----------
        original_img : numpy.ndarray
            Original loaded image
        image_path : str
            Path to image (for title)
        """
        # Re-process to get intermediate steps
        img = cv2.resize(original_img, self.target_size, interpolation=cv2.INTER_AREA)
        denoised = cv2.GaussianBlur(img, (5, 5), 0)
        normalized = img.astype(np.float32) / 255.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        edges = cv2.Canny(gray, 50, 150)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('1. Original', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('2. Resized', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title('3. Denoised (Gaussian Blur)', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(gray, cmap='gray')
        axes[1, 0].set_title('4. Grayscale', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(binary, cmap='gray')
        axes[1, 1].set_title('5. Binary (Adaptive Threshold)', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(edges, cmap='gray')
        axes[1, 2].set_title('6. Edges (Canny)', fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save with descriptive name
        filename = os.path.basename(image_path).replace('.png', '_preprocessing.png')
        output_path = f'outputs/{filename}'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Saved visualization: {output_path}")
        plt.show()

# Test the preprocessor
if __name__ == "__main__":
    print("\n" + "🔬 TESTING IMAGE PREPROCESSOR" + "\n")
    
    # Initialize
    preprocessor = ImagePreprocessor(target_size=(224, 224))
    
    # Test on a good image
    print("\n📸 Testing on GOOD image:")
    good_img_path = "data/raw/metal_nut/train/good/000.png"
    processed_good = preprocessor.preprocess_pipeline(good_img_path, show_steps=True)
    
    # Test on a defect image
    print("\n📸 Testing on DEFECT image (scratch):")
    defect_img_path = "data/raw/metal_nut/test/scratch/000.png"
    processed_defect = preprocessor.preprocess_pipeline(defect_img_path, show_steps=True)
