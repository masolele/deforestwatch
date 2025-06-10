import numpy as np
from skimage.util import view_as_windows
import tensorflow as tf
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

class LargeImagePredictor:
    def __init__(self, model, patch_size=64, overlap=32, batch_size=4):
        """
        Args:
            model: Loaded Keras/TensorFlow model
            patch_size: Model input size (square)
            overlap: Overlap between patches (pixels)
            batch_size: Prediction batch size
        """
        self.model = model
        self.patch_size = patch_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.step = patch_size - overlap
        
        # Create weighting window for smooth blending
        self.window = np.hanning(patch_size)[:, None] * np.hanning(patch_size)[None, :]
        self.window = np.repeat(self.window[..., np.newaxis], model.output_shape[-1], axis=-1)

    def predict_large_image(self, image):
        """Main prediction function with all optimizations"""
        # 1. Mirror padding to handle edges
        pad_h = (self.patch_size - image.shape[0] % self.step) % self.step
        pad_w = (self.patch_size - image.shape[1] % self.step) % self.step
        padded_img = np.pad(image, 
                          ((self.overlap, pad_h + self.overlap),
                          (self.overlap, pad_w + self.overlap),
                          (0, 0)),
                          mode='reflect')
        
        # 2. Extract overlapping patches
        patches = view_as_windows(
            padded_img,
            (self.patch_size, self.patch_size, image.shape[2]),
            step=self.step
        )
        n_h, n_w, _, _, _ = patches.shape
        patches = patches.reshape(-1, self.patch_size, self.patch_size, image.shape[2])
        
        # 3. Batch prediction with progress bar
        preds = np.zeros((len(patches), self.patch_size, self.patch_size, self.model.output_shape[-1]))
        for i in tqdm(range(0, len(patches), self.batch_size), desc="Processing Patches"):
            batch = patches[i:i+self.batch_size]
            preds[i:i+self.batch_size] = self.model.predict(batch, verbose=0)
        
        # 4. Reconstruct with weighted blending
        output_shape = (padded_img.shape[0], padded_img.shape[1], preds.shape[-1])
        full_pred = np.zeros(output_shape)
        weights = np.zeros(output_shape)
        
        idx = 0
        for y in range(n_h):
            for x in range(n_w):
                y_start = y * self.step
                x_start = x * self.step
                
                # Apply window weighting
                weighted_pred = preds[idx] * self.window
                
                full_pred[y_start:y_start+self.patch_size, 
                         x_start:x_start+self.patch_size] += weighted_pred
                weights[y_start:y_start+self.patch_size,
                       x_start:x_start+self.patch_size] += self.window
                idx += 1
        
        # Normalize by weights
        full_pred = full_pred / (weights + 1e-8)
        
        # 5. Remove padding and post-process
        full_pred = full_pred[self.overlap:self.overlap+image.shape[0],
                            self.overlap:self.overlap+image.shape[1]]
        
        # Gaussian smoothing at seams
        for c in range(full_pred.shape[-1]):
            full_pred[..., c] = gaussian_filter(full_pred[..., c], sigma=1)
        
        return full_pred
