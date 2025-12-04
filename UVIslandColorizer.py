#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UV Island Coloring Tool - Complete version with all methods
Includes: Original fast/complex, Optimized OpenCV/scikit-image
Full support for images with and without alpha channels
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import colorsys
import random
from collections import deque
import time
import warnings
warnings.filterwarnings('ignore')

# ==================== LIBRARY AVAILABILITY CHECK ====================
LIBRARIES = {
    'scipy': False,
    'cv2': False,
    'skimage': False
}

try:
    from scipy import ndimage
    from scipy.spatial import KDTree
    from scipy.ndimage import distance_transform_edt
    LIBRARIES['scipy'] = True
except ImportError:
    LIBRARIES['scipy'] = False
    print("Note: SciPy not available - fast mode will use fallback")

try:
    import cv2
    LIBRARIES['cv2'] = True
except ImportError:
    LIBRARIES['cv2'] = False

try:
    from skimage.measure import label as ski_label
    from skimage.morphology import dilation, disk
    from skimage.filters import threshold_otsu
    LIBRARIES['skimage'] = True
except ImportError:
    LIBRARIES['skimage'] = False

# ==================== SUPPORTED FORMATS ====================
SUPPORTED_FORMATS = {
    '.png', '.jpg', '.jpeg', '.bmp', '.tga', '.tiff', '.tif', 
    '.webp', '.gif', '.ico', '.ppm', '.pgm', '.pbm'
}

# ==================== IMAGE LOADER WITH ALPHA HANDLING ====================
def load_image_safe(image_path, use_opencv=True, solid_threshold=0.9, invert_mask=False):
    """
    Safely load image with alpha channel handling
    
    Args:
        image_path: Path to image file
        use_opencv: Whether to try OpenCV first
        solid_threshold: Brightness threshold for solid regions (0.0-1.0)
        invert_mask: Invert mask for images without alpha
        
    Returns:
        PIL Image object with RGBA mode
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File does not exist: {image_path}")
    
    # Get file extension
    _, ext = os.path.splitext(image_path.lower())
    
    # Check if format is supported
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported image format: {ext}")
    
    # List of formats OpenCV has issues with
    opencv_problem_formats = {'.tga', '.ico', '.gif', '.webp'}
    
    try:
        # Try OpenCV first (if requested and format is supported by OpenCV)
        if use_opencv and LIBRARIES['cv2'] and ext not in opencv_problem_formats:
            img_cv = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img_cv is not None:
                # Convert color space if needed
                if len(img_cv.shape) == 3 and img_cv.shape[2] == 3:
                    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                    pil_img = Image.new('RGBA', (img_cv.shape[1], img_cv.shape[0]))
                    pil_img.paste(Image.fromarray(img_cv, 'RGB'), (0, 0))
                elif len(img_cv.shape) == 3 and img_cv.shape[2] == 4:
                    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGBA)
                    pil_img = Image.fromarray(img_cv, 'RGBA')
                else:
                    pil_img = Image.fromarray(img_cv).convert('RGBA')
                
                print(f"Loaded with OpenCV: {image_path}")
                return pil_img
    except Exception as e:
        print(f"OpenCV loading failed, falling back to PIL: {e}")
    
    # Fallback to PIL for all formats
    try:
        pil_img = Image.open(image_path)
        
        # Ensure image is loaded
        pil_img.load()
        
        original_mode = pil_img.mode
        print(f"Original image mode: {original_mode}")
        
        # Handle different image modes
        if original_mode == 'RGBA':
            # Already has alpha channel
            pil_img = pil_img.convert('RGBA')
            
        elif original_mode in ['RGB', 'L', 'P', 'CMYK', '1', 'I', 'F']:
            # No alpha channel - create one based on brightness
            print(f"Note: Image has no alpha channel, creating from brightness threshold {solid_threshold}")
            
            if original_mode == 'P':
                # Palette mode - convert to RGBA with transparency
                pil_img = pil_img.convert('RGBA')
            elif original_mode == '1':
                # 1-bit images
                pil_img = pil_img.convert('L').convert('RGBA')
            elif original_mode in ['I', 'F']:
                # 32-bit integer or floating point
                pil_img = pil_img.convert('RGB').convert('RGBA')
            else:
                # RGB, L, CMYK
                rgb_image = pil_img.convert('RGB')
                pil_img = Image.new('RGBA', rgb_image.size)
                pil_img.paste(rgb_image, (0, 0))
                
                # Create alpha channel based on brightness
                gray_img = rgb_image.convert('L')
                gray_array = np.array(gray_img)
                
                # Normalize to 0-1
                gray_normalized = gray_array / 255.0
                
                # Apply threshold to create alpha mask
                if invert_mask:
                    # Bright areas become transparent
                    alpha_mask = (gray_normalized < solid_threshold).astype(np.uint8) * 255
                else:
                    # Dark areas become transparent (default)
                    alpha_mask = (gray_normalized > solid_threshold).astype(np.uint8) * 255
                
                # Apply alpha channel
                pil_img_array = np.array(pil_img)
                pil_img_array[:, :, 3] = alpha_mask
                pil_img = Image.fromarray(pil_img_array, 'RGBA')
        else:
            # Unknown mode, convert to RGBA
            pil_img = pil_img.convert('RGBA')
        
        print(f"Loaded with PIL: {image_path}")
        return pil_img
        
    except Exception as e:
        raise ValueError(f"Failed to load image with PIL: {e}")

# ==================== BASE COLORIZER ====================
class UVIslandColorizer:
    """Base UV Island Colorizer with common functionality"""
    
    def __init__(self, alpha_threshold=0.1, solid_threshold=0.9, invert_mask=False):
        self.alpha_threshold = alpha_threshold
        self.solid_threshold = solid_threshold
        self.invert_mask = invert_mask
        self.image = None
        self.width = 0
        self.height = 0
        self.pixels = None
    
    def load_image(self, image_path, use_opencv=True):
        """Load image with safe fallback and alpha handling"""
        try:
            self.image = load_image_safe(
                image_path, 
                use_opencv=use_opencv,
                solid_threshold=self.solid_threshold,
                invert_mask=self.invert_mask
            )
            self.width, self.height = self.image.size
            self.pixels = np.array(self.image)
            print(f"Image size: {self.width}x{self.height}")
            
            # Check if image has valid alpha data
            alpha_channel = self.pixels[:, :, 3]
            has_alpha = np.any(alpha_channel > 0)
            print(f"Has alpha data: {has_alpha}")
            if not has_alpha:
                print("Warning: Image appears to be completely transparent")
            
        except Exception as e:
            print(f"Failed to load image: {e}")
            sys.exit(1)
    
    def generate_colors(self, num_colors, mode="pure"):
        """Generate colors with multiple strategies"""
        if num_colors <= 0:
            return []
        
        colors = []
        
        if mode == "pure":
            # Pure colors - maximized contrast
            for i in range(num_colors):
                hue = i / max(num_colors, 1)
                saturation = 0.8 + random.random() * 0.15
                value = 0.7 + random.random() * 0.25
                
                rgb = colorsys.hsv_to_rgb(hue, saturation, value)
                colors.append((
                    int(rgb[0] * 255),
                    int(rgb[1] * 255),
                    int(rgb[2] * 255),
                    255
                ))
        else:
            # Blended colors
            base_num = min(num_colors, 12)
            base_colors = self.generate_colors(base_num, "pure")
            
            for i in range(num_colors):
                if i < len(base_colors):
                    colors.append(base_colors[i])
                else:
                    if len(base_colors) >= 2:
                        c1 = random.choice(base_colors)
                        c2 = random.choice(base_colors)
                        while c2 == c1 and len(base_colors) > 1:
                            c2 = random.choice(base_colors)
                        
                        ratio = random.random() * 0.6 + 0.2
                        r = int(c1[0] * ratio + c2[0] * (1 - ratio))
                        g = int(c1[1] * ratio + c2[1] * (1 - ratio))
                        b = int(c1[2] * ratio + c2[2] * (1 - ratio))
                        
                        hsv = colorsys.rgb_to_hsv(r/255, g/255, b/255)
                        hsv = ((hsv[0] + 0.3) % 1.0,
                               min(1.0, hsv[1] * 1.1),
                               min(1.0, hsv[2] * 1.05))
                        
                        rgb = colorsys.hsv_to_rgb(*hsv)
                        colors.append((
                            int(rgb[0] * 255),
                            int(rgb[1] * 255),
                            int(rgb[2] * 255),
                            255
                        ))
                    else:
                        colors.append((
                            random.randint(0, 255),
                            random.randint(0, 255),
                            random.randint(0, 255),
                            255
                        ))
        
        return colors
    
    def save_image(self, image, output_path):
        """Save image with optimization"""
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        try:
            # Always save as PNG for transparency support
            if not output_path.lower().endswith('.png'):
                output_path = os.path.splitext(output_path)[0] + '.png'
                print(f"Note: Output format changed to PNG for transparency support")
            
            # Try OpenCV for faster saving if available
            if LIBRARIES['cv2'] and image.mode == 'RGBA':
                try:
                    img_array = np.array(image)
                    bgra = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGRA)
                    cv2.imwrite(output_path, bgra, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                    print(f"Saved with OpenCV: {output_path}")
                    return
                except Exception as e:
                    print(f"OpenCV save failed, using PIL: {e}")
            
            # Fallback to PIL
            image.save(output_path, 'PNG', optimize=True)
            print(f"Saved with PIL: {output_path}")
            
        except Exception as e:
            print(f"Failed to save image: {e}")
    
    def create_mask_from_brightness(self, threshold=None):
        """
        Create alpha mask from brightness for images without alpha channel
        
        Args:
            threshold: Brightness threshold (0.0-1.0)
            
        Returns:
            Binary mask array
        """
        if threshold is None:
            threshold = self.solid_threshold
        
        # Convert to grayscale
        if len(self.pixels.shape) == 3 and self.pixels.shape[2] >= 3:
            # RGB or RGBA image
            gray = np.mean(self.pixels[:, :, :3], axis=2) / 255.0
        else:
            # Grayscale image
            gray = self.pixels[:, :, 0] / 255.0
        
        # Apply threshold
        if self.invert_mask:
            mask = gray < threshold
        else:
            mask = gray > threshold
        
        return mask.astype(np.uint8) * 255

# ==================== ORIGINAL FAST MODE ====================
class OriginalFastColorizer(UVIslandColorizer):
    """Original fast mode using SciPy"""
    
    def __init__(self, alpha_threshold=0.1, border_scale=0.1, fade_alpha=False, 
                 solid_threshold=0.9, invert_mask=False):
        super().__init__(alpha_threshold, solid_threshold, invert_mask)
        self.border_scale = max(0.0, min(2.0, border_scale))
        self.fade_alpha = fade_alpha
        
        if not LIBRARIES['scipy']:
            print("Warning: SciPy not available for fast mode")
    
    def colorize(self, use_pure_first=True):
        """Original fast colorization method"""
        print("Running original fast mode...")
        
        if not LIBRARIES['scipy']:
            print("Error: SciPy required for original fast mode")
            return self.image
        
        # Create mask from alpha channel or brightness
        if np.all(self.pixels[:, :, 3] == 0):
            # No alpha data, create from brightness
            print("Creating mask from brightness...")
            mask = self.create_mask_from_brightness()
        else:
            # Use alpha channel
            mask = (self.pixels[:, :, 3] > self.alpha_threshold * 255).astype(int)
        
        # Label connected regions
        structure = np.ones((3, 3), dtype=int)
        labels, num_islands = ndimage.label(mask, structure=structure)
        
        print(f"Found {num_islands} UV island(s)")
        
        if num_islands == 0:
            print("Warning: No UV islands found")
            return self.image
        
        # Generate colors
        if use_pure_first and num_islands <= 20:
            colors = self.generate_colors(num_islands, "pure")
        else:
            colors = self.generate_colors(num_islands, "blended")
        
        # Apply colors
        output = np.zeros_like(self.pixels)
        for label in range(1, num_islands + 1):
            mask = labels == label
            output[mask] = colors[label - 1]
        
        # Edge processing
        if self.border_scale > 0:
            output = self._process_edges(output, labels)
        
        return Image.fromarray(output, 'RGBA')
    
    def _process_edges(self, output, labels):
        """Process edges with distance transform"""
        background_mask = labels == 0
        
        if not np.any(background_mask):
            return output
        
        distances, indices = distance_transform_edt(
            background_mask, 
            return_distances=True, 
            return_indices=True
        )
        
        nearest_colors = output[indices[0], indices[1]]
        max_dist = np.max(distances) * self.border_scale
        
        if max_dist > 0:
            if self.fade_alpha:
                decay = np.clip(1.0 - distances / max_dist, 0, 1)
            else:
                decay = np.where(distances <= max_dist, 1.0, 0.0)
            
            close_enough = distances <= max_dist
            
            for c in range(3):
                output[:, :, c] = np.where(
                    background_mask & close_enough,
                    nearest_colors[:, :, c],
                    output[:, :, c]
                )
            
            if self.fade_alpha:
                output[:, :, 3] = np.where(
                    background_mask & close_enough,
                    nearest_colors[:, :, 3] * decay,
                    output[:, :, 3]
                )
            else:
                output[:, :, 3] = np.where(
                    background_mask & close_enough,
                    nearest_colors[:, :, 3],
                    output[:, :, 3]
                )
        
        return output

# ==================== ORIGINAL COMPLEX MODE ====================
class OriginalComplexColorizer(UVIslandColorizer):
    """Original complex mode with BFS search"""
    
    def __init__(self, alpha_threshold=0.1, max_search_steps=20, fade_alpha=False,
                 solid_threshold=0.9, invert_mask=False):
        super().__init__(alpha_threshold, solid_threshold, invert_mask)
        self.max_search_steps = max_search_steps
        self.fade_alpha = fade_alpha
        
        self.offsets = [
            (-1, 0), (1, 0), (0, 1), (0, -1),
            (-1, 1), (1, 1), (1, -1), (-1, -1)
        ]
    
    def find_islands(self):
        """Find islands using BFS (original method)"""
        island_labels = np.zeros((self.height, self.width), dtype=int)
        visited = np.zeros((self.height, self.width), dtype=bool)
        islands = []
        current_label = 0
        
        # Create mask from alpha or brightness
        if np.all(self.pixels[:, :, 3] == 0):
            # No alpha data, create from brightness
            alpha_mask = self.create_mask_from_brightness()
            use_alpha = False
        else:
            # Use alpha channel
            alpha_mask = self.pixels[:, :, 3]
            use_alpha = True
        
        total_pixels = self.height * self.width
        processed = 0
        
        for y in range(self.height):
            for x in range(self.width):
                processed += 1
                
                # Progress display
                if processed % (total_pixels // 10 + 1) == 0:
                    progress = processed / total_pixels * 100
                    sys.stdout.write(f"\rFinding islands: {progress:.1f}%")
                    sys.stdout.flush()
                
                # Determine if pixel is UV based on alpha or brightness
                if use_alpha:
                    is_uv = alpha_mask[y, x] > self.alpha_threshold * 255
                else:
                    is_uv = alpha_mask[y, x] > 0
                
                if is_uv and not visited[y, x]:
                    current_label += 1
                    island_pixels = set()
                    
                    queue = deque([(x, y)])
                    visited[y, x] = True
                    
                    while queue:
                        cx, cy = queue.popleft()
                        island_labels[cy, cx] = current_label
                        island_pixels.add((cx, cy))
                        
                        for dx, dy in self.offsets:
                            nx, ny = cx + dx, cy + dy
                            
                            if (0 <= nx < self.width and 0 <= ny < self.height and
                                not visited[ny, nx]):
                                
                                # Check neighbor pixel
                                if use_alpha:
                                    neighbor_is_uv = alpha_mask[ny, nx] > self.alpha_threshold * 255
                                else:
                                    neighbor_is_uv = alpha_mask[ny, nx] > 0
                                
                                if neighbor_is_uv:
                                    visited[ny, nx] = True
                                    queue.append((nx, ny))
                    
                    islands.append(island_pixels)
        
        print()  # New line after progress
        return island_labels, islands
    
    def colorize(self, use_pure_first=True):
        """Original complex colorization method"""
        print("Running original complex mode...")
        
        island_labels, islands = self.find_islands()
        num_islands = len(islands)
        print(f"Found {num_islands} UV island(s)")
        
        if num_islands == 0:
            print("Warning: No UV islands found")
            return self.image
        
        # Generate colors
        if use_pure_first:
            pure_limit = min(num_islands, 20)
            colors = self.generate_colors(pure_limit, "pure")
            
            if num_islands > pure_limit:
                additional_colors = self.generate_colors(num_islands - pure_limit, "blended")
                colors.extend(additional_colors)
        else:
            colors = self.generate_colors(num_islands, "blended")
        
        # Apply colors
        output_pixels = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        
        for i, island_pixels in enumerate(islands):
            color = colors[i % len(colors)]
            for x, y in island_pixels:
                output_pixels[y, x] = color
        
        # Edge processing
        if self.max_search_steps > 0:
            print("Processing edges...")
            self._process_edges_original(output_pixels, island_labels)
        
        return Image.fromarray(output_pixels, 'RGBA')
    
    def _process_edges_original(self, output_pixels, island_labels):
        """Original edge processing method"""
        # Find all background pixels
        background_indices = np.where(island_labels == 0)
        total_background = len(background_indices[0])
        
        if total_background == 0:
            return
        
        # Use KDTree if available for faster search
        if LIBRARIES['scipy']:
            foreground_indices = np.where(island_labels > 0)
            foreground_points = list(zip(foreground_indices[1], foreground_indices[0]))
            
            if foreground_points:
                tree = KDTree(foreground_points)
                use_kdtree = True
            else:
                use_kdtree = False
        else:
            use_kdtree = False
        
        # Process with progress display
        print("Edge processing progress: 0%", end="")
        
        for idx in range(total_background):
            y, x = background_indices[0][idx], background_indices[1][idx]
            
            # Progress update
            if idx % (total_background // 10 + 1) == 0:
                progress = idx / total_background * 100
                sys.stdout.write(f"\rEdge processing progress: {progress:.1f}%")
                sys.stdout.flush()
            
            # Find nearest island
            if use_kdtree:
                dist, nearest_idx = tree.query([(x, y)], k=1)
                nearest_x, nearest_y = foreground_points[nearest_idx[0]]
                min_dist = dist[0]
            else:
                # Brute force search within range
                min_dist = float('inf')
                nearest_x, nearest_y = 0, 0
                
                search_range = min(self.max_search_steps, 
                                  min(self.width, self.height) // 2)
                
                for dy in range(-search_range, search_range + 1):
                    fy = y + dy
                    if 0 <= fy < self.height:
                        for dx in range(-search_range, search_range + 1):
                            fx = x + dx
                            if 0 <= fx < self.width and island_labels[fy, fx] > 0:
                                dist = np.sqrt(dx*dx + dy*dy)
                                if dist < min_dist:
                                    min_dist = dist
                                    nearest_x, nearest_y = fx, fy
            
            # Apply color if within search range
            if min_dist < self.max_search_steps:
                nearest_color = output_pixels[nearest_y, nearest_x]
                
                if self.fade_alpha:
                    alpha = max(0, 255 - int(min_dist * (255 / self.max_search_steps)))
                else:
                    alpha = 255
                
                if alpha > 0:
                    output_pixels[y, x] = (
                        nearest_color[0],
                        nearest_color[1],
                        nearest_color[2],
                        alpha
                    )
        
        print("\rEdge processing completed.      ")

# ==================== OPTIMIZED OPENCV MODE ====================
class OpenCVColorizer(UVIslandColorizer):
    """Optimized mode using OpenCV"""
    
    def __init__(self, alpha_threshold=0.1, border_scale=0.1, fade_alpha=False,
                 solid_threshold=0.9, invert_mask=False):
        super().__init__(alpha_threshold, solid_threshold, invert_mask)
        self.border_scale = max(0.0, min(2.0, border_scale))
        self.fade_alpha = fade_alpha
        
        if not LIBRARIES['cv2']:
            print("Warning: OpenCV not available for this mode")
    
    def colorize(self, use_pure_first=True):
        """OpenCV optimized colorization"""
        print("Running OpenCV optimized mode...")
        
        if not LIBRARIES['cv2']:
            print("Error: OpenCV required for this mode")
            return self.image
        
        # Create mask from alpha channel or brightness
        if np.all(self.pixels[:, :, 3] == 0):
            # No alpha data, create from brightness
            print("Creating mask from brightness...")
            mask = self.create_mask_from_brightness()
        else:
            # Use alpha channel
            mask = (self.pixels[:, :, 3] > self.alpha_threshold * 255).astype(np.uint8)
        
        # Connected components with OpenCV
        num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
        num_islands = num_labels - 1
        
        print(f"Found {num_islands} UV island(s)")
        
        if num_islands == 0:
            print("Warning: No UV islands found")
            return self.image
        
        # Generate colors
        if use_pure_first and num_islands <= 20:
            colors = self.generate_colors(num_islands, "pure")
        else:
            colors = self.generate_colors(num_islands, "blended")
        
        # Apply colors
        output = np.zeros_like(self.pixels)
        for label in range(1, num_islands + 1):
            mask = labels == label
            output[mask] = colors[label - 1]
        
        # OpenCV edge processing
        if self.border_scale > 0:
            output = self._process_edges_opencv(output, labels)
        
        return Image.fromarray(output, 'RGBA')
    
    def _process_edges_opencv(self, output, labels):
        """OpenCV edge processing"""
        island_mask = (labels > 0).astype(np.uint8) * 255
        
        # Calculate border size
        border_size = int(min(self.width, self.height) * self.border_scale * 0.01)
        border_size = max(1, border_size)
        
        if self.fade_alpha:
            # Distance transform for fading
            dist_mask = 255 - island_mask
            dist_transform = cv2.distanceTransform(dist_mask, cv2.DIST_L2, 3)
            cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
            
            alpha_mask = np.clip(1.0 - dist_transform * self.border_scale, 0, 1)
            
            # Use inpainting to fill
            unknown_mask = (island_mask == 0).astype(np.uint8) * 255
            output_bgr = cv2.cvtColor(output, cv2.COLOR_RGBA2BGRA)
            
            inpainted = cv2.inpaint(
                output_bgr[:, :, :3],
                unknown_mask,
                border_size * 3,
                cv2.INPAINT_TELEA
            )
            
            inpainted_rgba = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGBA)
            background_mask = island_mask == 0
            
            for c in range(3):
                output[:, :, c] = np.where(
                    background_mask,
                    inpainted_rgba[:, :, c],
                    output[:, :, c]
                )
            
            output[:, :, 3] = np.where(
                background_mask,
                inpainted_rgba[:, :, 3] * alpha_mask,
                output[:, :, 3]
            )
        else:
            # Simple dilation for sharp edges
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_size*2+1, border_size*2+1))
            dilated_mask = cv2.dilate(island_mask, kernel, iterations=1)
            border_mask = cv2.bitwise_and(dilated_mask, cv2.bitwise_not(island_mask))
            
            if np.any(border_mask):
                # Find border coordinates
                border_coords = np.column_stack(np.where(border_mask))
                
                # For each border pixel, find nearest island pixel
                for y, x in border_coords:
                    # Search in a small region around the border pixel
                    y_start = max(0, y - 5)
                    y_end = min(self.height, y + 6)
                    x_start = max(0, x - 5)
                    x_end = min(self.width, x + 6)
                    
                    region = island_mask[y_start:y_end, x_start:x_end]
                    if np.any(region):
                        region_coords = np.argwhere(region)
                        if len(region_coords) > 0:
                            # Find closest in region
                            distances = np.sqrt((region_coords[:, 0] - (y - y_start))**2 + 
                                               (region_coords[:, 1] - (x - x_start))**2)
                            closest_idx = np.argmin(distances)
                            nearest_y = region_coords[closest_idx, 0] + y_start
                            nearest_x = region_coords[closest_idx, 1] + x_start
                            output[y, x] = output[nearest_y, nearest_x]
        
        return output

# ==================== OPTIMIZED SKIMAGE MODE ====================
class SkimageColorizer(UVIslandColorizer):
    """Optimized mode using scikit-image"""
    
    def __init__(self, alpha_threshold=0.1, border_scale=0.1, fade_alpha=False,
                 solid_threshold=0.9, invert_mask=False):
        super().__init__(alpha_threshold, solid_threshold, invert_mask)
        self.border_scale = max(0.0, min(2.0, border_scale))
        self.fade_alpha = fade_alpha
        
        if not LIBRARIES['skimage']:
            print("Warning: scikit-image not available for this mode")
    
    def colorize(self, use_pure_first=True):
        """scikit-image optimized colorization"""
        print("Running scikit-image optimized mode...")
        
        if not LIBRARIES['skimage']:
            print("Error: scikit-image required for this mode")
            return self.image
        
        # Create mask from alpha channel or brightness
        if np.all(self.pixels[:, :, 3] == 0):
            # No alpha data, create from brightness
            print("Creating mask from brightness...")
            mask = self.create_mask_from_brightness()
        else:
            # Use alpha channel
            mask = (self.pixels[:, :, 3] > self.alpha_threshold * 255).astype(np.uint8)
        
        # Label with scikit-image
        labels = ski_label(mask, connectivity=2, background=0)
        num_islands = labels.max()
        
        print(f"Found {num_islands} UV island(s)")
        
        if num_islands == 0:
            print("Warning: No UV islands found")
            return self.image
        
        # Generate colors
        if use_pure_first and num_islands <= 20:
            colors = self.generate_colors(num_islands, "pure")
        else:
            colors = self.generate_colors(num_islands, "blended")
        
        # Apply colors
        output = np.zeros_like(self.pixels)
        for label in range(1, num_islands + 1):
            mask = labels == label
            output[mask] = colors[label - 1]
        
        # Edge processing
        if self.border_scale > 0:
            output = self._process_edges_skimage(output, labels)
        
        return Image.fromarray(output, 'RGBA')
    
    def _process_edges_skimage(self, output, labels):
        """scikit-image edge processing"""
        island_mask = (labels > 0).astype(np.uint8)
        
        border_size = int(min(self.width, self.height) * self.border_scale * 0.01)
        border_size = max(1, border_size)
        
        if self.fade_alpha and LIBRARIES['scipy']:
            # Distance transform with fading
            distances = distance_transform_edt(~island_mask.astype(bool))
            max_dist = distances.max()
            
            if max_dist > 0:
                normalized_dist = distances / max_dist
                alpha_mask = np.clip(1.0 - normalized_dist * self.border_scale, 0, 1)
                
                background_mask = island_mask == 0
                _, indices = distance_transform_edt(background_mask, return_indices=True)
                nearest_colors = output[indices[0], indices[1]]
                
                for c in range(3):
                    output[:, :, c] = np.where(
                        background_mask,
                        nearest_colors[:, :, c],
                        output[:, :, c]
                    )
                
                output[:, :, 3] = np.where(
                    background_mask,
                    nearest_colors[:, :, 3] * alpha_mask,
                    output[:, :, 3]
                )
        else:
            # Morphological dilation
            selem = disk(border_size)
            dilated_mask = dilation(island_mask, selem)
            border_mask = (dilated_mask - island_mask) > 0
            
            if np.any(border_mask) and LIBRARIES['scipy']:
                # Find nearest island for each border pixel
                island_coords = np.column_stack(np.where(island_mask))
                border_coords = np.column_stack(np.where(border_mask))
                
                if len(island_coords) > 0:
                    tree = KDTree(island_coords)
                    distances, indices = tree.query(border_coords)
                    
                    for i, (y, x) in enumerate(border_coords):
                        nearest_y, nearest_x = island_coords[indices[i]]
                        output[y, x] = output[nearest_y, nearest_x]
        
        return output

# ==================== BATCH PROCESSING ====================
def batch_process(input_dir, output_dir, args):
    """Process all images in a directory"""
    if not os.path.isdir(input_dir):
        print(f"Error: Input is not a directory: {input_dir}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect image files
    image_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in SUPPORTED_FORMATS:
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        print(f"No supported image files found in: {input_dir}")
        return
    
    print(f"Found {len(image_files)} image files to process")
    
    # Process each file
    for i, input_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] Processing: {input_path}")
        
        # Create output path
        rel_path = os.path.relpath(input_path, input_dir)
        output_path = os.path.join(output_dir, rel_path)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Process single file
        process_single_image(input_path, output_path, args)

# ==================== SINGLE IMAGE PROCESSING ====================
def process_single_image(input_path, output_path, args):
    """Process a single image file"""
    start_time = time.time()
    
    # Create appropriate colorizer based on mode
    if args.mode == "fast":
        colorizer = OriginalFastColorizer(
            alpha_threshold=args.threshold,
            border_scale=args.border_scale,
            fade_alpha=args.fade_alpha,
            solid_threshold=args.solid_threshold,
            invert_mask=args.invert_mask
        )
        use_opencv_loading = False
        
    elif args.mode == "complex":
        colorizer = OriginalComplexColorizer(
            alpha_threshold=args.threshold,
            max_search_steps=int(args.border_scale * 100),
            fade_alpha=args.fade_alpha,
            solid_threshold=args.solid_threshold,
            invert_mask=args.invert_mask
        )
        use_opencv_loading = False
        
    elif args.mode == "opencv":
        colorizer = OpenCVColorizer(
            alpha_threshold=args.threshold,
            border_scale=args.border_scale,
            fade_alpha=args.fade_alpha,
            solid_threshold=args.solid_threshold,
            invert_mask=args.invert_mask
        )
        use_opencv_loading = True
        
    elif args.mode == "skimage":
        colorizer = SkimageColorizer(
            alpha_threshold=args.threshold,
            border_scale=args.border_scale,
            fade_alpha=args.fade_alpha,
            solid_threshold=args.solid_threshold,
            invert_mask=args.invert_mask
        )
        use_opencv_loading = True
        
    else:
        print(f"Error: Unknown mode '{args.mode}'")
        return
    
    # Load image
    colorizer.load_image(input_path, use_opencv=use_opencv_loading)
    
    # Colorize
    use_pure = not args.blended_only
    result = colorizer.colorize(use_pure_first=use_pure)
    
    # Save
    colorizer.save_image(result, output_path)
    
    elapsed = time.time() - start_time
    print(f"Processing time: {elapsed:.3f} seconds")

# ==================== COMMAND LINE INTERFACE ====================
def main():
    parser = argparse.ArgumentParser(
        description="UV Island Coloring Tool - Complete version with alpha handling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Supported Formats: {', '.join(sorted(SUPPORTED_FORMATS))}

Available Modes:
  fast      - Original fast mode (requires SciPy)
  complex   - Original complex mode (slow but precise, no external dependencies)
  opencv    - Optimized with OpenCV (fastest, requires OpenCV)
  skimage   - Optimized with scikit-image (requires scikit-image)

For images without alpha channels:
  - Alpha mask is created from brightness
  - Use --solid-threshold to adjust sensitivity (higher = more strict)
  - Use --invert-mask for light backgrounds

Library Status:
  SciPy:       {'✓ Available' if LIBRARIES['scipy'] else '✗ Not available'}
  OpenCV:      {'✓ Available' if LIBRARIES['cv2'] else '✗ Not available'}
  scikit-image: {'✓ Available' if LIBRARIES['skimage'] else '✗ Not available'}

Examples:
  Single file with alpha:
    %(prog)s input.tga output.png --mode fast
  
  Single file without alpha (dark lines on light background):
    %(prog)s input.jpg output.png --mode opencv --solid-threshold 0.8
  
  Single file without alpha (light lines on dark background):
    %(prog)s input.jpg output.png --mode opencv --solid-threshold 0.8 --invert-mask
  
  Batch processing:
    %(prog)s input_folder output_folder --batch --mode fast
        """
    )
    
    # Input/output arguments
    parser.add_argument('input', help='Input image file or directory')
    parser.add_argument('output', help='Output image file or directory')
    
    # Processing mode
    parser.add_argument('--mode', choices=['fast', 'complex', 'opencv', 'skimage'],
                       default='fast', help='Processing mode (default: fast)')
    
    # Alpha channel parameters
    parser.add_argument('--threshold', type=float, default=0.1,
                       help='Alpha threshold for images with alpha (0.0-1.0, default: 0.1)')
    parser.add_argument('--solid-threshold', type=float, default=0.9,
                       help='Brightness threshold for images without alpha (0.0-1.0, default: 0.9)')
    parser.add_argument('--invert-mask', action='store_true',
                       help='Invert mask for images without alpha (default: dark=transparent)')
    
    # Border processing
    parser.add_argument('--border-scale', type=float, default=0.1,
                       help='Border scale (0.0-2.0, default: 0.1)')
    parser.add_argument('--fade-alpha', action='store_true',
                       help='Fade alpha at borders (default: sharp edges)')
    
    # Color settings
    parser.add_argument('--blended-only', action='store_true',
                       help='Use only blended colors (default: pure colors first)')
    
    # Batch processing
    parser.add_argument('--batch', action='store_true',
                       help='Batch process all images in input directory')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("UV ISLAND COLORING TOOL - COMPLETE VERSION")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Alpha threshold: {args.threshold}")
    print(f"Brightness threshold: {args.solid_threshold}")
    print(f"Invert mask: {'Yes' if args.invert_mask else 'No'}")
    print(f"Border scale: {args.border_scale}")
    print(f"Alpha fading: {'Enabled' if args.fade_alpha else 'Disabled'}")
    print(f"Color strategy: {'Blended only' if args.blended_only else 'Pure first'}")
    print("-" * 60)
    
    # Check mode availability
    if args.mode == 'fast' and not LIBRARIES['scipy']:
        print("Error: fast mode requires SciPy")
        print("Install: pip install scipy")
        sys.exit(1)
    elif args.mode == 'opencv' and not LIBRARIES['cv2']:
        print("Error: opencv mode requires OpenCV")
        print("Install: pip install opencv-python")
        sys.exit(1)
    elif args.mode == 'skimage' and not LIBRARIES['skimage']:
        print("Error: skimage mode requires scikit-image")
        print("Install: pip install scikit-image")
        sys.exit(1)
    
    # Process based on batch mode
    if args.batch:
        batch_process(args.input, args.output, args)
    else:
        process_single_image(args.input, args.output, args)
    
    print("=" * 60)
    print("PROCESSING COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()