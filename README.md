# UV Island Coloring Tool

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A powerful command-line tool for automatically coloring UV islands in UV unwrap maps. Each UV island gets a unique color with maximized contrast between neighboring islands.

## Features

### 🎨 Multi-Mode Support
- **Fast Mode**: Uses SciPy for quick processing (original implementation)
- **Complex Mode**: Precise BFS algorithm with detailed control
- **OpenCV Mode**: Optimized with OpenCV for maximum performance
- **Skimage Mode**: Uses scikit-image for advanced image processing

### 📁 Format Compatibility
- **With Alpha**: PNG, TGA, TIFF, WebP, etc.
- **Without Alpha**: JPEG, BMP, PPM, etc. (automatic alpha generation)
- **Batch Processing**: Process entire folders of images

### 🎯 Intelligent Coloring
- **Pure Colors First**: Maximize contrast between islands
- **Blended Colors**: When many islands need distinct colors
- **Automatic Detection**: Works with or without alpha channels
- **Border Control**: Adjustable border expansion with fade options

### ⚡ Performance Optimizations
- **Vectorized Operations**: NumPy-based for speed
- **Parallel Processing**: Multi-core support where available
- **Memory Efficient**: Handles large images smoothly

## Installation

### Quick Install (Recommended)
```
# Install all dependencies
python install.py all

# Or install specific mode
python install.py opencv  # Fastest mode
python install.py fast    # Original fast mode
python install.py minimal # Minimal dependencies only
```

### Manual Installation
```
# Install core dependencies
pip install Pillow numpy

# For fast mode
pip install scipy

# For opencv mode (fastest)
pip install opencv-python

# For skimage mode
pip install scikit-image

# Or install everything at once
pip install Pillow numpy scipy opencv-python scikit-image
```
### Platform-Specific Notes
- Windows: If OpenCV installation fails, try opencv-python-headless
- Mac/Linux: Should work out of the box with pip
- Python Version: Python 3.8 or higher required

## Usage

### Basic Usage
```
# Single image processing
python UVIslandColorizer.py input.png output.png

# With specific mode
python UVIslandColorizer.py input.tga output.png --mode opencv
python UVIslandColorizer.py input.jpg output.png --mode fast
python UVIslandColorizer.py input.png output.png --mode complex
```
### Batch Processing
```
# Process entire folder
python UVIslandColorizer.py input_folder output_folder --batch --mode fast
```
### Command Line Options
```
Positional Arguments:
  input                 Input image file or directory
  output                Output image file or directory

Processing Modes:
  --mode {fast,complex,opencv,skimage}
						Processing mode (default: fast)

Alpha Channel Settings:
  --threshold FLOAT     Alpha threshold for images with alpha (0.0-1.0, default: 0.1)
  --solid-threshold FLOAT
						Brightness threshold for images without alpha (0.0-1.0, default: 0.9)
  --invert-mask         Invert mask for images without alpha (default: dark=transparent)

Border Processing:
  --border-scale FLOAT  Border scale (0.0-2.0, default: 0.1)
  --fade-alpha          Fade alpha at borders (default: sharp edges)

Color Settings:
  --blended-only        Use only blended colors (default: pure colors first)

Batch Processing:
  --batch               Batch process all images in input directory

Help:
  --help                Show help message and exit
```
### Examples
```
# TGA with alpha channel
python UVIslandColorizer.py "T_SM_BuddhaHead_MM_OpaqueSimple_Inst_C.TGA" result.png --mode opencv

# JPEG without alpha (automatic detection)
python UVIslandColorizer.py texture.jpg result.png --solid-threshold 0.8

# Inverted mask for light-on-dark textures
python UVIslandColorizer.py texture.jpg result.png --solid-threshold 0.8 --invert-mask

# Sharp borders (no fade)
python UVIslandColorizer.py input.png output.png --border-scale 0.1

# Faded borders
python UVIslandColorizer.py input.png output.png --border-scale 0.2 --fade-alpha

# No border expansion
python UVIslandColorizer.py input.png output.png --border-scale 0.0

# Pure colors only (best contrast)
python UVIslandColorizer.py input.png output.png --mode fast

# Blended colors only (when many islands)
python UVIslandColorizer.py input.png output.png --blended-only

# Complex mode with custom search range
python UVIslandColorizer.py input.png output.png --mode complex --border-scale 0.5
```
## How It Works
1. UV Island Detection
- Alpha-based: Uses alpha channel threshold for transparent background
- Brightness-based: For images without alpha, uses brightness threshold
- Connectivity: 8-connected component analysis to find islands
2. Color Generation
- Pure Colors: Generated in HSV space for maximum contrast
- Blended Colors: Created by mixing base colors when needed
- Contrast Optimization: Colors are spaced to maximize distinction
3. Border Processing
- Edge Detection: Finds boundaries between islands and background
- Border Expansion: Controlled by --border-scale parameter
- Alpha Fading: Optional smooth transitions at edges
4. Performance Optimization
- Vectorization: NumPy operations instead of Python loops
- Library Acceleration: Uses optimized libraries when available
- Memory Mapping: Efficient handling of large images

## Performance Comparison
|Mode		|Speed	|Memory	|Dependencies	|Best For						|
|-----------|-------|-------|---------------|-------------------------------|
|OpenCV		|⭐⭐⭐⭐⭐	|Low	|OpenCV			|Large images, batch processing	|
|Fast		|⭐⭐⭐⭐	|Medium	|SciPy			|Balanced performance			|
|Skimage	|⭐⭐⭐	|Medium	|scikit-image	|Advanced operations			|
|Complex	|⭐⭐		|High	|None			|Precision, small images		|

## Use Cases
1. Game Development
- UV map visualization for 3D models
- Texture atlas debugging
- Material assignment previews
2. 3D Modeling
- UV layout analysis
- Island identification
- Texture painting preparation
3. Research & Education
- Computer graphics teaching
- Image processing research
- Algorithm visualization

## Troubleshooting

### Common Issues
1. OpenCV fails to load TGA files
- Solution: Tool automatically falls back to PIL for TGA support
2. Memory issues with large images
- Solution: Use OpenCV mode, reduce border scale
3. Poor results with images without alpha
- Solution: Adjust --solid-threshold and try --invert-mask
4. Slow processing on large folders
- Solution: Use --mode opencv for batch processing

### Error Messages
- "SciPy not available": Install with pip install scipy or use --mode complex
- "OpenCV not available": Install with pip install opencv-python or use another mode
- "Image loading failed": Check file format and permissions

### Advanced Topics
1. Customizing Color Generation
- The tool uses HSV color space for optimal contrast. You can modify the generate_colors method in the source code to implement custom color schemes.
2. Extending for Specific Formats
- To add support for additional image formats, modify the load_image_safe function in UVIslandColorizer.py.
3. Integration with Pipelines
- The tool can be integrated into automated pipelines:
```
# Example pipeline integration
find . -name "*.tga" -exec python UVIslandColorizer.py {} {}.colored.png \;
```
## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Acknowledgments
- Inspired by HLSL shader code for UV island coloring
- Built with Pillow, NumPy, SciPy, OpenCV, and scikit-image
- Thanks to all contributors and testers

## Support
- Issues: Use the GitHub issue tracker
- Questions: Check the examples above or open an issue
- Feature Requests: Submit via GitHub issues