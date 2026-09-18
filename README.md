# SmartFruit-Vision
Digital Image Processing project for fruit surface damage analysis



## Explainable Fruit Surface Damage Analysis Using Digital Image Processing

### Project Description

SmartFruit Vision is a Digital Image Processing project that analyzes visible surface damage in fruit images.

The project applies image restoration, enhancement, segmentation and morphological processing techniques to identify visible damaged or discolored regions.

### Objectives

- Remove simulated Gaussian noise.
- Compare Mean, Median and Gaussian filtering.
- Evaluate restoration using MSE, PSNR and SSIM.
- Enhance the image using Histogram Equalization and CLAHE.
- Segment the fruit from the background using GrabCut.
- Segment fruit surface regions using K-Means clustering.
- Identify visible damaged regions.
- Calculate visible defect percentage.
- Determine project-defined defect severity.

### Dataset

Dataset: Fruits Fresh and Rotten for Classification

Classes used:

- Fresh Apple
- Rotten Apple
- Fresh Banana
- Rotten Banana
- Fresh Orange
- Rotten Orange

The main analysis uses Fresh Apple 7 and Rotten Apple 20.

### Methodology

Input Image  
↓  
Gaussian Noise  
↓  
Image Filtering  
↓  
MSE / PSNR / SSIM  
↓  
Image Enhancement  
↓  
GrabCut Segmentation  
↓  
Fruit Extraction  
↓  
K-Means Segmentation  
↓  
Defect Mask  
↓  
Morphological Processing  
↓  
Defect Percentage  
↓  
Final Analysis

### Technologies

- Python
- OpenCV
- NumPy
- Matplotlib
- Pillow
- Scikit-image
- Scikit-learn
- Google Colab

### Project Output

The system produces:

- Restored images
- Enhanced images
- Fruit segmentation
- K-Means clusters
- Defect masks
- Highlighted defect regions
- Visible defect percentage
- Defect severity
- Defect distribution

### Limitations

The system analyses only visible surface regions. It does not determine internal fruit quality or food safety.

### Future Scope

- Larger dataset
- Better segmentation methods
- Deep learning-based defect detection
- Real-time fruit inspection
- Web or mobile application
