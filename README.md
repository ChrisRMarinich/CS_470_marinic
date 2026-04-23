# CS 470: Computer Vision and Image Processing
***Spring 2026***  
***Author: Christopher Marinich***  
***Original Author: Dr. Michael J. Reale***  
***SUNY Polytechnic Institute*** 
***testing the password***
## Runnable Python Scripts

### BasicVision.py
A basic sample that loads up the relevant libraries, prints versions numbers, and either 1) loads an image from a path specified on the command line, or 2) opens a webcam.
Image(s) will be displayed until a key is hit.

### A02.py,
This program performs grayscale image convolution using loop based, NumPy vectorized, and Fourier
 transform methods to compare correctness and efficiency. It reads kernels from text files, 
 converts data to float64, and applies zero padding to maintain output size. The loop method 
 follows the mathematical definition, the fast method uses NumPy broadcasting, and the Fourier 
 method uses FFTs for efficiency. Results can be scaled and converted to uint8 with OpenCV. A 
 Gradio interface allows users to load an image and kernel, choose a method, adjust parameters, 
and view the output.

### A03.py,
My white blood cell detector uses color segmentation to find likely cell regions. The image is 
first divided into superpixels, and the average color of each region is computed. These colors 
are then grouped with k means clustering, and the group most similar to blue is selected as the 
most likely white blood cell region. A binary mask is created from that group, cleaned with 
morphological operations, and separated into connected components. Bounding boxes are placed 
around valid components after filtering by size and shape, and overlapping duplicate detections 
are removed with non maximum suppression.




