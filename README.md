# Aruco-marker-detection-decoding-and-tracking
ENPM673 Perception for Autonomous Robots

## Detection of AR Tag

### Corner Detection
- The Image is forst converted to grayscale, then compute the fourier transform of the grayscale image.
- Define a gaussian mask and blur the FFT output image, after applying mask do an inverse fourier transform on the blurred image.
- Convert the resultant image to binary and then apply a circular mask, then do FFT x resultant image and perform inverse FFT on the result.

![Image processing](https://user-images.githubusercontent.com/106445479/197946607-f842a946-ab03-4b3d-8458-12dc24b4c057.jpeg)
