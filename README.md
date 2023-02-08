# Aruco-marker-detection-decoding-and-tracking

## Objective

This project will focus on detecting a custom AR Tag (a form of **fiducial marker**), that is used for obtaining a point of reference in the real world, such as in augmented reality applications. There are two aspects to using an AR Tag, namely **Detection and Tracking**, both of which will be implemented in this project. The detection stage will involve finding the AR Tag from a given image sequence while the tracking stage will involve keeping the tag in“view” throughout the sequence and performing image processing operations based on the tag's orientation and position (a.k.a. the pose).

## Contents

```
├───LICENSE
├───Report.pdf
├───README.md
├───AR_Tag_detection.py
├───1tagvideo.mp4
├───testudo.png
└───outputs

```

## Requirements

- OpenCV `pip install opencv-python`
- NumPy `pip install numpy`
- glob `pip install glob2`
- Matplotlib `pip install matplotlib`
- SciPy `python -m pip install scipy`
- math
- Download and install Anaconda {easy}

## Instructions to run

1. Clone the repository

   ```
   git clone https://github.com/bharadwaj-chukkala/Aruco-marker-detection-decoding-and-tracking.git
   ```
2. Open the folder in the IDE
3. Run the AR_Tag_detection.py file

   ```
   cd <root>
   python AR_Tag_detection.py
   ```
4. Uncomment the commented lines at the end to save outputs to outputs folder

## Implementation Steps and Results

### Aruco Marker/ AR Tag Detection and Tracking

#### Image Preprocessing

<p align="center">
  <img width="500" height="300" src="https://github.com/bharadwaj-chukkala/Aruco-marker-detection-decoding-and-tracking/blob/main/outputs/Image%20processing.jpeg">
</p>
<p align="center">Fig 1.</p>


#### Edge and Corner Detection

<p align="center">
  <img width="500" height="300" src="https://github.com/bharadwaj-chukkala/Aruco-marker-detection-decoding-and-tracking/blob/main/outputs/corner%20detection.jpeg">
</p>
<p align="center">Fig 2.</p>

#### Decoding the AR Tag through Warping

<p align="center">
  <img width="500" height="300" src="https://github.com/bharadwaj-chukkala/Aruco-marker-detection-decoding-and-tracking/blob/main/outputs/inverse%20warping.png">
</p>
<p align="center">Fig 3.</p>

### Tracking the detected Tag

#### Superimposing the Testudo Image onto the TAG

<p align="center">
  <img width="500" height="300" src="https://github.com/bharadwaj-chukkala/Aruco-marker-detection-decoding-and-tracking/blob/main/outputs/testudo%20on%20AR%20tag.png">
</p>
<p align="center">Fig 4.</p>

#### Projecting a Cube on the Edges of the Image

<p align="center">
  <img width="500" height="300" src="https://github.com/bharadwaj-chukkala/Aruco-marker-detection-decoding-and-tracking/blob/main/outputs/Cube%20projection.jpeg">
</p>
<p align="center">Fig 5.</p>

### [Implementation Video](https://github.com/bharadwaj-chukkala/Aruco-marker-detection-decoding-and-tracking/blob/main/outputs/Testudo%20Superimposition%20and%20Cube%20Projection.mp4)

## References

- OpenCV documentation: https://docs.opencv.org/
- NumPy documentation: https://numpy.org/doc/stable/
- Matplotlib documentation: https://matplotlib.org/stable/index.html
- SciPy documentation: https://docs.scipy.org/doc/scipy/

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author Contact

**Bharadwaj Chukkala** <br>
UID: 118341705 <br>
Bharadwaj Chukkala is currently a Master's student in Robotics at the University of Maryland, College Park, MD (Batch of 2023). His interests include Machine Learning, Perception and Path Planning for Autonomous Robots. <br>
[![Contact](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](bchukkal@umd.edu)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/bharadwaj-chukkala/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/bharadwaj-chukkala)
