# Advanced Lane Finding Project
***Vinayak Kamath | 09 October 2019***

---

**Project Goals**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/test_image_undistort.jpg "Undistorted"
[image2]: ./camera_cal/test_image.jpg "Distorted"
[image3]: ./camera_cal/test_image_2.jpg "Example"
[image4]: ./output_images/test_image_2_undistort.jpg "Example Undistorted"
[image5]: ./output_images/undistorted5.jpg "Undistorted Example"
[image6]: ./output_images/threshold5.jpg "Threshold Example"
[image7]: ./output_images/warp5.jpg "Warp Example"
[image8]: ./output_images/polysearch5.jpg "Poly Search Example"
[image9]: ./output_images/final5.jpg "Final Example"
[image10]: ./output_images/windowsearch5.jpg "Window Search Example"
[image11]: ./Video_ERROR/errorframe.jpg "Video Error Frame"
[video1]: ./project_video_output.mp4 "Video Output"


## Camera Calibration
Initially, it is essential to import the necessary modules needed for various filtering and image processing steps further in the project. It also gives a good overview of the modules needed, if done in a consolidated fashion.

The Camera Calibration is essentially performed as per the intruction in the course exercise, using the cv2 Library function of ChessboardCorners. The distortion correction is subsequently performed after finding the transformation matrix between the object and image points. These matrix values are stored in a pickle file for further usage.

****Distorted Image****
![alt text][image2]

****Undistored Image****
![alt text][image1]


These distortion corrections were further tested on a home made image for cross-checking the extent of compatibility of the transformation matrix.

****Distorted Image****
![alt text][image3]

****Distortion Correction****
![alt text][image4]


## Color & Gradient Thresholding

The principle thresholding technique used (after many variations and iterations of threshold values) to generate a binary image were:

* S Channel Gradient from HLS color space || s_thresh=(80, 255)
* L Thresholding for light intensity filtering || l_thresh=(90,255)
* Sobel X-Direction gradient thresholding || sx_thresh=(10, 255)


****Corrected Image****
![alt text][image5]

****Thresholded Image****
![alt text][image6]

### Perspective Transformation

On achieving a robust thresholding mechanism, its time to transform the image into a "birds-eye" perspective. The choice for the source and destination points are as follows: 

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 570, 470      | 320, 1        | 
| 722, 470      | 920, 1        |
| 1110, 720     | 920, 720      |
| 220, 720      | 960, 720      |

****Perspective Transformed Image****
![alt text][image7]

## Lane Detection: Slide Window Search

This is where the heavy-math starts to show light. The window search is primarily sectioning of the lane images on detection and creating its progession based on:
* identified pixels, and recentering of the search Window in its progression.
* In case of absence of pixels, creating a progression based on the previous window coordinates.

To detect the start point of a search window, it is necessary to first identify the left and right lane sources (in this case: its concentrating primarily on the lower half of the transposed image, since the lane progression is safely assumed to proceed in the vertical direction). The source coordinates are determined on the histogram weight across the image (shape[0] axis). This ensures a good starting point for progressing the search windows, of which the size is determined based on:
* pre-defined window height based on no. of windows needed vertically to the overall size of the image.
* pre-defined size of 50 pixels as margins on either side of both left and right lanes.

The arrays of coordinates, both for left and right lanes are used as input parameters for the cv2.Polyfit function to obtain the coefficients of a second order polynomial.

![alt text][image10]

## Lane Detection: Identify lanes around previously detected lines ###

Instead of initiating a new search for lane line coordinates with progressive windows, it is more efficient to obtain the next line coordinates for either lanes.
The polynomial is threby fitted for the coordinates from the previous frame that represents the lane center. The margin are set to similar values as the old search window.

****Lane lines around previous lines****

![alt text][image8]

## Transposing the lanes to the original traffic image ###

The area of interest, which was previous warped for lane detection is reverse transposed to the original image, including highlighting the lanes. The inverse matrix is determined using the same source and destination image coordinates. The Numpy and cv2 libraries are quite useful for superimposing these entities on the traffic image.

![alt text][image9]

---

## Pipeline Video ##
### Sanity Checks and Lane averaging ###

Proceeding on to implementing the lane detection process used on traffic images, the algorithm is implemented with a traffic video.

It is crucial that the video frames are:
* correctly corrected for distortion
* thresholded using the principles mentioned above.
* warped and lanes detected based on detected pixels for either lanes
* Transposed back from the warped image to the original video frame

A sanity check is necessary to ensure that the frame images have lanes detected and that the polynomial fitted lanes are progessive to previously created lanes from former frames.

This is done by keeping a log of predicted lanes in an array, which is used for:
* Checking if predictions exist based on polyfitted lines on an array of past good lines. If not, a reset search with reinitialising the window search is initiated.
* A deviation exceeding 30% between the predictions and previously obtained lanes initiates a averaging and updating a running mean array between the two lane lines.
* In case of missing lane pixels on any subsequent frame, the averaging is create the missing lane with reference to the other detected lane. This is because of the high likelyhood, that the missing lane would also be in the same direction as the detected lane, just on the otherside.

## Summary
The comprehensive code description and various other test image files are found in the Jupyter Notebook, submitted with the project.

[Link to the Jupyter Notebook](./AdvancedLaneFinding_CodeDocumentation.ipynb)

Here's a [link to my video result](./project_video_output.mp4)

---

## Discussion
An issue faced during the project was certainly observing failure of lane detection on a warped frame during processing a video file. This is definitely possible in real-life driving conditions, considering road surface changes (colorations) and eroded lane lines which make detection with the static filtering threshold values not possible.

An obvious enhancement which could be included in having a adaptive parameter adjustments for the image thresholding based on varying environmental parameters (probably even set through a look up table) such as:
* Weather conditions through modern navigation systems
* Double checking for functional safety by cross verification of image recognised with navigational day of HD Maps.
* Compensation the light intensity filtering through measurement of light disturbances coming through external factors such as sunlight, traffic, shades due to overlaying objects, etc.

****Warped Image lane line detection failure****
![alt text][image11]

***Status submission: 10.10.2019***:

 **The harder Challenge Video would be dealt with at a later stage in time after the course delay has beeen duly compensated before final submission**
