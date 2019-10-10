# Advanced Lane Finding Project
***Vinayak Kamath | 09.Sept.2019***

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
[video1]: ./project_video_output.mp4 "Video Output"


### Camera Calibration
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


### Color & Gradient Thresholding

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

***Perspective Transformed Image
![alt text][image7]

### Lane Detection: Slide Window Search

This is where the heavy-math starts to show light. The window search is primarily sectioning of the lane images on detection and creating its progession based on:
* identified pixels, and recentering of the search Window in its progression.
* In case of absence of pixels, creating a progression based on the previous window coordinates.

To detect the start point of a search window, it is necessary to first identify the left and right lane sources (in this case: its concentrating primarily on the lower half of the transposed image, since the lane progression is safely assumed to proceed in the vertical direction). The source coordinates are determined on the histogram weight across the image (shape[0] axis). This ensures a good starting point for progressing the search windows, of which the size is determined based on:
* pre-defined window height based on no. of windows needed vertically to the overall size of the image.
* pre-defined size of 50 pixels as margins on either side of both left and right lanes.

![alt text][image10]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
