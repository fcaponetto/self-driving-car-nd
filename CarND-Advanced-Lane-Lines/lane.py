import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from collections import deque

objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.


def chessboard_corners(fimage):
    img = cv2.imread(fimage)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (9, 6), corners, ret)
        return img


# Remove distortion from images
def camera_cal_and_undistort(image):
    img = cv2.imread(image)
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

def show_images_3(image1, image2, image3):
    if image1 is not None:
        image1 = cv2.imread(image1)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,6))
    ax1.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image', fontsize=18)
    # ax2.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    # ax2.set_title('With Corners', fontsize=18)
    ax2.imshow(cv2.cvtColor(image3, cv2.COLOR_BGR2RGB))
    ax2.set_title('Undistorted', fontsize=18)

def show_images_2(original, modified):
    if original is not None:
        original = cv2.imread(original)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,6))
    ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image', fontsize=20)
    ax2.imshow(cv2.cvtColor(modified, cv2.COLOR_BGR2RGB))
    ax2.set_title('Undistorted Image', fontsize=20)

images = glob.glob('camera_cal/calibration*.jpg')
for idx, fimage in enumerate(images):
    img_corners = chessboard_corners(fimage)
    undistorted = camera_cal_and_undistort(fimage)
    show_images_3(fimage, img_corners, undistorted)