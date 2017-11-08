import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import glob



def calibrate(images):
    objpoints = []
    imgpoints = []
    row = 6
    col = 9
    objp = np.zeros((row * col, 3),np.float32)
    objp[:,:2] = np.mgrid[0:col,0:row].T.reshape(-1,2)
    
    for frame in images:
        img = cv2.imread(frame)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (col, row), None)
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)     
        cv2.drawChessboardCorners(img, (col, row), corners, ret)
        plt.imshow(img)
    return imgpoints, objpoints


"""
Calibrate Image. Obatain points
"""

images = glob.glob('camera_cal/calibration*.jpg')

imgpoints, objpoints = calibrate(images)

pickle.dump( imgpoints, open( "imgpoints.p", "wb" ) )
pickle.dump( objpoints, open( "objpoints.p", "wb" ) )

