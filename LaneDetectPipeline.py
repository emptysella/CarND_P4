import cv2
import matplotlib.pyplot as plt
import pickle
import glob
import cameraHelper as cam
import numpy as np
import math
from moviepy.editor import VideoFileClip
from moviepy.editor import ImageSequenceClip




# ///////////////////////////////////////////////// Load Calibration parameters

imgpoints = pickle.load( open( "imgpoints.p", "rb" ) )
objpoints = pickle.load( open( "objpoints.p", "rb" ) )
img = cv2.cvtColor(cv2.imread('test_images/test1.jpg'), cv2.COLOR_BGR2RGB)


lineLeft = cam.Line()
lineRight = cam.Line()


clip = VideoFileClip('project_video.mp4')
count = 0
for frame in clip.iter_frames():
    
    C_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # /////////////////////////////////////////////: Calibrate and Undistortion
    unImg = cam.cal_undistort(C_frame, objpoints, imgpoints)
    #unImg = C_frame
    # /////////////////////////////////////////////:    Compute Threshold

    ksize = 15
    gradyTh         = (80, 255)
    mag_binaryTh    = (80, 250)
    dir_binaryTh    = (0.6, np.pi/3)
    hsvTh_H           = (50, 100)
    hsvTh_S           = (230, 255)
    
    comb = cam.comb_threshold(unImg, ksize, gradyTh, mag_binaryTh, dir_binaryTh, hsvTh_H,hsvTh_S)
    

    # Compute Perspective
    left_bottom     = [202*0.12, 720]
    right_bottom    = [1280-(150*0.10), 720]
    left_top        = [560*0.95, 470]
    right_top       = [725*1.035, 470]   
    
    Per, M = cam.unwarp(comb,left_bottom,right_bottom,left_top,right_top)
    
    # /////////////////////////////////////////////:    Finding Peacks
    
    #if not math.fmod(count,10):

    ploty, IOut, lineLeft,lineRight  =  cam.fit_lane(Per,lineLeft, lineRight)

    # /////////////////////////////////////////////:    Invert Wrap. Mark Lane   
       
    ImOut = cam.invertWrap(Per, C_frame, lineLeft.bestx, lineRight.bestx, ploty, M)
               
    # /////////////////////////////////////////////:    Display info
    
    # Udpdate info every 15 frames to allow a smooth visualization
    if not math.fmod(count,3):
        CurvatureRadius_Km  = lineLeft.radius_of_curvature if lineLeft.lane_inds_size >= lineRight.lane_inds_size else lineLeft.radius_of_curvature   
        LeftLineOffset      = lineLeft.line_base_pos
        RightLineOffset     = lineRight.line_base_pos
        LRC                 = lineLeft.radius_of_curvature
        RRC                 = lineRight.radius_of_curvature
    

    text = 'Curv. Radius L : ' + '{:03.2f}'.format(LRC) + ' Km'
    cv2.putText(ImOut,text,(10,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA)
    
    text = 'Curv. Radius R : ' + '{:03.2f}'.format(RRC) + ' Km'
    cv2.putText(ImOut,text,(800,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA)
    
    text = 'Left Offset : ' + '{:03.2f}'.format(LeftLineOffset) + ' m'
    cv2.putText(ImOut,text,(10,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA)
    text = 'Right Offset : ' + '{:03.2f}'.format(RightLineOffset) + ' m'
    cv2.putText(ImOut,text,(800,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA)

    text = 'Curvature Radius : ' + '{:03.2f}'.format(CurvatureRadius_Km) + ' Km'
    cv2.putText(ImOut,text,(450,650), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2,cv2.LINE_AA)
    
    pathSave = './temp/' + '{:04d}'.format(count) + 'frames' + '.png'
    cv2.imwrite(pathSave,ImOut)
    
    pathSaveB = './temp/' + '{:03d}'.format(count) + 'framesB' + '.png'
    b=cv2.normalize(Per,None,0,255,cv2.NORM_MINMAX)
    cv2.imwrite('Per.png',b)

    
    print(('Frame : ' + str(count)))
    count += 1
    

#Create Video
temp = glob.glob('temp/*.png')
clip = ImageSequenceClip(temp, fps=24)
clip.write_videofile('detectedProject.mp4')






