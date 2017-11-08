import numpy as np
import cv2



""" $-> Calibrate and undistortion image """
def cal_undistort(img, objpoints, imgpoints):
    val, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, 
    imgpoints, img.shape[0:2], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


""" $-> corners_unwarp  """
def corners_unwarp(undist, nx, ny):
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    warped = []
    M=0
    if ret == True:
        cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
        offset = 10
        img_size = (gray.shape[1], gray.shape[0])
        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                     [img_size[0]-offset, img_size[1]-offset], 
                                     [offset, img_size[1]-offset]])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(undist, M, img_size)
    return warped, M



""" $-> unwarp  """
def unwarp(undist,left_bottom,right_bottom,left_top,right_top):
    #gray = cv2.cvtColor(undist, cv2.COLOR_RGB2GRAY)
    gray = undist
    warped = []
    M=0
    offset = 10
    img_size = (gray.shape[1], gray.shape[0])
    src = np.float32([left_top,right_top,right_bottom, left_bottom ])
    dst = np.float32([[offset, offset], 
                      [img_size[0]-offset, offset], 
                      [img_size[0]-offset, img_size[1]-offset], 
                      [offset, img_size[1]-offset]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(undist, M, img_size)
    
    return warped, M


""" $-> Absolute Sobel  """
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output

""" $-> Magnitude Sobel  """
def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

    # Return the binary image
    return binary_output

""" $-> Direction Threshold  """
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image

    return binary_output


""" $-> HSV Threshold  """
def hsv_threshold(img, sxbinary, thresh_H=(0, 255), thresh_S=(0, 255)):
    
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hsv[:,:,1]
    h_channel = hsv[:,:,0]
    
    s_binary = np.zeros_like(s_channel)
    h_binary = np.zeros_like(s_channel)
    comb = np.zeros_like(s_channel)
    
    s_binary[(s_channel >= thresh_S[0]) & (s_channel <= thresh_S[1])] = 1
    h_binary[(h_channel >= thresh_H[0]) & (h_channel <= thresh_H[1])] = 1
    comb[( s_binary == 1  ) | ( h_binary == 1  ) ] = 1 
         
    return comb


""" $-> Combine Thresholds  """
def comb_threshold(unImg,ksize, gradyTh, mag_binaryTh,dir_binaryTh, hsvTh_H, hsvTh_S ):

    # Apply each of the thresholding functions
    #gradx = abs_sobel_thresh(unImg, orient='x', sobel_kernel=ksize, thresh=(10, 55))
    grady = abs_sobel_thresh(unImg, orient='y', sobel_kernel=ksize, thresh=gradyTh)
    mag_binary = mag_thresh(unImg, sobel_kernel=ksize, thresh=mag_binaryTh)
    dir_binary = dir_threshold(unImg, sobel_kernel=ksize, thresh=dir_binaryTh )
    
    combined = np.zeros_like(dir_binary)
    #combined[( (mag_binary == 1)) | ((  grady== 1) & (dir_binary == 1))] = 1
    combined[( (mag_binary == 1)) | ((  grady== 1) & (dir_binary == 1))] = 1
   
    ImThres = hsv_threshold(unImg, combined, thresh_H=hsvTh_H , thresh_S=hsvTh_S)
    
    comb = np.zeros_like(dir_binary)
    comb[( combined == 1  ) | ( ImThres == 1  ) ] = 1 
    
    return comb


""" $->Fit Lane """
def fit_lane(Per, lineLeft, lineRight):
    histogram = np.sum(Per[Per.shape[0]/2:,:], axis=0)
    out_img = np.dstack((Per, Per, Per))*1
    
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    nwindows = 10
    window_height = np.int(Per.shape[0]/nwindows)
    
    nonzero = Per.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []
    
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = Per.shape[0] - (window+1)*window_height
        win_y_high = Per.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,1,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,1,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    ploty = np.linspace(0, Per.shape[0]-1, Per.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [1, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 1]
    
    y_eval = np.max(ploty)
    
    ym_per_pix = 30/ out_img.shape[0] # meters per pixel in y dimension
    #xm_per_pix = 3.7/ out_img.shape[1] # meters per pixel in x dimension
    xm_per_pix = 3.7/900
       
    # Fit new polynomials to x,y in world space    
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    
    leftRadiusList = []
    rightRadiusList = []
    for y in range(0,int(y_eval),2):
        leftRadiusList.append (((1 + (2*left_fit_cr[0]*y*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0]) )
        rightRadiusList.append( ((1 + (2*right_fit_cr[0]*y*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0]) )
    
    leftRadius = np.mean(np.array( leftRadiusList )) 
    rightRadius = np.mean(np.array( rightRadiusList ))      
    

    #### Lane Sanity Check and popullate Line properties  
    # Check Curvature Similarity         
    SimilarCurvature = abs((leftRadius-rightRadius)/(leftRadius+rightRadius)) <= 0.5
    # Check Lane Wide 
    LaneWide =  ((rightx_current - leftx_current)*xm_per_pix ) >= 3.0     
    # Check if lines are parallels
    yTop = 50           
    WideTop = (right_fitx[yTop] - left_fitx[yTop])*xm_per_pix   
    yBot = 700            
    WideBot = (right_fitx[yBot] - left_fitx[yBot])*xm_per_pix  
    pararell = abs(WideTop - WideBot) < 0.9

    laneSanityCheck =pararell*5 + LaneWide*3 + SimilarCurvature*2
                                                                  
    # Popullate Lines                 
    lineLeft.detected  = True
    lineRight.detected = True
    lineLeft.radius_of_curvature  = leftRadius*0.001 # Convertion to Km
    lineRight.radius_of_curvature = rightRadius*0.001 # Cnvertion to Km
    lineLeft.lane_inds_size  = len(left_lane_inds)
    lineRight.lane_inds_size = len(right_lane_inds)
    
    
    # If Pass lane sanity check append line tu cue
    if laneSanityCheck>7:
        # x values of the last n fits of the line        
        lineLeft.recent_xfitted.append(left_fitx)
        lineRight.recent_xfitted.append(right_fitx)
        
    # average x values of the fitted line over the last n iterations
    lineLeft.bestx  = np.mean(np.array(lineLeft.recent_xfitted),axis=0)
    lineRight.bestx = np.mean(np.array(lineRight.recent_xfitted),axis=0) 
    
    # polynomial coefficients for the most recent fit
    lineLeft.current_fit.append(left_fit)
    lineRight.current_fit.append(right_fit)
    # polynomial coefficients averaged over the last n iterations
    lineLeft.best_fit  = left_fit
    lineRight.best_fit = right_fit
    # distance in meters of vehicle center from the line
    lineLeft.line_base_pos  = (midpoint - leftx_current)*xm_per_pix 
    lineRight.line_base_pos = (rightx_current - midpoint)*xm_per_pix
       
    if len(lineLeft.recent_xfitted)  >= 10:
        del lineLeft.recent_xfitted[0]
        del lineRight.recent_xfitted[0]
    if len(lineLeft.recent_xfitted)  == 0:
        lineLeft.current_fit = left_fitx
        lineRight.current_fit = right_fitx
        
        
    return  ploty, out_img, lineLeft,lineRight


def invertWrap(Per,img,left_fitx, right_fitx, ploty,M):
    

    warp_zero = np.zeros_like(Per).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)

    Minv = cv2.invert(M)
    newwarp = cv2.warpPerspective(color_warp,Minv[1], (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    
    return  result

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        #indices array size
        self.lane_inds_size = None
        






    





