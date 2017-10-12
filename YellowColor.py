# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 11:58:47 2017

@author: fahad
"""

import cv2
import numpy as np
from collections import deque
import time
import math




cap = cv2.VideoCapture(0) # capturing the video from Cam
fourcc =cv2.VideoWriter_fourcc(*'XVID') # Codec for the video
out = cv2.VideoWriter('Output.avi', fourcc, 20.0, (640,480)) # making video

pts = deque(maxlen=None) ## Unbound Buffer of datatype Deque
(dX, dY) = (0, 0)
counter = 0
direction = ''

while True:
    ret, frame = cap.read()
    start = time.clock()
    frame = cv2.resize(frame,(640,480)) ## resizing the frame pixel grid
    if (frame is None): ## Break the loop if there is no frame from the Cam
        break
        

    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # converting frame from BGR to
    ## HSV format. HSV: hue sat value
    lower_yellow = np.array([20,100,100]) # range of yellow in BGR color space
    upper_yellow = np.array([30,255,255]) # Find it on internet
#    lower_blue = np.array([110,100,100])
#    upper_blue = np.array([130,255,255]) 

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    #mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame, frame, mask = mask) # Performing bitwise and-
    ## operation between frame and mask.
    median = cv2.medianBlur(mask,15) # Image filtering/Noise removing
    
    
    
    
    #cv2.imshow('median',median)
    #cv2.imshow('mask',mask)
    #cv2.imshow('result',res)
    
    ############################################################################# 
    ## Finding contours of the object and its tracking
    
    _, cnts, _ = cv2.findContours(median.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centerObject = None
    ## Only proceed if at least one contour was found
    if len(cnts)>0:
        ## find the largest contour in the mask, then use it to compute the
        ## minimum enclosing circle and centroid
        c = max(cnts, key = cv2.contourArea)
        ((x,y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        ## Finding X and Y of the centroid of contour c
        centerObject = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
        print('Object Position: ', centerObject)
        
        ## Only proceed if the radius meets a minimum size
        if radius>10: ## atleast 50 pixel radius to track it
            ## draw the circle and centroid on the frame, then update the
            ## list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius), (50, 55, 150),2)
            cv2.circle(frame, centerObject, 5, (0, 0, 255), -1)
            pts.appendleft(centerObject)
    
    ## loop over the set of tracked points
    for i in np.arange(1, len(pts)):
        ## if either of the tracked points are none, ignore them
        if pts[i-1] is None or pts[i] is None:
            continue
        ## check to see if enough points have been accumulated in the buffer
        if counter>=10 and i==10 and pts[i-10] is not None:
            ## compute the difference between X- and Y-coordinates, and 
            ## re-initialize the direction text variables
            dX = pts[i-10][0] - pts[i][0]
            dY = pts[i-10][1] - pts[i][1]
            (dirX, dirY) = ('', '')
            
            ## ensure there is enough movement in the x-direction 
            if np.abs(dX)>10:
                dirX = 'East' if np.sign(dX) == 1 else 'West'
            ## ensure there is enough movement in the y-direction
            if np.abs(dY)>10:
                dirY = 'North' if np.sign(dY) == 1 else 'South'
            
            ## Handle when both directions are non-empty
            if dirX != '' and dirY != '':
                direction = '{}-{}'.format(dirY, dirX)
            ## otherwise, only one direction is non-empty    
            else:
                direction = dirX if dirX != '' else dirY
                
        ## draw the connecting lines
        cv2.line(frame, pts[i-1], pts[i], (0, 0, 255), thickness = 1)
        
    ## Show the direction of the movement and movement deltas on the frame 
    cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,(0, 0, 255), 3)
    cv2.putText(frame, 'dx: {}, dy: {}'.format(dX, dY),(10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    IterationTime = time.clock() - start
    #print('Time of loop', IterationTime)
    speed = math.sqrt(dX**2+dY**2)/IterationTime
    print('Speed (Pixels/secs) :',speed,'\n')
    
    ## Show the frame on the screen and increment the frame counter
    cv2.imshow('MyFrame',frame)
    counter += 1
     
    
    out.write(frame) ## Recording the video
    ##########################################################
    
    
    
    
       
    key = cv2.waitKey(1) & 0xFF # Wait for 1 ms to get the key from user
    if key == ord('q'): # break the loop when pressed q
       break


cap.release() # release the camera
out.release()
cv2.destroyAllWindows()