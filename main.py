import numpy as np
import nanocamera as nano
import cv2
import time

# cams_test = 10
# for i in range(0, cams_test):
#     vid = cv2.VideoCapture(i)
#     test, frame = vid.read()
#     print("i : "+str(i)+" /// result: "+str(test))
#     while(True): 
        
#         # Capture the video frame 
#         # by frame 
#         ret, frame = vid.read() 
    
#         # Display the resulting frame 
#         cv2.imshow('frame', frame) 
        
#         # the 'q' button is set as the 
#         # quitting button you may use any 
#         # desired button of your choice 
#         if cv2.waitKey(1) & 0xFF == ord('q'): 
#             break
    
#     # After the loop release the cap object 
#     vid.release() 
#     # Destroy all the windows 
#     cv2.destroyAllWindows() 

vid = cv2.VideoCapture(0) 
  
while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
  
    # Display the resulting frame 
    cv2.imshow('frame', frame) 
    cv2.imwrite("test.jpg", frame)
    time.sleep(30)
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 
