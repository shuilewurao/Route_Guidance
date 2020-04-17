import cv2
import numpy as np
 
# Create a VideoCapture object
cap = cv2.VideoCapture('/Users/xuanchen/Desktop/edited1.mov')
 
# Check if camera opened successfully
if (cap.isOpened() == False): 
  print("Unable to read camera feed")
 
# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(720)
frame_height = int(1280)
 
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('/Users/xuanchen/Desktop/edited1_cropped.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 10, (550,1080))
 
while(True):
  ret, frame = cap.read()
 
  if ret == True: 
    print(frame.shape)
    # Write the frame into the file 'output.avi'
    frame = frame[0:1080, 50:600, :]
    out.write(frame)
    # Display the resulting frame    
    cv2.imshow('frame',frame)
 
    # Press Q on keyboard to stop recording
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else:
    break 
 
# When everything done, release the video capture and video write objects
cap.release()
out.release()
 
# Closes all the frames
cv2.destroyAllWindows() 