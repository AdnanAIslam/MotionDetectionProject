import imutils
import cv2
import numpy as np

# Parameters to prevent flase positives
# Minimum number of frames to filter before determining whether a change is present
FRAMES_TO_PERSIST = 10
# Size of the area for movement detection
MIN_SIZE_FOR_MOVEMENT = 2000
# Minimum length of time where no motion is detected
MOVEMENT_DETECTED_PERSISTENCE = 100

# Create capture object
cap = cv2.VideoCapture(1) 

# Frame variables
first_frame = None
next_frame = None

# Initializing display variables
font = cv2.FONT_HERSHEY_SIMPLEX
delay_counter = 0
movement_persistent_counter = 0

# Loop frame reads
while True:

    # Set transient motion detected as false
    transient_movement_flag = False
    
    # Read frame
    ret, frame = cap.read()
    text = "Unoccupied"

    # Error test
    if not ret:
        print("CAPTURE ERROR")
    
    # Set sizing of original and grey scale for outlines
    frame = imutils.resize(frame, width = 1500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Adding blur to reduce false positives
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    hsv = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
    # cv2.rectangle(frame,(600,350), (1250,550), (255,0,0), 2)

    # Initialize first frame at the start
    if first_frame is None:
        first_frame = gray   

    delay_counter += 1

    # Otherwise, set the first frame to compare as the previous frame
    # But only if the counter reaches the appriopriate value
    # The delay is to allow relatively slow motions to be counted as large
    # motions if they're spread out far enough
    if delay_counter > FRAMES_TO_PERSIST:
        delay_counter = 0
        first_frame = next_frame

        
    # Set the next frame to compare to initial
    next_frame = gray

    # Compare the two frames, find the difference
    frame_delta = cv2.absdiff(first_frame, next_frame)
    thresh = cv2.threshold(frame_delta , 25, 255, cv2.THRESH_BINARY)[1]

    # Find contours of the thesholds
    thresh = cv2.dilate(thresh[800:1000, 200:550], None, iterations = 2)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the contours
    for c in cnts:

        # Save the coordinates of all found contours
        (x, y, w, h) = cv2.boundingRect(c)
        
        # Assigning minimum size of movement to contours
        if cv2.contourArea(c) > MIN_SIZE_FOR_MOVEMENT:
            transient_movement_flag = True
            
            # Draw a rectangle at recorded coordinates
            cv2.rectangle(frame, (x+600, y+1250), (x + w, y + h), (100, 200, 0), 2)

    # Reset persistence
    if transient_movement_flag == True:
        movement_persistent_flag = True
        movement_persistent_counter = MOVEMENT_DETECTED_PERSISTENCE

    # As long as there was a recent transient movement, say a movement
    # was detected    
    if movement_persistent_counter > 0:
        text = "Movement Detected " + str(movement_persistent_counter)
        movement_persistent_counter -= 1
    else:
        text = "No Movement Detected"

    # Print the text on the screen, and display the raw and processed video 
    # feeds
    cv2.putText(frame, str(text), (10,35), font, 2.0, (255,255,255), 2, cv2.LINE_AA)
    
    # Convert the frame_delta to color for splicing
    frame_delta = cv2.cvtColor(frame_delta, cv2.COLOR_GRAY2BGR)
    
    # hsv = cv2.cvtColor(frame_delta, cv2.COLOR_BGR2HSV)
 
    # define range of blue color in HSV
    lower_white = np.array([0,0,0])
    upper_white = np.array([150,0,255])
    
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    # Splice the two video frames together to make one long horizontal one
    cv2.imshow("frame", np.hstack((frame_delta, frame, res)))


    # Interrupt trigger by pressing q to quit the open CV program
    ch = cv2.waitKey(1)
    if ch & 0xFF == ord('q'):
        break

# Cleanup when closed
cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()