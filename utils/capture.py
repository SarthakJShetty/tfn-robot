"""Adapted from Sarthak's code for PeelingBot.

For the Azure Kinect, the image size defaults to: (720, 1280, 3).
"""
import os
import cv2
from datetime import datetime

# Creating the video stream here to access the kinects. THE INDEX MAY VARY.
cidx = 0
kinect = cv2.VideoCapture(cidx)
print(f'Created VideoCapture idx {cidx}.')

while True:
    # Reading the actual images from the videostream
    _, kinect_img = kinect.read()

    # We view the images to make sure they are capturing what we want
    cv2.imshow('Kinect Mixed Media', kinect_img)

    # Look at the key returned by waitKey (which will be the ASCII value of 
    # the key pressed by the user), binary & with 0xFF ensures that even if
    # num lock is pressed the same binary value is returned, and we check 
    # that with the binary value of the 'q' or 's' keys to carry out 
    # quitting or saving operations respectively. NOTE(daniel) do cv2.waitKey(1)
    # ONCE in a given for loop, then compare with target strings.
    current_key = (cv2.waitKey(1) & 0xFF) 

    if current_key == ord('q'):
        print('Quitting')
        kinect.release()
        cv2.destroyAllWindows()
        break
    elif current_key == ord('s'):
        # When to save the stream -> when the 's' button is pressed
        # Time-stamping the files up to seconds.
        imgstr = (f'ak_{datetime.today().year}_{datetime.today().month}_'
                f'{datetime.today().day}_HMS_{datetime.today().hour}_'
                f'{datetime.today().minute}_{datetime.today().second}.png')
        imgstr = os.path.join('images', imgstr)
        print(f'Saving: {imgstr}, size: {kinect_img.shape}')
        cv2.imwrite(imgstr, kinect_img)
