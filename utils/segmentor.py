"""Use to get tuned values for segmentation from a GUI.

This is easier than trying different values in code.
Reasonable values for my setup as of late March 2022 seem to be:

Yellow (target) balls:
    lower HSV: [ 15  70 170], upper HSV: [ 60 255 255]
Blue (distractor) balls:
    lower HSV: [ 70  70  70], upper HSV: [155 230 255]
Black ladle? Seems like V is all that matters! Adjusting lower ranges of all
other parts leads to near-immediate removal of the ladle.
    lower HSV: [  0   0   0], upper HSV: [255 255  45]
To get all points corresponding to the 'inner' region (the water, and all the
balls, EXCEPT the ladle), just increase the saturation value:
    lower HSV: [  0  70   0], upper HSV: [255 255 255]

As for the red water itself, it doesn't look like there's a sufficient value for
that, but we can get all points for the 'inner' region and remove anything that
is a ball or the ladle. I think HSV segmentation may  be sufficient for now.
"""
import cv2
import rospy
import argparse
import numpy as np
import matplotlib.pyplot as plt
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class Segmenter():
    """Based on code from Thomas Weng."""

    def __init__(self, args):
        self.args = args
        rospy.init_node("fg_bg_segmenter")
        self.sub = rospy.Subscriber(args.img_topic, Image, self.cb)
        self.use_bgr = (args.bgr == 1)
        if self.use_bgr:
            self.title_window = 'Fore-Back ground Color Segmentor'
        else:
            self.title_window = 'Fore-Back ground HSV Segmentor'

        # If HSV segmentation.
        self.lower_hue =   0
        self.upper_hue = 255
        self.lower_sat =   0
        self.upper_sat = 255
        self.lower_val =   0
        self.upper_val = 255

        # If BGR segmentation.
        self.lower_B =   0
        self.upper_B = 255
        self.lower_G =   0
        self.upper_G = 255
        self.lower_R =   0
        self.upper_R = 255

        self.bgr = None
        self.hsv = None
        self.dst = None
        self.bridge = CvBridge()
        cv2.namedWindow(self.title_window)

        if self.use_bgr:
            cv2.createTrackbar('lower B', self.title_window,   0, 255, self.on_lower_b)
            cv2.createTrackbar('upper B', self.title_window, 255, 255, self.on_upper_b)
            cv2.createTrackbar('lower G', self.title_window,   0, 255, self.on_lower_g)
            cv2.createTrackbar('upper G', self.title_window, 255, 255, self.on_upper_g)
            cv2.createTrackbar('lower R', self.title_window,   0, 255, self.on_lower_r)
            cv2.createTrackbar('upper R', self.title_window, 255, 255, self.on_upper_r)
        else:
            cv2.createTrackbar('lower hue', self.title_window,   0, 255, self.on_lower_h)
            cv2.createTrackbar('upper hue', self.title_window, 255, 255, self.on_upper_h)
            cv2.createTrackbar('lower sat', self.title_window,   0, 255, self.on_lower_s)
            cv2.createTrackbar('upper sat', self.title_window, 255, 255, self.on_upper_s)
            cv2.createTrackbar('lower val', self.title_window,   0, 255, self.on_lower_v)
            cv2.createTrackbar('upper val', self.title_window, 255, 255, self.on_upper_v)

    def update(self):
        # Keep pixel from 'bgr' image on (white) if the mask is 1 at the pixel.
        if self.use_bgr:
            lower = np.array([self.lower_B, self.lower_G, self.lower_R], dtype='uint8')
            upper = np.array([self.upper_B, self.upper_G, self.upper_R], dtype='uint8')
            mask = cv2.inRange(self.bgr, lower, upper)
            print("lower BGR: {}, upper BGR: {}".format(lower, upper))
        else:
            lower = np.array([self.lower_hue, self.lower_sat, self.lower_val], dtype='uint8')
            upper = np.array([self.upper_hue, self.upper_sat, self.upper_val], dtype='uint8')
            mask = cv2.inRange(self.hsv, lower, upper)
            print("lower HSV: {}, upper HSV: {}".format(lower, upper))
        # kernel = np.ones((9,9),np.uint8)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        self.dst = cv2.bitwise_and(np.stack([mask, mask, mask], axis=2), self.bgr)

    # ============================== BGR ============================== #

    def on_lower_b(self, val):
        self.lower_B = val
        self.update()

    def on_upper_b(self, val):
        self.upper_B = val
        self.update()

    def on_lower_g(self, val):
        self.lower_G = val
        self.update()

    def on_upper_g(self, val):
        self.upper_G = val
        self.update()

    def on_lower_r(self, val):
        self.lower_R = val
        self.update()

    def on_upper_r(self, val):
        self.upper_R = val
        self.update()

    # ============================== HSV ============================== #

    def on_lower_h(self, val):
        self.lower_hue = val
        self.update()

    def on_upper_h(self, val):
        self.upper_hue = val
        self.update()

    def on_lower_s(self, val):
        self.lower_sat = val
        self.update()

    def on_upper_s(self, val):
        self.upper_sat = val
        self.update()

    def on_lower_v(self, val):
        self.lower_val = val
        self.update()

    def on_upper_v(self, val):
        self.upper_val = val
        self.update()

    # ============================== Update window ============================== #

    def cb(self, msg):
        # In BGR mode. If we do `cv2.imwrite(..., im)` we get 'correct' colors.
        im = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Make sure this crop matches what we are actually using!
        x = 840
        y = 450
        w = 300
        h = 300
        im = im[y:y+h, x:x+w]

        # Both the BGR and HSV. Not sure why Thomas had BGR -> RGB here?
        #self.bgr = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        self.bgr = im.copy()
        self.hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    def spin(self):
        """Continually display the image.

        If self.bgr is in BGR mode (the default) then just play it and it will
        look correctly to our eyes.
        """
        while not rospy.is_shutdown():
            if self.bgr is not None:
                self.update()
            if self.dst is None:
                rospy.sleep(0.1)
            else:
                cv2.imshow(self.title_window, self.dst)
                #cv2.imshow(self.title_window,
                #           cv2.cvtColor(self.dst, cv2.COLOR_BGR2RGB))
                cv2.waitKey(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_topic', help='ROS topic to subscribe to for RGB image',
        default='k4a_top/rgb/image_rect_color')
    parser.add_argument('--bgr', type=int, help='1 if using BGR, else HSV', default=0)
    args, _ = parser.parse_known_args()

    s = Segmenter(args)
    s.spin()
