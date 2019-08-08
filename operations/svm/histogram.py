# More imports
import cv2
import imutils

class Histograms:    
    
    @staticmethod
    def extract_color_histogram(image, bins=(8,8,8)):
        # extract a 3D color histogram from the HSV color space using
        # the supplied number of 'bins' per channel
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0,1,2], None, bins, [0,180,0,256,0,256])
    
        # handle normalizing the histogram if we are using OpenCV 2.4.X
        if imutils.is_cv2():
            hist = cv2.normalize(hist)
    
        # otherwise, perform "in place" normaliation in OpenCV 3
        else:
            cv2.normalize(hist, hist)
    
        return hist.flatten()
    
    def extract_blue_histogram(image):
        # extract blue histogram from the image
        hist = cv2.calcHist([image], [0], None, [265], [0,256])
    
        # handle normalizing the histogram if we are using OpenCV 2.4.X
        if imutils.is_cv2():
            hist = cv2.normalize(hist)
    
        # otherwise, perform "in place" normaliation in OpenCV 3
        else:
            cv2.normalize(hist, hist)
    
        return hist.flatten()
    
    def extract_green_histogram(image):
        # extract green histogram from the image
        hist = cv2.calcHist([image], [1], None, [265], [0,256])
    
        # handle normalizing the histogram if we are using OpenCV 2.4.X
        if imutils.is_cv2():
            hist = cv2.normalize(hist)
    
        # otherwise, perform "in place" normaliation in OpenCV 3
        else:
            cv2.normalize(hist, hist)
    
        return hist.flatten()
    
    def extract_red_histogram(image):
        # extract blue histogram from the image
        hist = cv2.calcHist([image], [2], None, [265], [0,256])
    
        # handle normalizing the histogram if we are using OpenCV 2.4.X
        if imutils.is_cv2():
            hist = cv2.normalize(hist)
    
        # otherwise, perform "in place" normaliation in OpenCV 3
        else:
            cv2.normalize(hist, hist)
    
        return hist.flatten()