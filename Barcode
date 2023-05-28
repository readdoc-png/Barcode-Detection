import  sys
import  cv2
from  PyQt5 import  QtCore, QtWidgets
from  PyQt5.QtCore import *
from  PyQt5.QtGui import *
from  PyQt5.QtWidgets import *
from GUI import Ui_MainWindow
import numpy as np
import math
from matplotlib import pyplot as plt

class ShowImage(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        self.setupUi(self)  # Set up the UI elements
        # Rest of your code...
        self.cap = cv2.VideoCapture(0)

        self.actionGrayscale.triggered.connect(self.gray)
        self.actionCalculate_Gradient.triggered.connect(self.gradient)
        self.actionBlur_Image.triggered.connect(self.blur)
        self.actionBinary_Tresholding_2.triggered.connect(self.binarythresh)
        self.actionMorphology_3.triggered.connect(self.morphology)
        self.actionErode_and_Dilate.triggered.connect(self.erode_dilate)
        self.actionDetect_Barcode.triggered.connect(self.find_barcode)


    def gray(self):
        while True:
            _, frame = self.cap.read()  # Read camera frame

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.imshow('Grayscale', gray)  # Display grayscale frame

            key = cv2.waitKey(1)  # Prevent window from closing
            if key == 27:  # Press ESC to stop
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def gradient(self):
        while True:
            _, frame = self.cap.read()  # Read camera frame

            # Grayscale conversion
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate gradient using Sobel kernel
            gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

            # Combine x and y gradients to get magnitude and angle
            gradient = cv2.subtract(gradient_x, gradient_y)
            gradient = cv2.convertScaleAbs(gradient)

            # Display gradient magnitude
            cv2.imshow('Gradient', gradient)

            key = cv2.waitKey(1)  # Prevent window from closing
            if key == 27:  # Press ESC to stop
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def blur(self):
        while True:
            _, frame = self.cap.read()  # Read camera frame

            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply blurring to the grayscale image
            blurred = cv2.blur(gray, (3, 3))

            # Display the blurred frame
            cv2.imshow('Blurred', blurred)

            key = cv2.waitKey(1)  # Prevent window from closing
            if key == 27:  # Press ESC to stop
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def binarythresh(self):
        while True:
            _, frame = self.cap.read()  # Read camera frame

            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply binary thresholding with intensity threshold of 225
            _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)

            # Display the binary thresholded frame
            cv2.imshow('Binary Threshold', binary)

            key = cv2.waitKey(1)  # Prevent window from closing
            if key == 27:  # Press ESC to stop
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def morphology(self):
        while True:
            _, frame = self.cap.read()  # Read camera frame

            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply binary thresholding with intensity threshold of 225
            _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)

            # Create a 21x7 structuring element for morphology
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))

            # Apply morphology operation
            result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # Display the morphology result
            cv2.imshow('Morphology', result)

            key = cv2.waitKey(1)  # Prevent window from closing
            if key == 27:  # Press ESC to stop
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def erode_dilate(self):
        while True:
            _, frame = self.cap.read()  # Read camera frame

            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply binary thresholding with intensity threshold of 225
            _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)

            # Create a 21x7 structuring element for morphology
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))

            # Apply erosion and dilation operations
            eroded = cv2.erode(binary, kernel, iterations=4)
            dilated = cv2.dilate(eroded, kernel, iterations=4)

            # Display the eroded and dilated result
            cv2.imshow('Erosion and Dilation', dilated)

            key = cv2.waitKey(1)  # Prevent window from closing
            if key == 27:  # Press ESC to stop
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def find_barcode(self):
        while True:
            # Read frame from the camera
            ret, frame = self.cap.read()
            if not ret:
                break

            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate x and y gradient
            gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
            gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

            # Subtract the y-gradient from the x-gradient
            gradient = cv2.subtract(gradX, gradY)
            gradient = cv2.convertScaleAbs(gradient)

            # Blur the image
            blurred = cv2.blur(gradient, (3, 3))

            # Threshold the image
            _, thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

            # Perform morphology operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            # Erosion and dilation
            for i in range(4):
                closed = cv2.erode(closed, None, iterations=4)
                closed = cv2.dilate(closed, None, iterations=4)

            # Find contours in the thresholded image
            cnts, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(cnts) > 0:
                c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

                # Compute the rotated bounding box of the largest contour
                rect = cv2.minAreaRect(c)
                box = np.int0(cv2.boxPoints(rect))

                # Draw a bounding box around the detected barcode
                cv2.drawContours(frame, [box], -1, (0, 255, 0), 3)

            # Display the frame with the bounding box
            cv2.imshow("Camera", frame)

            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        # Release the video capture object and close all windows
        self.cap.release()
        cv2.destroyAllWindows()

app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('Project')
window.show()
sys.exit(app.exec_())
