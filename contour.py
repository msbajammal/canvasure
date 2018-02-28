import numpy as np

import matplotlib.pyplot as plt

from scipy.ndimage import distance_transform_edt
from skimage.color import rgb2gray
from skimage.filters import scharr
from skimage.morphology import label, remove_small_holes, remove_small_objects
import thinning

from skimage.filters import threshold_otsu
from skimage.transform import hough_circle, hough_circle_peaks

import copy as obj_copy

from geometry import Point, Line, Circle

from LinesDetector import LinesDetector

class Contour:
    
    def __init__(self, image):
        if image.ndim == 3:
            image = rgb2gray(image)    
            schr = scharr(image)
            otsu = threshold_otsu(schr)
            contourImage = schr >= otsu*0.33
            
            contourImage = remove_small_holes(contourImage, min_size=10)
            contourImage = thinning.guo_hall_thinning(contourImage.astype('uint8'))
            self.image = contourImage
            self.removeSmallObjects()
            
        elif image.ndim == 2:
            contourImage = distance_transform_edt(image)
            contourImage[contourImage != 1] = 0
            self.image = contourImage
            self.removeSmallObjects()
        
    
    def detectCircles(self):
        max_radius = int(max(self.image.shape[0], self.image.shape[1])/2)
        test_radii = range(4, max_radius, 1)
        
        hspace = hough_circle(self.image, test_radii)
        h_max = np.max(hspace)
    
        accum, cx, cy, radii = hough_circle_peaks(hspace, test_radii,
                                                  min_xdistance=10,
                                                  min_ydistance=10,
                                                  threshold=0.75*h_max)
        circles = []
        
        # verify detected circles
        for i in range(len(radii)):
            circle_i = Circle(Point(cy[i], cx[i]), radii[i])
            pixels = circle_i.getFromImage(self.image)
            prob = np.count_nonzero(pixels) / len(pixels)
            
            if prob >= 0.50:
                circles.append(circle_i)
            
        return circles
    
    
    def removeCircles(self, circles=None):
        if circles == None:
            circles = self.detectCircles()
        
        for i in range(len(circles)):
            center = circles[i].center
            radius = circles[i].radius
            Circle(center, radius).drawOnImage(self.image, 0)
            Circle(center, radius - 1).drawOnImage(self.image, 0)
            Circle(center, radius + 1).drawOnImage(self.image, 0)
        
        self.removeSmallObjects()
        return self
        
    
    def getImage(self):
        return self.image
    
    
    def removeSmallObjects(self):
        labled = label(self.image, connectivity=2)
        labeled = remove_small_objects(labled, min_size=6)
        self.image = labeled != 0
        return self
        
    
    def getCoords(self):
        coords_row, coords_col = np.where(self.image != 0)
        return list(coords_row), list(coords_col)
    
    
    def getLineSegments2(self, step=5):
        colin_thresh = 2
        
        rows, cols = self.getCoords()
        
        final_rows = obj_copy.copy(rows)
        final_cols = obj_copy.copy(cols)        
        
        for i in range(0, len(rows) - 2*step, step):
            dy_1 = rows[i+step] - rows[i]
            dx_1 = cols[i+step] - cols[i]
            
            dy_2 = rows[i+step+step] - rows[i+step]
            dx_2 = cols[i+step+step] - cols[i+step]
            
            if   (dx_1 == 0 and dx_2 == 0) or (dy_1 == 0 and dy_2 == 0):
                # its line, so keep it
                continue
            
            elif ( (dx_1 < colin_thresh and dx_2 < colin_thresh)
                or (dy_1 < colin_thresh and dy_2 < colin_thresh) ):
                # its line, so keep it
                continue
            
            elif ( (dx_1 < colin_thresh and dx_2 > colin_thresh)
                or (dx_1 > colin_thresh and dx_2 < colin_thresh) ):
                # not a line, so remove it
                del final_rows[i:i+step+step]
                del final_cols[i:i+step+step]
                
            elif ( (dy_1 < colin_thresh and dy_2 > colin_thresh)
                or (dy_1 > colin_thresh and dy_2 < colin_thresh) ):
                # not a line, so remove it
                del final_rows[i:i+step+step]
                del final_cols[i:i+step+step]
            
            else:
                line1 = Line(Point(rows[i],cols[i]), Point(rows[i+step],cols[i+step]))
                line2 = Line(Point(rows[i+step],cols[i+step]), Point(rows[i+step+step],cols[i+step+step]))
            
                if not line1.isParallelTo(line2):
                    del final_rows[i:i+step+step]
                    del final_cols[i:i+step+step]    
            
        return final_rows, final_cols
        
    
    def getLineSegments(self, length=7, prob1=0.95, prob2=0.80):
        segment_test_length = length
        
        rows, cols = self.getCoords()
        
        # initialize rows and cols
        final_rows = obj_copy.copy(rows)
        final_cols = obj_copy.copy(cols)
        
        for i in range(0, len(rows) - segment_test_length, 2):
            start = Point(rows[i], cols[i])
            end   = Point(rows[i + segment_test_length], cols[i + segment_test_length])
            
            L = Line(start, end)
            
            pixels = L.getFromImage(self.image)
            line_probability = np.count_nonzero(pixels)/segment_test_length
            
            if line_probability <= prob2 and line_probability >= prob1:
                del final_rows[i : i + segment_test_length]
                del final_cols[i : i + segment_test_length]
        
        return final_rows, final_cols
                
    
    def getCurveSegments(self):
        segment_test_length = 7
        
        rows, cols = self.getCoords()
        
        # initialize rows and cols
        final_rows = obj_copy.copy(rows)
        final_cols = obj_copy.copy(cols)
                
        for i in range(0, len(rows) - segment_test_length, segment_test_length):
            start = Point(rows[i], cols[i])
            end   = Point(rows[i + segment_test_length], cols[i + segment_test_length])
            
            L = Line(start, end)
            
            pixels = L.getFromImage(self.image)
            line_probability = np.count_nonzero(pixels)/np.size(pixels)
            
            if line_probability >= 0.90:
                del final_rows[i : i + segment_test_length]
                del final_cols[i : i + segment_test_length]
                
        return final_rows, final_cols
            

    def removeSegments(self, *args):
        if len(args) == 2 and len(args[0]) == len(args[1]):
            rows = args[0]
            cols = args[1]
        elif len(args) == 1 and len(args[0][0]) == len(args[0][1]):
            rows = args[0][0]
            cols = args[0][1]
        else:
            raise TypeError('unknown arguments.')

        self.image[rows, cols] = 0
                  
                  
    def addSegments(self, *args):
        if len(args) == 2 and len(args[0]) == len(args[1]):
            rows = args[0]
            cols = args[1]
        elif len(args) == 1 and len(args[0][0]) == len(args[0][1]):
            rows = args[0][0]
            cols = args[0][1]
        else:
            raise TypeError('unknown arguments.')
            
        self.image[rows, cols] = 1

    
    def show(self, title='', cmap='gray'):
        fig = plt.figure()
    
        plt.imshow(self.image, cmap)
        plt.axis('off')
        plt.title(title)
        plt.tight_layout()
        
        # bring it to front
        fig.canvas.manager.window.raise_()
        return self
    
        
    def backup(self):
        self.imageBackup = np.copy(self.image)
    
    def restore(self):
        self.image = np.copy(self.imageBackup)
        return self
        
    def detectLines(self):
        return LinesDetector(self).getLines()
    