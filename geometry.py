from math import sqrt, inf, atan, cos, pi
import numpy as np

from abc import ABCMeta, abstractmethod


class SpatialObject(metaclass=ABCMeta):

    @abstractmethod
    def getMBR(self): raise NotImplemented()
    # (xmin, ymin, xmax, ymax)
    
    @abstractmethod
    def getPoints(self): raise NotImplemented()
    # list[Point, Point, ...]
    
    @abstractmethod
    def drawOnImage(self, image): raise NotImplemented()
    # list[Point, Point, ...]
        
    @abstractmethod
    def __eq__(self, other): raise NotImplemented()
    # T/F
    
    @abstractmethod
    def __ne__(self, other): raise NotImplemented()
    # T/F



            

class Point(SpatialObject):
    
    def __init__(self, y, x):
        self.y = y
        self.x = x
    
    def getMBR(self):
        return (self.x, self.y, self.x, self.y)
    
    
    def getPoints(self):
        points = []
        points.append(self)
        return points
        
    def __repr__(self):
        return "(" + str(self.y) + ", " + str(self.x) + ")"
    
    
    def __str__(self):
        return "(" + str(self.y) + ", " + str(self.x) + ")"


    def __hash__(self):
        return hash(str(self))
    
    
    def __add__(self, otherPoint):
        if not isinstance(otherPoint, Point):
            raise TypeError("both operands of point addition must be Point objects")

        return Point(self.y + otherPoint.y, self.x + otherPoint.x)


    def __sub__(self, otherPoint):
        if not isinstance(otherPoint, Point):
            raise TypeError("both operands of point subtraction must be Point objects")

        return Point(self.y - otherPoint.y, self.x - otherPoint.x)
    
    
    def __lt__(self, otherPoint):
        if not isinstance(otherPoint, Point):
            raise TypeError("both operands of point subtraction must be Point objects")

        return (self.y < otherPoint.y and self.x < otherPoint.x)


    def __le__(self, otherPoint):
        if not isinstance(otherPoint, Point):
            raise TypeError("both operands of point subtraction must be Point objects")

        return (self.y <= otherPoint.y and self.x <= otherPoint.x)
 
    
    def __gt__(self, otherPoint):
        if not isinstance(otherPoint, Point):
            raise TypeError("both operands of point subtraction must be Point objects")

        return (self.y > otherPoint.y and self.x > otherPoint.x)  
    
    
    def __ge__(self, otherPoint):
        if not isinstance(otherPoint, Point):
            raise TypeError("both operands of point subtraction must be Point objects")

        return (self.y >= otherPoint.y and self.x >= otherPoint.x)
    
    
    def __eq__(self, otherPoint):
        if not isinstance(otherPoint, Point):
            raise TypeError("both operands of point subtraction must be Point objects")

        return (self.y == otherPoint.y and self.x == otherPoint.x)


    def __ne__(self, otherPoint):
        if not isinstance(otherPoint, Point):
            raise TypeError("both operands of point subtraction must be Point objects")

        return (self.y != otherPoint.y or self.x != otherPoint.x)


    def drawOnImage(self, image, value='default'):
        if value != 'default':
            if not isinstance(image, np.ndarray):
                raise TypeError('The first argument, "image", must be a numpy ndarray.')
            
            if image.ndim == 2 and not isinstance(value, int):
                raise TypeError('The "value" argument must be an integer for binary images.')
                
            elif image.ndim >= 3 and not isinstance(value, tuple):
                raise TypeError('The "value" argument must be a list of integers for RGB images.')
            
            elif image.ndim >= 3 and isinstance(value, tuple):
                if not len(value) == 3:
                    raise TypeError('The "value" argument must contain 3 integers for RGB images.')
                else:
                    if not isinstance(value[0], float) or not isinstance(value[1], float) or not isinstance(value[2], float):
                        raise TypeError('One or more elements of the "value" argument are not floats.')
        
        if value == 'default' and image.ndim == 2:
            value = 1
        
        if value == 'default' and image.ndim == 3:
            if image.shape[2] == 3:
                value = (0.0, 0.0, 0.0)
            if image.shape[2] == 4:
                value = (0.0, 0.0, 0.0, 1.0)
        
        
        if image.ndim == 2:
            image[self.y, self.x] = value
                 
        if image.ndim == 3:
            if image.shape[2] == 4 and len(value) == 3:
                value = (value[0], value[1], value[2], 1.0)
            
            image[self.y, self.x, :] = value

    
    def getDistanceTo(self, Point_B):
        y1 = self.y
        x1 = self.x
        
        y2 = Point_B.y
        x2 = Point_B.x
        
        distance = sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
        
        return distance
    

            

class Line(SpatialObject):
    
    def __init__(self, A: Point, B: Point):
        if not isinstance(A, Point) or not isinstance(B, Point):
            raise TypeError("arguments to Line constructor should be of type Point")
            
        self.pointA = A
        self.pointB = B

    
    def __repr__(self):
        return self.pointA.__repr__() + " - " + self.pointB.__repr__()
    
    
    def __str__(self):
        return self.pointA.__str__() + " - " + self.pointB.__str__()

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, otherLine):
#        if not isinstance(otherLine, Line):
#            raise TypeError("both operands of line operation must be Line objects")
#
#        return (self.pointA == otherLine.pointA and self.pointB == otherLine.pointB)
        return hash(self) == hash(otherLine)


    def __ne__(self, otherLine):
#        if not isinstance(otherLine, Line):
#            raise TypeError("both operands of line operation must be Line objects")
#
#        return (self.pointA != otherLine.pointA or self.pointB != otherLine.pointB)
        return hash(self) != hash(otherLine)
    
    
    def getMBR(self):
        pA = self.pointA
        pB = self.pointB
        xmin   = min(pA.x, pB.x)
        ymin   = min(pA.y, pB.y)
        xmax   = max(pA.x, pB.x)
        ymax   = max(pA.y, pB.y)
        
        return (xmin, ymin, xmax, ymax)
    
    
    def getPoints(self):
        return self.getAllPointsInLine(True)

    
    def getLength(self):
        return self.pointA.getDistanceTo(self.pointB)
    
    
    def getSlope(self):
        dy = self.pointB.y - self.pointA.y
        dx = self.pointB.x - self.pointA.x
        
        if dx == 0:
            return inf
        else:
            return dy / dx
    
    
    def getYIntercept(self):
        y = self.pointA.y
        mx = self.getSlope() * self.pointA.x
        return y - mx
    
    
    def getXIntercept(self):
        slope = self.getSlope()
        
        if slope == 0 or slope == inf:
            return inf
        else:
            return -1 * self.getYIntercept() / slope


    def getDistanceTo(self, anotherLine, min_or_max_distance = 'min'):
        dists = []
        dists.append(self.pointA.getDistanceTo(anotherLine.pointA))
        dists.append(self.pointA.getDistanceTo(anotherLine.pointB))
        dists.append(self.pointB.getDistanceTo(anotherLine.pointA))
        dists.append(self.pointB.getDistanceTo(anotherLine.pointB))
        
        if   min_or_max_distance == 'min':
            return min(dists)
        elif min_or_max_distance == 'max':
            return max(dists)
        else:
            raise ValueError('second argument to "getDistanceTo" must be either "min" or "max".')
            
    
    def isCCWwithPoint(self, point):
        def ccw(A, B, C):
            return (B.x - A.x) * (C.y - A.y) > (B.y - A.y) * (C.x - A.x)
        
        distA = self.pointA.getDistanceTo(point)
        distB = self.pointB.getDistanceTo(point)
        
        if distA >= distB:
            A = self.pointA
            B = self.pointB
        else:
            A = self.pointB
            B = self.pointA
        
        C = point
        
        return ccw(A, B, C)


    def isCWwithPoint(self, point):
        def cw(A, B, C):
            return (B.x - A.x) * (C.y - A.y) < (B.y - A.y) * (C.x - A.x)
        
        distA = self.pointA.getDistanceTo(point)
        distB = self.pointB.getDistanceTo(point)
        
        if distA >= distB:
            A = self.pointA
            B = self.pointB
        else:
            A = self.pointB
            B = self.pointA
            
        C = point
        
        return cw(A, B, C)
    
    
    def isColinearWithPoint(self, point):
        
        return not self.isCCWwithPoint(point) and not self.isCWwithPoint(point)
    
    
    def getAngle(self, deg_or_rad='deg'):
        '''Returns the angle of the line with respect to positive x axis.'''
        
        angle = atan(self.getSlope()) * (180/pi)

        if angle < 0:
            angle = 180 + angle
        
        angle = abs(angle)
        
        if deg_or_rad == 'deg':
            return angle
        elif deg_or_rad == 'rad':
            return angle * (pi/180)
        else:
            raise ValueError('second argument to "getAngle" must be either "deg" or "rad"')    
    
        
    
    def getAngleWith(self, anotherLine, deg_or_rad='deg'):
        angle_A = self.getAngle(deg_or_rad)
        angle_B = anotherLine.getAngle(deg_or_rad)
        
        angle = abs(angle_A - angle_B)                
        
        return angle        
    
    
    def isParallelTo(self, anotherLine, degrees_tolerance=18):
        diff = abs(self.getAngleWith(anotherLine))
        
        if diff <= degrees_tolerance:
            return True
        else:
            return False
        
        
    def isPerpendicularTo(self, anotherLine, degrees_tolerance=10):
        diff = abs(self.getAngleWith(anotherLine))
        
        if diff <= 90 + degrees_tolerance and diff >= 90 - degrees_tolerance:
            return True
        else:
            return False
    
    
    def isSkewTo(self, anotherLine):
        return not self.isIntersectingWith(anotherLine) and not self.isParallelTo(anotherLine)
    
    
    def isColinearTo(self, anotherLine, degrees_tolerance=18, slope_tolerance=0.25):
        if not self.isParallelTo(anotherLine, degrees_tolerance):
            return False
        else:
            slope = self.getSlope()
            test_line_1 = Line(self.pointA, anotherLine.pointA)
            test_line_2 = Line(self.pointA, anotherLine.pointB)
            
            test_1_slope = test_line_1.getSlope()
            test_2_slope = test_line_2.getSlope()
            
            if test_1_slope == slope and test_2_slope == slope:
                return True
            else:
                diff_1 = abs(slope - test_1_slope)
                diff_2 = abs(slope - test_2_slope)
                print(slope)
                print(' ')
                print(test_1_slope)
                print(diff_1)
                print(' ')
                print(test_2_slope)
                print(diff_2)
                print(' ')
                
                thresh = abs(slope_tolerance*slope)
                print(thresh)
                
                if ( (diff_1 <= thresh or test_line_1.getLength() == 0.0)
                    and (diff_2 <= thresh or test_line_2.getLength() == 0.0)):
                    return True
                else:
                    return False
     
        
    def mergeWith(self, anotherLine):
        x_coords = [self.pointA.x,
                    self.pointB.x,
                    anotherLine.pointA.x,
                    anotherLine.pointB.x]

        y_coords = [self.pointA.y,
                    self.pointB.y,
                    anotherLine.pointA.y,
                    anotherLine.pointB.y]            
        
        if max(y_coords) != min(y_coords):
            start_idx = y_coords.index(min(y_coords))
            end_idx   = y_coords.index(max(y_coords))
        else:
            start_idx = x_coords.index(min(x_coords))
            end_idx   = x_coords.index(max(x_coords))
            
        start_y = y_coords[start_idx]
        start_x = x_coords[start_idx]
        
        end_y = y_coords[end_idx]
        end_x = x_coords[end_idx]
        
        self.pointA.y = start_y
        self.pointA.x = start_x
        self.pointB.y = end_y
        self.pointB.x = end_x
        
        return self
            
            #merged_line = (  (start_y, start_x)  ,  (end_y, end_x)   )


    def splitAt(self, splitPoint):
        if not self.hasPoint(splitPoint):
            raise ValueError('the specified "splitPoint" is not in the line.')
        
        split_1 = Line(self.pointA, splitPoint)
        split_2 = Line(self.pointB, splitPoint)
        
        return [split_1, split_2]
        
    
    
    def getClosestPointTo(self, aGivenPoint):
        allPoints = self.getAllPointsInLine(True)
         
        dists = []
        for point in allPoints:
            dists.append(point.getDistanceTo(aGivenPoint))
             
        min_idx = dists.index(min(dists))
        return allPoints[min_idx]        
    
    

    def hasPoint(self, testPoint: Point):
        if self.isColinearTo(Line(self.pointB, testPoint)):
            y = testPoint.y
            x = testPoint.x
            
            pAx = self.pointA.x
            pAy = self.pointA.y
            pBx = self.pointB.x
            pBy = self.pointB.y
            
            if (x <= max((pAx, pBx)) and x >= min((pAx, pBx)) and 
                y <= max((pAy, pBy)) and y >= min((pAy, pBy)) ):
                return True
            else:
                print('bp1')
                return False
                
        else:
            print('bp2')
            return False

            
    def getIntersectionWith(self, anotherLine):
        return list(set(self.getPoints()).intersection(set(anotherLine.getPoints())))        


    def isIntersectingWith(self, anotherLine):
        return len(self.getIntersectionWith(anotherLine)) > 0
    
        
    def getProjectionOn(self, anotherLine):
        theta = self.getAngleWith(anotherLine, 'rad')
        a_norm = self.getLength()
        projection = a_norm * cos(theta)
        return projection
    
    
    def drawOnImage(self, image, value='default'):
        if value != 'default':
            if not isinstance(image, np.ndarray):
                raise TypeError('The first argument, "image", must be a numpy ndarray.')
            
            if image.ndim == 2 and not isinstance(value, int):
                raise TypeError('The "value" argument must be an integer for binary images.')
                
            elif image.ndim >= 3 and not isinstance(value, tuple):
                raise TypeError('The "value" argument must be a list of integers for RGB images.')
            
            elif image.ndim >= 3 and isinstance(value, tuple):
                if not len(value) == 3:
                    raise TypeError('The "value" argument must contain 3 integers for RGB images.')
                else:
                    if not isinstance(value[0], float) or not isinstance(value[1], float) or not isinstance(value[2], float):
                        raise TypeError('One or more elements of the "value" argument are not floats.')
        
        if value == 'default' and image.ndim == 2:
            value = 1
        
        if value == 'default' and image.ndim == 3:
            if image.shape[2] == 3:
                value = (0.0, 0.0, 0.0)
            if image.shape[2] == 4:
                value = (0.0, 0.0, 0.0, 1.0)
        
        rows, cols = self.getAllPointsInLine()
        
        if image.ndim == 2:
            image[rows, cols] = value
                 
        if image.ndim == 3:
            if image.shape[2] == 4 and len(value) == 3:
                value = (value[0], value[1], value[2], 1.0)
            
            image[rows, cols, :] = value
    
    
    def getFromImage(self, image):
        '''Returns a list of pixel values in the image which lie on the line'''
        rows, cols = self.getAllPointsInLine()
        
        if image.ndim == 2:
            return image[rows, cols]
        
        if image.ndim == 3:
            return image[rows, cols, :]
    
    
    def getAllPointsInLine(self, return_as_point_objects=False):
        # Setup initial conditions
        #x1, y1 = start
        #x2, y2 = end
        x1 = self.pointA.x
        y1 = self.pointA.y
        x2 = self.pointB.x
        y2 = self.pointB.y
        
        # the above matches numpy's matrix order (e.g. row, col). Below is wrong
        #y1, x1 = start
        #y2, x2 = end
        dx = x2 - x1
        dy = y2 - y1
     
        # list of row and col coords
        rows = []
        cols = []
        
        # Determine how steep the line is
        is_steep = abs(dy) > abs(dx)
     
        # Rotate line
        if is_steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2
     
        # Swap start and end points if necessary and store swap state
        swapped = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            swapped = True
     
        # Recalculate differentials
        dx = x2 - x1
        dy = y2 - y1
     
        # Calculate error
        error = int(dx / 2.0)
        ystep = 1 if y1 < y2 else -1
     
        # Iterate over bounding box generating points between start and end
        y = y1
        for x in range(x1, x2 + 1):
            coord = (y, x) if is_steep else (x, y)
            rows.append(coord[0])
            cols.append(coord[1])
    
            error -= abs(dy)
            if error < 0:
                y += ystep
                error += dx
     
        # Reverse the list if the coordinates were swapped
        if swapped:
            rows.reverse()
            cols.reverse()
            
        if return_as_point_objects:
            point_obj_list = []
            for i in range(len(rows)):
                point_obj_list.append(Point(rows[i], cols[i]))
                
            return point_obj_list
        
        else:
            #return rows, cols
            return cols, rows


class Circle(SpatialObject):
    
    def __init__(self, center: Point, radius):
        if not isinstance(center, Point):
            raise TypeError("First argument to Circle constructor should be of type Point")
            
        self.center = center
        self.radius = radius   
   
    
    def __repr__(self):
        return "(" + str(self.center.y) + ", " + str(self.center.x) + "), (" + str(self.radius) + ")"
    
    
    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, otherCircle):
        if not isinstance(otherCircle, Line):
            raise TypeError("both operands of circle comparison must be Circle objects")

        return (self.center == otherCircle.center and self.radius == otherCircle.radius)


    def __ne__(self, otherCircle):
        if not isinstance(otherCircle, Line):
            raise TypeError("both operands of circle comparison must be Circle objects")

        return (self.center != otherCircle.center or self.radius != otherCircle.radius)


    def getMBR(self):
        xmin   = self.center.x - self.radius
        ymin   = self.center.y - self.radius
        xmax   = self.center.x + self.radius
        ymax   = self.center.y + self.radius
        
        return (xmin, ymin, xmax, ymax)
    
    
    def getPoints(self):
        return self.getAllPointsOnCircle(True)
     
    
    def drawOnImage(self, image, value='default'):
        if value != 'default':
            if not isinstance(image, np.ndarray):
                raise TypeError('The first argument, "image", must be a numpy ndarray.')
            
            if image.ndim == 2 and not isinstance(value, int):
                raise TypeError('The "value" argument must be an integer for binary images.')
                
            elif image.ndim >= 3 and not isinstance(value, tuple):
                raise TypeError('The "value" argument must be a list of integers for RGB images.')
            
            elif image.ndim >= 3 and isinstance(value, tuple):
                if not len(value) == 3:
                    raise TypeError('The "value" argument must contain 3 integers for RGB images.')
                else:
                    if not isinstance(value[0], float) or not isinstance(value[1], float) or not isinstance(value[2], float):
                        raise TypeError('One or more elements of the "value" argument are not floats.')
        
        if value == 'default' and image.ndim == 2:
            value = 1
        
        if value == 'default' and image.ndim == 3:
            if image.shape[2] == 3:
                value = (0.0, 0.0, 0.0)
            if image.shape[2] == 4:
                value = (0.0, 0.0, 0.0, 1.0)
        
        rows, cols = self.getAllPointsOnCircle()
        
        if image.ndim == 2:
            image[rows, cols] = value
                 
        if image.ndim == 3:
            if image.shape[2] == 4 and len(value) == 3:
                value = (value[0], value[1], value[2], 1.0)
            
            image[rows, cols, :] = value
    
    
    def getFromImage(self, image):
        '''Returns a list of pixel values in the image which lie on the line'''
        rows, cols = self.getAllPointsOnCircle()
        
        if image.ndim == 2:
            return image[rows, cols]
        
        if image.ndim == 3:
            return image[rows, cols, :]

        
    def getAllPointsOnCircle(self, return_as_point_objects=False):
        center = self.center
        radius = self.radius
        
        y0 = center.y
        x0 = center.x
        
        f = 1 - radius
        ddf_x = 1
        ddf_y = -2 * radius
        
        # list of row and col coords
        rows = []
        cols = []
        
        x = 0
        y = radius
        
        #self.set(x0, y0 + radius, colour)
        cols.append(x0)
        rows.append(y0 + radius)
        
        #self.set(x0, y0 - radius, colour)
        cols.append(x0)
        rows.append(y0 - radius)
        
        #self.set(x0 + radius, y0, colour)
        cols.append(x0 + radius)
        rows.append(y0)
        
        #self.set(x0 - radius, y0, colour)
        cols.append(x0 - radius)
        rows.append(y0)
    
        while x < y:
            if f >= 0: 
                y -= 1
                ddf_y += 2
                f += ddf_y
            x += 1
            ddf_x += 2
            f += ddf_x
            
            cols.append(x0 + x)
            rows.append(y0 + y)
            
            cols.append(x0 - x)
            rows.append(y0 + y)

            cols.append(x0 + x)
            rows.append(y0 - y)
            
            cols.append(x0 - x)
            rows.append(y0 - y)
            
            cols.append(x0 + y)
            rows.append(y0 + x)
            
            cols.append(x0 - y)
            rows.append(y0 + x)
            
            cols.append(x0 + y)
            rows.append(y0 - x)
            
            cols.append(x0 - y)
            rows.append(y0 - x)
        
        if return_as_point_objects:
            point_obj_list = []
            for i in range(len(rows)):
                point_obj_list.append(Point(rows[i], cols[i]))
                
            return point_obj_list
        
        else:
            return rows, cols
        
        
class Polygon:
    
    def __init__(self, pointsList):
        
        self.lineSegments = []
        rows, cols = pointsList[:,0], pointsList[:,1]
        
        for p in range(len(rows)-1):
            pA = Point(rows[p]  , cols[p]  )
            pB = Point(rows[p+1], cols[p+1])
            self.lineSegments.append(Line(pA, pB))
    
        
    def drawOnImage(self, image, value=1):
        for line in self.lineSegments:
            line.drawOnImage(image, value)
            
    def __len__(self):
        return len(self.lineSegments)+1
    
        