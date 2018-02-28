# -*- coding: utf-8 -*-

from skimage.feature import local_binary_pattern, corner_harris, corner_peaks
from skimage.color import rgb2gray, rgb2ycbcr
from skimage.morphology import skeletonize, medial_axis, remove_small_holes, \
                        binary_dilation, binary_erosion, square, disk

from skimage.filters import scharr
from skimage.measure import label, regionprops, find_contours, approximate_polygon
from skimage.morphology import convex_hull_image, remove_small_objects
from skimage.transform import hough_line, hough_line_peaks

from scipy.ndimage.morphology import distance_transform_edt, binary_fill_holes

import matplotlib.pyplot as plt
import numpy as np
import cv2

import PIL.Image, PIL.ImageDraw, PIL.ImageFont
from util import pad_zeros_around_image
import base64

from LinesDetector import LinesDetector, LinesCollection

from geometry import Polygon

from RtreeIndex import RtreeIndex

import networkx as nx

import thinning

from lxml import etree



def __sortCoords(coords):

    # coords = [  [y,x] , [y,x] , ...]
    return sorted(coords , key=lambda c: (c[0], c[1]))

class VisualObject:

    def __init__(self, regionObject, rgbImage, objId=''):
        self.regionObject = regionObject
        self.rootRGBImage = rgbImage
        self.image = np.copy(regionObject.image)
        self.wholeImageShape = rgbImage.shape[0], rgbImage.shape[1]
        self.objectId = objId

        bbox = regionObject.bbox
        # format: (min_row, min_col, max_row, max_col)  where row = y, col =x
        # convert to Rtrees' (xmin,ymin,xmax,ymax) format
        self.MBR = (bbox[1], bbox[0], bbox[3], bbox[2])

        self.regionImage = None
        self.convexRegionImage = None

        self.boundaryImage = None
        self.convexBoundaryImage = None

        self.polygon = None
        self.convexPolygon = None

        self.parent = None
        self.children = []

        self.DOM = None

        # DOM properties
        self.classification = ''
        self.center = regionObject.centroid
        self.zorder = 0
        self.color = None

        self.pointSet = None
        self.diameter = None
        self.height = None
        self.width = None
        self.orientation = None

        self.tempImage1 = np.zeros(self.wholeImageShape, dtype=bool)
        self.tempImage2 = np.zeros(self.wholeImageShape, dtype=bool)


    def calculateColor(self):
        region = self.getRegionImage()

        rows, cols = np.where(region == 1)
        (xmin,ymin,xmax,ymax) = self.MBR

        rows = rows + ymin
        cols = cols + xmin

        pixelValues = np.uint8( self.rootRGBImage[rows, cols, 0:3] *255 )

        self.color = np.uint8(np.mean(pixelValues, axis=0))

        return self.color


    def getDOM(self, returnAsString=False):
        if self.DOM == None:
            if self.classification != '':
                element = etree.Element(self.classification)

                if self.classification == 'line':
                    element.attrib['point-a'] = '(' + str(int(self.pointSet[0][0])) + ',' + str(int(self.pointSet[0][1])) + ')'
                    element.attrib['point-b'] = '(' + str(int(self.pointSet[1][0])) + ',' + str(int(self.pointSet[1][1])) + ')'


                if self.classification == 'circle':
                    element.attrib['center'] = '(' + str(int(self.center[0])) + ',' + str(int(self.center[1])) + ')'
                    (xmin,ymin,xmax,ymax) = self.MBR
                    width = xmax - xmin
                    height = ymax - ymin

                    diameter = np.uint16(np.mean([width, height]))
                    element.attrib['diameter'] = str(diameter)


                if self.classification == 'rectangle':  #includes square as a special case
                    element.attrib['center'] = '(' + str(int(self.center[0])) + ',' + str(int(self.center[1])) + ')'
                    angle = self.regionObject.orientation * 180/np.pi

                    if self.regionObject.minor_axis_length / self.regionObject.major_axis_length > 0.90:
                        element.tag = 'square'
                        element.attrib['size'] = str(int(self.regionObject.major_axis_length))
                    else:
                        if angle < 15 and angle > -15:
                            element.attrib['width'] = str(int(self.regionObject.major_axis_length))
                            element.attrib['height'] = str(int(self.regionObject.minor_axis_length))
                        else:
                            element.attrib['width'] = str(int(self.regionObject.minor_axis_length))
                            element.attrib['height'] = str(int(self.regionObject.major_axis_length))

                    if (not (abs(angle) <= 90 and abs(angle) >= 85)
                        and not (abs(angle) <= 5)):
                        element.attrib['rotation'] = str(int(angle))


                if self.classification == 'triangle':
                    element.attrib['center'] = '(' + str(int(self.center[0])) + ',' + str(int(self.center[1])) + ')'
                    element.attrib['point-a'] = '(' + str(int(self.pointSet[0][0])) + ',' + str(int(self.pointSet[0][1])) + ')'
                    element.attrib['point-b'] = '(' + str(int(self.pointSet[1][0])) + ',' + str(int(self.pointSet[1][1])) + ')'
                    element.attrib['point-c'] = '(' + str(int(self.pointSet[2][0])) + ',' + str(int(self.pointSet[2][1])) + ')'


                if self.classification == 'polygon':
                    element.attrib['center'] = '(' + str(int(self.center[0])) + ',' + str(int(self.center[1])) + ')'
                    pointsStr = ''

                    for point in self.pointSet:
                        pointsStr += '(' + str(int(point[0])) + ',' + str(int(point[1])) + '),'

                    element.attrib['points'] = pointsStr[:-1] # to remove the last comma


                # shared attributes
                element.attrib['z-order'] = str(int(self.zorder))
                color = self.calculateColor()
                element.attrib['color'] = '#%02X%02X%02X' % (color[0], color[1], color[2])


                if returnAsString:
                    result = etree.tostring(element,
                                               pretty_print=True,
                                               encoding='unicode')
                else:
                    result = element

                self.DOM = element

                return result

            else:
                return ''
        else:

            return self.DOM


    def __str__(self):
        return str(self.objectId)


    def __repr__(self):
        return str(self.objectId)


    def __eq__(self, other):
        if self.objectId != '' and other.objectId != '':
            return self.objectId == other.objectId
        else:
            return hash(self) == hash(other)


    def __ne__(self, other):
        if self.objectId != '' and other.objectId != '':
            return self.objectId != other.objectId
        else:
            return hash(self) != hash(other)


    def getMBR(self):
        return self.MBR

    def getRegionImage(self):
        if self.boundaryImage == None:
            self.boundaryImage = self.getBoundaryImage()

        self.regionImage = binary_fill_holes(self.boundaryImage)
        return self.regionImage

    def getConvexRegionImage(self):
        if self.boundaryImage == None:
            self.boundaryImage = self.getBoundaryImage()

        self.convexRegionImage = convex_hull_image(self.boundaryImage)
        return self.convexRegionImage


    def getBoundaryImage(self):
        dst = distance_transform_edt(self.image)
        self.boundaryImage = (dst == 1)
        return self.boundaryImage

    def getConvexBoundaryImage(self):
        if self.convexRegionImage == None:
            self.convexRegionImage = self.getConvexRegionImage()

        dst = distance_transform_edt(self.convexRegionImage)
        self.convexBoundaryImage = (dst == 1)
        return self.convexBoundaryImage


    def getPolygon(self):
        if self.boundaryImage == None:
            self.boundaryImage = self.getBoundaryImage()

        rows, cols = self.getSortedPoints(self.boundaryImage)
        coords = np.array([rows, cols]).transpose()
        polygon = approximate_polygon(coords, 10)
        self.polygon = Polygon(polygon)
        return self.polygon


    def getPolygonImage(self):
        if self.polygon == None:
            self.polygon = self.getPolygon()

        image = np.zeros(self.image.shape)
        self.polygon.drawOnImage(image)

        return image


    def getSortedPoints(self, image):
        contours = find_contours(image, 0.0, fully_connected='high',
                          positive_orientation='high')

        len_array = []
        for contour in contours:
            len_array.append(len(contour))

        idx = np.argsort(len_array)[-1]

        return np.uint16(contours[idx][:,0]), np.uint16(contours[idx][:,1])


    def getDistanceTo(self, other):
        if self.boundaryImage == None:
            self.boundaryImage = self.getBoundaryImage()

        if other.boundaryImage == None:
            other.boundaryImage = other.getBoundaryImage()

        this_y, this_x   = self.getSortedPoints(self.boundaryImage)
        other_y, other_x = other.getSortedPoints(other.boundaryImage)

        dist = np.sqrt( (this_x - other_x)**2 + (this_y - other_y)**2 )

        return np.mean(dist), np.std(dist)


    def getConvexPolygon(self):
        if self.convexBoundaryImage == None:
            self.convexBoundaryImage = self.getConvexBoundaryImage()

        rows, cols = self.getSortedPoints(self.convexBoundaryImage)
        coords = np.array([rows, cols]).transpose()
        polygon = approximate_polygon(coords, 10)
        self.convexPolygon = Polygon(polygon)
        return self.convexPolygon


    def getConvexPolygonImage(self):
        if self.convexPolygon == None:
            self.convexPolygon = self.getConvexPolygon()

        image = np.zeros(self.image.shape)
        self.convexPolygon.drawOnImage(image)

        return image


    def getSolidity(self):
        regionImage = self.getRegionImage()
        convexImage = self.getConvexRegionImage()
        solidity = np.count_nonzero(regionImage) / np.count_nonzero(convexImage)
        return solidity


    def show(self):
        f, axarr = plt.subplots(3, 3)

        polygon = self.getPolygon()
        convexPolygon = self.getConvexPolygon()

        axarr[0, 1].imshow(self.image)
        axarr[0, 1].set_title('Original (input) image')

        axarr[1, 0].imshow(self.getBoundaryImage())
        axarr[1, 0].set_title('getBoundaryImage()')

        axarr[1, 1].imshow(self.getRegionImage())
        axarr[1, 1].set_title('getRegionImage()')

        axarr[1, 2].imshow(self.getPolygonImage())
        axarr[1, 2].set_title('getPolygonImage()\n# polygon points: {}'.format(len(polygon)))

        axarr[2, 0].imshow(self.getConvexBoundaryImage())
        axarr[2, 0].set_title('getConvexBoundaryImage()')

        axarr[2, 1].imshow(self.getConvexRegionImage())
        axarr[2, 1].set_title('getConvexRegionImage()')

        axarr[2, 2].imshow(self.getConvexPolygonImage())
        axarr[2, 2].set_title('getConvexPolygonImage()\n# polygon points: {}'.format(len(convexPolygon)))


    def isInsideQuickCheck(self, other):
        thisMBR = self.getMBR()
        otherMBR = other.getMBR()

        return RtreeIndex.isMbrAInsideB(thisMBR, otherMBR)


    def verifyIsInside(self, other):
        thisImage = self.getConvexRegionImage()
        (xmin,ymin,xmax,ymax) = self.getMBR()
        self.tempImage1[ymin:ymax, xmin:xmax] = thisImage

        otherImage = other.getConvexRegionImage()
        (xmin,ymin,xmax,ymax) = other.getMBR()
        self.tempImage2[ymin:ymax, xmin:xmax] = otherImage

        ANDimage = np.logical_and(self.tempImage1, self.tempImage2)

        result = np.array_equal(ANDimage, self.tempImage1)

        self.tempImage1.fill(False)
        self.tempImage2.fill(False)

        return result


    def isIntersectingWithQuickCheck(self, other):
        thisMBR = self.getMBR()
        otherMBR = other.getMBR()

        return RtreeIndex.areMBRsIntersecting(thisMBR, otherMBR)


    def verifyIsIntersectingWith(self, other):
        thisImage = self.getConvexRegionImage()
        (xmin,ymin,xmax,ymax) = self.getMBR()
        self.tempImage1[ymin:ymax, xmin:xmax] = thisImage

        otherImage = other.getConvexRegionImage()
        (xmin,ymin,xmax,ymax) = other.getMBR()
        self.tempImage2[ymin:ymax, xmin:xmax] = otherImage

        ANDimage = np.logical_and(self.tempImage1, self.tempImage2)
        ANDimage = remove_small_objects(ANDimage, min_size=15)

        result = np.count_nonzero(ANDimage) > 5

        self.tempImage1.fill(False)
        self.tempImage2.fill(False)

        return result


    def isOcclusionPossible(self):
        convexComplement = self.getConvexRegionImage() - self.getRegionImage()
        convexComplement = remove_small_objects(convexComplement, min_size=10)

        if np.count_nonzero(convexComplement) < 10:
            return False
        else:
            return True


    def isBehind(self, other):
        convexComplement = np.logical_xor(self.getConvexRegionImage(), self.getRegionImage())
        convexComplement = remove_small_objects(convexComplement, min_size=50)

        if np.count_nonzero(convexComplement) < 10:
            return False
        else:
            thisImage = convexComplement
            (xmin,ymin,xmax,ymax) = self.MBR
            self.tempImage1[ymin:ymax, xmin:xmax] = thisImage

            otherImage = other.getConvexRegionImage()
            (xmin,ymin,xmax,ymax) = other.MBR
            self.tempImage2[ymin:ymax, xmin:xmax] = otherImage

            ANDimage = np.logical_and(self.tempImage1, self.tempImage2)
            ANDimage = remove_small_objects(ANDimage, min_size=15)

            result = np.count_nonzero(ANDimage) > 20

            self.tempImage1.fill(False)
            self.tempImage2.fill(False)

            return result
#            rows, cols = np.where(convexComplement != 0)
#            xmin   = min(cols)
#            ymin   = min(rows)
#            xmax   = max(cols)
#            ymax   = max(rows)
#
#            thisMBR = (xmin, ymin, xmax, ymax)
#            otherMBR = other.getMBR()
#
#            return RtreeIndex.areMBRsIntersecting(thisMBR, otherMBR) \
#                    or RtreeIndex.isMbrAInsideB(thisMBR, otherMBR)

    def __hash__(self):
        return hash(base64.b64encode(self.getBoundaryImage()))


class ShapesDetector:

    def backup__init__(self, rgbImage):
        img = rgb2gray(rgbImage)
        img_scharr = scharr(img)

        img_interiors = (img_scharr < 0.01)
        img_interiors = self.removeBorderObjects(img_interiors)
        img_interiors_dst = distance_transform_edt(img_interiors)

        boundariesImage = (img_interiors_dst == 1)

        labeled = label(boundariesImage)
        self.image = labeled
        regions = regionprops(labeled)

        imageShape = self.image.shape

        self.visualObjects = []

        for r in range(len(regions)):
            self.visualObjects.append(VisualObject(regions[r], imageShape, r))

        self.index = RtreeIndex(self.visualObjects)

        self.graph = nx.DiGraph()


    def __init__(self, rgbImage):
        small_obj_thresh = 50

        #img = rgb2gray(rgbImage)
        img = rgb2ycbcr(rgbImage[:,:,0:3])[:,:,0]
        img_scharr = scharr(img)

        img_edges = (img_scharr > 0.01)
#        img_edges = remove_small_objects(img_edges, min_size=20)
        edge_1_split = np.zeros(img_edges.shape, dtype=bool)
        edge_2_split = np.zeros(img_edges.shape, dtype=bool)

        labeled_img = label(img_edges)
        regions = regionprops(labeled_img)

        #imageShape = img_edges.shape
        idx = 0
        self.visualObjects = []

#        print(len(regions))
        for r in range(len(regions)):

            if np.count_nonzero(regions[r].image) > small_obj_thresh:
                hspace, angles, dists = hough_line(regions[r].image)
                hspace, angles, dists = hough_line_peaks(hspace, angles, dists,
                                                         threshold=0.25*np.max(hspace),
                                                         num_peaks = 15)

#                plt.figure()
#                plt.imshow(regions[r].image)

                filled_image = binary_fill_holes(regions[r].image)
                geometricness = np.count_nonzero(filled_image) / np.count_nonzero(regions[r].convex_image)


#                print('--' + str(len(hspace)))

                if len(hspace) in [1,2,3,4]:
                    shape_is_geometric = True
                elif regions[r].euler_number == 0:
                    shape_is_geometric = True
                else:
#                    print('geo: {}'.format(geometricness))
                    shape_is_geometric = geometricness >= 0.98

    #            f, axarr = plt.subplots(1,2)
    #            axarr[0].imshow(filled_image)
    #            axarr[0].set_title('solidity: {}'.format(round(geometricness,3)))
    #            axarr[1].imshow(regions[r].convex_image)
    #            axarr[1].set_title('is_geometric: {}'.format(shape_is_geometric))


                if shape_is_geometric:
                    self.visualObjects.append(VisualObject(regions[r], rgbImage, idx))
#                    plt.figure()
#                    plt.imshow(regions[r].image)
                    idx += 1
                    img_edges[labeled_img == regions[r].label] = 0
                    edge_1_split[labeled_img == regions[r].label] = 1


        img_interiors = (img_edges != 1)
        img_interiors = self.removeBorderObjects(img_interiors)
        img_interiors_dst = distance_transform_edt(img_interiors)

        boundariesImage = (img_interiors_dst == 1)
#        boundariesImage = remove_small_objects(boundariesImage, min_size=10)
        edge_2_split = boundariesImage


        labeled = label(boundariesImage)
        regions = regionprops(labeled)

        for r in range(len(regions)):
            if np.count_nonzero(regions[r].image) > small_obj_thresh:
                self.visualObjects.append(VisualObject(regions[r], rgbImage, idx))
    #            plt.figure()
    #            plt.imshow(regions[r].image)
                idx += 1

#        edge_1_split = thinning.guo_hall_thinning(edge_1_split.astype('uint8'))

        self.image = np.logical_or(edge_1_split, edge_2_split)


        self.index = None
        self.childrenGraph = None
        self.intersectionsGraph = None
        self.zOrderGraph = None



    def showObjects(self, objectIds, title=''):
        image = np.zeros(self.image.shape)

        for ID in objectIds:
            mbr = self.visualObjects[ID].getMBR()
            objImg = self.visualObjects[ID].getBoundaryImage()

            image[mbr[1]:mbr[3], mbr[0]:mbr[2]] = objImg
            image[mbr[1], mbr[0]:mbr[2]] = 1
            image[mbr[3], mbr[0]:mbr[2]] = 1
            image[mbr[1]:mbr[3], mbr[0]] = 1
            image[mbr[1]:mbr[3], mbr[2]] = 1

        fig = plt.figure()
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.tight_layout()
        fig.canvas.manager.window.raise_()


    def buildGraph(self):
        if  self.index == None:
            self.index = RtreeIndex(self.visualObjects)

        if  self.childrenGraph == None:
            self.childrenGraph = nx.DiGraph()

        if  self.intersectionsGraph == None:
            self.intersectionsGraph = nx.Graph()

        for vizObject in self.visualObjects:
            self.childrenGraph.add_node(vizObject)
            self.intersectionsGraph.add_node(vizObject)

            intersections = self.index.getIntersections(vizObject)
            for neighbor in intersections:
                self.intersectionsGraph.add_edge(neighbor, vizObject)

                if neighbor != vizObject and neighbor.isInsideQuickCheck(vizObject):
                    if neighbor.verifyIsInside(vizObject):
                        self.childrenGraph.add_edge(neighbor, vizObject)

        return self


    def generateDOM(self):
        if self.childrenGraph == None:
            self.classify()
            self.buildGraph()
            self.calculateZorders()


        root = etree.Element('canvas')

#        for node in self.childrenGraph.nodes():
#            node.getDOM()

        for node, out_degree in self.childrenGraph.out_degree_iter():
            if out_degree == 0:
                DOM = node.getDOM()
                if DOM != '':
                    root.append(node.getDOM())

            elif out_degree == 1:
                child, parent = self.childrenGraph.edges(node)[0]
                DOM = parent.getDOM()
                if DOM != '' and child.getDOM() != '':
                    parent.getDOM().append(child.getDOM())



        result = etree.tostring(root,
                                       pretty_print=True,
                                       encoding='unicode')

        return result



    def classify(self):
        padding = 30
        print(len(self.visualObjects))

        for vizObject in self.visualObjects:
            image = np.copy(vizObject.getBoundaryImage())
            image = pad_zeros_around_image(image, padding)

            plt.figure()
            plt.imshow(image)

            hspace, angles, dists = hough_line(image)
            hspace, angles, dists = hough_line_peaks(hspace, angles, dists,
                                                     threshold=0.25*np.max(hspace),
                                                     num_peaks = 15)

#            hspace = LinesDetector(image).merge().merge().getLines()
#            print('   len(hspace) = {}'.format(len(hspace)))

#            hspace.plotOnImage(image)

            if len(hspace) == 1:
                vizObject.classification = 'line'

                corners = corner_harris(image, sigma=2)
                coords = corner_peaks(corners, exclude_border=False)

                offset_y = vizObject.MBR[1]
                offset_x = vizObject.MBR[0]

                for i in range(len(coords)):
                    coords[i] = offset_y + coords[i][0] - padding, offset_x + coords[i][1] - padding

                if len(coords) == 2:
                    vizObject.pointSet = []
                    vizObject.pointSet.append(coords[0])
                    vizObject.pointSet.append(coords[1])
                else:
                    self.visualObjects.remove(vizObject)
#                    print('removed')


            elif len(hspace) == 3:
                vizObject.classification = 'triangle'

                corners = corner_harris(image, sigma=2)
                coords = corner_peaks(corners, exclude_border=False)

                offset_y = vizObject.MBR[1]
                offset_x = vizObject.MBR[0]

                for i in range(len(coords)):
                    coords[i] = offset_y + coords[i][0] - padding, offset_x + coords[i][1] - padding

                if len(coords) == 3:
                    vizObject.pointSet = []
                    vizObject.pointSet.append(coords[0])
                    vizObject.pointSet.append(coords[1])
                    vizObject.pointSet.append(coords[2])
                else:
                    self.visualObjects.remove(vizObject)
#                    print('removed')

            elif len(hspace) == 4:
                vizObject.classification = 'rectangle'

            elif len(hspace) >= 8 and vizObject.regionObject.eccentricity < 0.35:
                vizObject.classification = 'circle'

            else:
                vizObject.classification = 'polygon'

                corners = corner_harris(image, sigma=2)
                coords = corner_peaks(corners, exclude_border=False)

                offset_y = vizObject.MBR[1]
                offset_x = vizObject.MBR[0]

                vizObject.pointSet = []
                for i in range(len(coords)):
                    coords[i] = offset_y + coords[i][0] - padding, offset_x + coords[i][1] - padding
                    vizObject.pointSet.append(coords[i])

                # fallback / re-adjust polygon classification
                if len(coords) == 3:
                    vizObject.classification = 'triangle'

                elif len(coords) == 4:
                    vizObject.classification = 'rectangle'

                elif len(coords) == 1:
                    self.visualObjects.remove(vizObject)

        print(len(self.visualObjects))


    def showClassification(self):
        pil_rgb_img = PIL.Image.fromarray(np.uint8(self.image*255))
        drawing_context = PIL.ImageDraw.Draw(pil_rgb_img)
        font = PIL.ImageFont.truetype('/Library/Fonts/Microsoft/Calibri Bold.ttf', 12)
        del_x = 20
        del_y = 14

        for vizObj in self.visualObjects:
            centroid = vizObj.center
            draw_at =         (int(centroid[1]-del_x), int(centroid[0]-del_y))
            annotation_text = vizObj.classification + '\n' + str(vizObj.objectId)
            drawing_context.text(draw_at, annotation_text, fill=(255), font=font)

        fig = plt.figure()
        plt.imshow(pil_rgb_img, cmap='gray')
        plt.tight_layout()
        fig.canvas.manager.window.raise_()


    def calculateZorders(self):
        if self.intersectionsGraph == None:
            self.buildGraph()

        self.zOrderGraph = nx.DiGraph()

        idx = []
        # loop through groups of intersections
        for nodeA, nodeB in self.intersectionsGraph.edges():
            self.zOrderGraph.add_node(nodeA)

            if nodeA == nodeB:
                continue

            self.zOrderGraph.add_node(nodeB)

            if (nodeB.isInsideQuickCheck(nodeA) and nodeB.verifyIsInside(nodeA)) \
                or nodeA.isBehind(nodeB):
#                print('-----------------------------')
#                print('node {} is behind node {}'.format(nodeA, nodeB))
#                print('   node {} zorder before: {}'.format(nodeB, nodeB.zorder))
#                print('   node {} zorder before: {}'.format(nodeA, nodeA.zorder))
                #nodeA.zorder = nodeB.zorder - 1
                self.zOrderGraph.add_edge(nodeA, nodeB)
                idx.append(nodeA.objectId)
#                print('   node {} zorder after: {}'.format(nodeB, nodeB.zorder))
#                print('   node {} zorder after: {}'.format(nodeA, nodeA.zorder))
#                print('-----------------------------')

            if (nodeA.isInsideQuickCheck(nodeB) and nodeA.verifyIsInside(nodeB)) \
               or nodeB.isBehind(nodeA):
#                print('-----------------------------')
#                print('node {} is behind node {}'.format(nodeB, nodeA))
#                print('   node {} zorder before: {}'.format(nodeA, nodeA.zorder))
#                print('   node {} zorder before: {}'.format(nodeB, nodeB.zorder))
                #nodeB.zorder = nodeA.zorder - 1
                self.zOrderGraph.add_edge(nodeB, nodeA)
                idx.append(nodeB.objectId)
#                print('   node {} zorder after: {}'.format(nodeA, nodeA.zorder))
#                print('   node {} zorder after: {}'.format(nodeB, nodeB.zorder))
#                print('-----------------------------')

        idx = np.unique(idx).astype('uint16')
        for index in idx:
            try:
                self.visualObjects[index].zorder = -1
            except:
                pass

        return idx


    def showChildrenGraph(self, title=''):
        if self.childrenGraph == None:
            self.buildGraph()

        fig = plt.figure()

        nx.draw_networkx(self.childrenGraph,
                         pos=nx.layout.random_layout(self.childrenGraph))

        plt.title(title)
        plt.tight_layout()
        fig.canvas.manager.window.raise_()


    def showZOrderGraph(self, title=''):
        if self.zOrderGraph == None:
            self.calculateZorders()

        fig = plt.figure()

        nx.draw_networkx(self.zOrderGraph,
                         pos=nx.layout.circular_layout(self.zOrderGraph))

        plt.title(title)
        plt.tight_layout()
        fig.canvas.manager.window.raise_()


    def showIntersectionsGraph(self, title=''):
        if self.intersectionsGraph == None:
            self.buildGraph()

        fig = plt.figure()

        nx.draw_networkx(self.intersectionsGraph,
                         pos=nx.layout.circular_layout(self.intersectionsGraph))

        plt.title(title)
        plt.tight_layout()
        fig.canvas.manager.window.raise_()


    def removeBorderObjects(self, binary_image):
        labeled = label(binary_image)
        regions = regionprops(labeled)

        for region in regions:
            if (0,0) in region.coords:
                bg_mask = (labeled == region.label)

                img_type = str(binary_image.dtype)

                if img_type[0:3] == 'int':
                    binary_image[bg_mask] = 0
                else:
                    binary_image[bg_mask] = False

                break

        return binary_image
