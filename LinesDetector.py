import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import probabilistic_hough_line
from geometry import Point, Line
from RtreeIndex import RtreeIndex
from copy import deepcopy

import networkx as nx

class LinesCollection(object):
    
    def __init__(self, linesList):
        self.lines = []
        self.visitedObjects = []
        self.graph = None
        
        if isinstance(linesList, list) or isinstance(linesList, LinesCollection):
            for line in linesList:
                self.lines.append(Line(line.pointA, line.pointB))
        else:
            line = linesList
            self.lines.append(Line(line.pointA, line.pointB))

    
    def copy(self):
        return LinesCollection(self.lines)
    
    
    def __del__(self):
        self.lines.clear()
        self.visitedObjects.clear()
        del self.lines
        del self.visitedObjects
    
    
    def remove(self, line):
        for i in range(len(self.lines)):
            if self.lines[i] == line:
                self.lines.remove(line)
                return self
        
        return self
    
    
    def append(self, line):
        self.lines.append(line)
        return self
    
    
    def __iter__(self):
        return self
    
    
    def __next__(self):
        for line in self.lines:
            if line not in self.visitedObjects:
                self.visitedObjects.append(line)
                return line
        
        # finished
        self.visitedObjects.clear()
        raise StopIteration
    
    
    def  __getitem__(self, index):
        return self.lines[index]
    
    
    def __contains__(self, value):
        for i in range(len(self.lines)):
            if self.lines[i] == value:
                return True
        
        return False
    
    
    def __len__(self):
        return len(self.lines)

    
    def getMBR(self):
        x_coords = []
        y_coords = []
        
        for line in self.lines:
            x_coords.append(line.pointA.x)
            x_coords.append(line.pointB.x)
            y_coords.append(line.pointA.y)
            y_coords.append(line.pointB.y)

        xmin = min(x_coords)
        ymin = min(y_coords)
        xmax = max(x_coords)
        ymax = max(y_coords)
        
        return (xmin,ymin,xmax,ymax)


    def merge(self):
        distance_thresh = 7
        
        linesCollection = self
                
        for line in linesCollection:
            rtreeIndex = RtreeIndex(linesCollection)
            
            neighbors = rtreeIndex.getNeighbors(line)
            
            for lineX in neighbors:
                if line.isParallelTo(lineX):
                    if (line.isIntersectingWith(lineX) 
                        or
                        line.getDistanceTo(lineX) <= distance_thresh):
                        line.mergeWith(lineX)
                        linesCollection.remove(lineX)
                        
        return self

    
    def getGraph(self):
        graph = nx.Graph()
        
        linesCollection = self
        
        rtreeIndex = RtreeIndex(linesCollection)
        
        for line in linesCollection:    
            neighbors = rtreeIndex.getNeighbors(line)
            
            for lineX in neighbors:
                if line.isParallelTo(lineX):
                    graph.add_weighted_edges_from([(line, lineX, 0)])
                    
                elif line.isPerpendicularTo(lineX):
                    graph.add_weighted_edges_from([(line, lineX, 90)])
                    
                else:
                    angle = round(line.getAngleWith(lineX))
                    graph.add_weighted_edges_from([(line, lineX, angle)])
                
        self.graph = graph
        return self.graph
    
    
    def showGraph(self, title=''):
        if self.graph == None:
            self.graph = self.getGraph()
        
        fig = plt.figure()
        
        layout = nx.circular_layout(self.graph)
        nx.draw_networkx_nodes(self.graph, pos=layout)
        nx.draw_networkx_edges(self.graph, pos=layout)
        
        labels = dict([((u,v,),d['weight'])
             for u,v,d in self.graph.edges(data=True)])
        
        nx.draw_networkx_edge_labels(self.graph, pos=layout, edge_labels=labels)
        
        plt.title(title)
        plt.tight_layout()        
        fig.canvas.manager.window.raise_()
        
    
    def show(self):
        mbr = self.getMBR()
        
        width = mbr[2] + 20
        height = mbr[3] + 20
                    
        image = np.ones((height, width))
        
        self.plotOnImage(image)
        
        
    def drawOnImage(self, image, value=1):
        for line in self:
            line.drawOnImage(image, value)
        
        return self
        

    def plotOnImage(self, image, title='', cmap='gray'):
        fig = plt.figure()
    
        plt.imshow(image, cmap)
        plt.axis('off')
        title = title + '\n{} lines'.format(len(self.lines))
        plt.title(title)
        plt.tight_layout()
        
        fig.canvas.manager.window.raise_()
        
        for line in self.lines:
            pA_y, pA_x = line.pointA.y, line.pointA.x
            pB_y, pB_x = line.pointB.y, line.pointB.x
            plt.plot((pA_x, pB_x), (pA_y, pB_y), linewidth=2)

        return self        
        
        
class LinesDetector(object):
    
    def __init__(self, contourObject):
        objtype = str(type(contourObject))
        
        if objtype[-15:-2] == "numpy.ndarray":
            self.image = contourObject
        else:
            self.image = contourObject.getImage()
        
        lines = probabilistic_hough_line(self.image,
                                     threshold=1,
                                     line_gap=4,
                                     line_length=7,
                                     theta=np.linspace(-np.pi/2, np.pi/2, 180))
        # lines = [   ((x0, y0), (x1, y1)),    ((x0, y0), (x1, y1)), ... ]
        # ... so we need to reverse the order,
        # in order to match numpy's format: (row, col) e.g. (y, x)
        
        lineObjects = []
        for i in range(len(lines)):
            pA, pB = lines[i]
            pA = Point(pA[1], pA[0])
            pB = Point(pB[1], pB[0])
            lineObjects.append(Line(pA, pB))


        # verification step (ensure detected lines are actually lines)
        linesCollection = LinesCollection(lineObjects)
#        for line in linesCollection:
#            pixels = line.getFromImage(self.image)
#            prob = np.count_nonzero(pixels) / len(pixels)
#            
#            if prob <= 0.75:
#                linesCollection.remove(line)
        
        self.linesCollection = linesCollection
    
                
    def getLines(self):
        return self.linesCollection
    
    
    def merge(self):
        self.linesCollection.merge()
        
#        distance_thresh = 5
#        
#        linesCollection = self.lines
#                
#        for line in linesCollection:
#            rtreeIndex = RtreeIndex(linesCollection)
#            
#            neighbors = rtreeIndex.getNeighbors(line)
            #neighbors.extend(rtreeIndex.getIntersections(line))
#            
#            LinesCollection(line).plotOnImage(self.image, title='object: line')
#            LinesCollection(neighbors).plotOnImage(self.image, title='object: neighbors')
#            cmdin = input('Continue? [y]/n: ')
#            if cmdin == 'y' or cmdin == '':
#                pass
#            else:
#                return False
#            
#            print('------------------')
#            print(line)
#            
#            for lineX in neighbors:
#                print('\n   ** neighbor **    ')
#                print('   ' + str(lineX))
#                if line.isParallelTo(lineX):
#                    print('   parallel')
#                    
#                    if (line.isIntersectingWith(lineX) 
#                        or
#                        line.getDistanceTo(lineX) <= distance_thresh):
#                        print('   ... and overlapping/colinear')
#                        line.mergeWith(lineX)
#                        linesCollection.remove(lineX)
#            print('------------------')

        return self
                 
    
    
    def show(self, title='', cmap='gray'):
        fig = plt.figure()
    
        plt.imshow(self.image, cmap)
        plt.axis('off')
        title = title + '\n{} lines'.format(len(self.linesCollection))
        plt.title(title)
        plt.tight_layout()
        
        fig.canvas.manager.window.raise_()
        
        for line in self.lines:
            pA_y, pA_x = line.pointA.y, line.pointA.x
            pB_y, pB_x = line.pointB.y, line.pointB.x
            plt.plot((pA_x, pB_x), (pA_y, pB_y), linewidth=2)

        return self
    

