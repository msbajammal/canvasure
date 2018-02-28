from rtree import index as rtree_index
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt



class RtreeIndex(object):
    
    def __init__(self, list_SpatialObject):
        #SOs = deepcopy(list_SpatialObject)
        SOs = list_SpatialObject
        
        self.objects = SOs
        self.index = rtree_index.Index(interleaved=True)
        self.ids = []
        
        for i in range(len(SOs)):
            self.ids.append(i)
            self.index.insert(i, SOs[i].getMBR(), obj=SOs[i])

    @staticmethod
    def isMbrAInsideB(mbr_A, mbr_B):            
        xmin_A, ymin_A, xmax_A, ymax_A = mbr_A
        xmin_B, ymin_B, xmax_B, ymax_B = mbr_B
    
        if (xmin_A >= xmin_B and
            ymin_A >= ymin_B and
            xmax_A <= xmax_B and
            ymax_A <= ymax_B):    
            return True
        else:
            return False

    @staticmethod
    def areMBRsIntersecting(spatialObjectA, spatialObjectB):
        if isinstance(spatialObjectA, tuple) and len(spatialObjectA) == 4:
            xmin_A, ymin_A, xmax_A, ymax_A = spatialObjectA
        else:
            xmin_A, ymin_A, xmax_A, ymax_A = spatialObjectA.getMBR()

        if isinstance(spatialObjectB, tuple) and len(spatialObjectB) == 4:
            xmin_B, ymin_B, xmax_B, ymax_B = spatialObjectB
        else:
            xmin_B, ymin_B, xmax_B, ymax_B = spatialObjectB.getMBR()
        
        r1_left   = xmin_A
        r1_right  = xmax_A
        r1_bottom = ymin_A
        r1_top    = ymax_A
        
        r2_left   = xmin_B
        r2_right  = xmax_B
        r2_bottom = ymin_B
        r2_top    = ymax_B       
        
        hoverlaps = True
        voverlaps = True
        if (r1_left > r2_right) or (r1_right < r2_left):
            hoverlaps = False
        if (r1_top < r2_bottom) or (r1_bottom > r2_top):
            voverlaps = False
            
        return hoverlaps and voverlaps
    

    def getIntersections(self, querySpatialObject):
        intersections = list(self.index.intersection(querySpatialObject.getMBR(), objects=True))
        # each item in intersections has: item.id, item.object, item.bbox
        
        SOs = []
        for item in intersections:
            SOs.append(item.object)
        
        return SOs
    
    
    def getNeighbors(self, querySpatialObject, numNeighbors=4):
        neighbors = list(self.index.nearest(querySpatialObject.getMBR(), numNeighbors, objects=True))
        
        SOs = []
        for item in neighbors:
            SOs.append(item.object)
        
        return SOs
        
        
    def copy(self):        
        return RtreeIndex(self.objects)
            

    def insert(self, spatialObject):
        
        i = max(self.ids)
        self.objects.append(deepcopy(spatialObject))
        self.ids.append(i)
        
        self.index.insert(i, spatialObject.getMBR(), obj=spatialObject)
        
        return self
    
    
    def delete(self, obj_or_id):
                               
        if isinstance(obj_or_id, int):
            _id = obj_or_id
            idx = self.ids.index(_id)
            _obj = self.objects[idx]
                            
        else: #SpatialObject
            _obj = obj_or_id
            idx = self.objects.index(_obj)
            _id = self.ids[idx]
            

        _coords = _obj.getMBR()

        self.index.delete(_id, _coords)
        self.ids.remove(_id)
        self.objects.remove(_obj)
        
        return self
    
    
    def show(self, title='', cmap='gray'):
        entire_bbox = self.index.bounds
        # (xmin, ymin, xmax, ymax)
        width = int(entire_bbox[2]) + 10
        height = int(entire_bbox[3]) + 10
        
        self.image = np.zeros( (height, width) )
        
        for spatialObject in self.objects:
            spatialObject.drawOnImage(self.image)
        
        fig = plt.figure()
    
        plt.imshow(self.image, cmap)
        plt.axis('off')
        plt.title(title)
        plt.tight_layout()
        
        # bring it to front
        fig.canvas.manager.window.raise_()
        return self
        
        