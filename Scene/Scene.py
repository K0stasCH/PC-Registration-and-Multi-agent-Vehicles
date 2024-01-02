import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from Scene.Graph import Graph
from sklearn.cluster import HDBSCAN, DBSCAN
import copy

class Scene():
    def __init__(self, points3D, labelsPoints, palette, classes):
        self.palette = np.array(palette) / 255.0
        self.classes = classes
        self.labels = labelsPoints
        self.pcd= self._createPointCloud(points3D, labelsPoints)
        return

    def _createEgoVehicle(self, numPoints=1000):
        x_range = (-3, 1)
        y_range = (-1, 1)
        z_range = (0, 2)
        allX = np.random.uniform(x_range[0], x_range[1], numPoints) 
        allY = np.random.uniform(y_range[0], y_range[1], numPoints)
        allZ = np.random.uniform(z_range[0], z_range[1], numPoints)
        indx = self.classes.index('vehicle')
        colors = np.tile(self.palette[indx], (numPoints, 1))
        return np.vstack([allX, allY, allZ]).T, colors

    def _createPointCloud(self, points, labels):
        int_color = self.palette[labels.astype(int)]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.T)
        pcd.colors = o3d.utility.Vector3dVector(int_color)
        return pcd
    
    def visualizePCD(self):
        o3d.visualization.draw_geometries([self.pcd])
        return
    
    def _getPoints_idx(self, className:str='vehicle'):
        indx = self.classes.index(className)
        idxes =  np.where(self.labels == indx)[0]
        return idxes
    
    def createPCD(self, labels:list[str], cropDownLimits:list[float]=[None,None,-1.2]):
        '''
        create a new point cloud only with points labeled in list "labels"
        '''
        assert len(cropDownLimits) == 3
        
        classIndexes = [labels.index(label) for label in labels]
        pointsIndexes = np.where(np.isin(self.labels, classIndexes))[0]
        points3d = np.array(self.pcd.points)[pointsIndexes,:]

        points3d = filter(points3d, cropDownLimits)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points3d)

        return pcd
    
    def findcenters(self, 
                    _min_points, 
                    _eps,
                    className:str='vehicle',
                    visualize:bool=True,
                    cropDownLimits:list[float]=[None,None,-1.2]
                    ):
        indxes = self._getPoints_idx(className)
        points3d = np.array(self.pcd.points)[indxes,:]
        
        points3d = filter(points3d, cropDownLimits)

        if className=='vehicle':
            nodes = np.array([0,0,-0.5]) #initialize nodes with the ego-vehicle
        else:
            nodes = np.empty((0, 3))

        hdb = HDBSCAN(min_cluster_size=10, min_samples=None, cluster_selection_epsilon=0.5,
                      max_cluster_size=None, alpha=1.0, store_centers="centroid").fit(points3d)
        labeledCluster = hdb.labels_
        labeledCluster[hdb.probabilities_<0.95] = -1
        nodes =  np.vstack((nodes,hdb.centroids_))

        # db = DBSCAN(eps=_eps, min_samples=_min_points).fit(points3d)
        # labeledCluster = db.labels_

        #find the centers
        # max_label = labeledCluster.max()
        # for i in range(0,max_label+1):
        #     cluster_points = points3d[labeledCluster == i]
        #     center = np.mean(cluster_points, axis=0)
        #     nodes.append(center)
        # nodes = np.array(nodes)

        if visualize:
            fig, ax = plt.subplots(figsize=(18, 9), nrows=1, ncols=2)
            ax[0].scatter(points3d[:,0],points3d[:,1], s=2)

            max_label = labeledCluster.max()
            for i in range(0,max_label+1):
                indxes =  np.where(labeledCluster == i)[0]
                ax[1].scatter(points3d[indxes,0], points3d[indxes,1], s=2)

            ax[1].scatter(nodes[:,0], nodes[:,1], marker='x')
            plt.show()

        return nodes
    
    def generateGraph(self,
                      _min_points:int=17,
                      _eps:float=0.63,
                      classNames:[str]=['vehicle'],
                      cropDownLimits:list[float]=[None,None,-1.2]
                      ):
        nodes={}
        for className in classNames:
            nodes[className] = self.findcenters(className=className, visualize=False,
                                                 _min_points=_min_points, _eps=_eps,
                                                 cropDownLimits=cropDownLimits)
        return Graph(nodes, self.classes)


def filter(points3d:np.array, cropDownLimits:list[float]=[None,None,-1.2]):
    """
    delete the points that coordinates are smaller than the 'cropDownLimits'
    """
    assert points3d.shape[1] == 3
    assert len(points3d.shape) == 2
    assert len(cropDownLimits) == 3

    pointsCopy = copy.deepcopy(points3d)

    if cropDownLimits[0] is not None:
        pointsCopy = pointsCopy[pointsCopy[:,0] > cropDownLimits[0]]
    if cropDownLimits[1] is not None:
        pointsCopy = pointsCopy[pointsCopy[:,1] > cropDownLimits[1]]
    if cropDownLimits[2] is not None:
        pointsCopy = pointsCopy[pointsCopy[:,2] > cropDownLimits[2]]

    return pointsCopy