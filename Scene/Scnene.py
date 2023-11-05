import open3d as o3d
import numpy as np
# from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.colors as colors
# from collections import Counter
from Scene.Graph import Graph

class Scene():
    def __init__(self, points3D, labelsPoints, palette, classes):
        self.palette = np.array(palette) / 255.0
        self.classes = classes
        self.labels = labelsPoints
        self.pcd= self._createPointCloud(points3D, labelsPoints)
        # x = self.findcenters('vehicle')
        # self.VoxelSize = 2
        # self.voxel_grid = self._voxelize(self.VoxelSize)
        # self.pcd = o3d.geometry.PointCloud()
        # self.pcd.points = o3d.utility.Vector3dVector(points3D.T)
        # x = pcd.points
        # pcd.labels = 
        # self.resolution = (50,50,2)    #the num of voxels in each axis ,(x,y,z)

        # self.pcd = np.asarray(points3D)
        # self.labels = np.asarray(labelsPoints, dtype=np.int8)
        
        # x= self.pcd.colors[0]
        # self.visualizePCD()
        # self.AABB = self._compute_AABB()
        # self.starts, self.steps = self._findSteps(self.resolution)
        # self.occupancyLookUp=self._construct_Occupancy_LookUp()  #dict
        
        # self.voxelsData = np.random.randint(-1, len(classes), size=(10, 10, 10)) #[low,high) , no label-> -1

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
    
    def _clustering(self, _min_points:int, _eps:float, className:str='vehicle', visualize:bool=False):
        indxes = self._getPoints_idx(className)

        points3d = np.array(self.pcd.points)[indxes,:]
        subPCD = o3d.geometry.PointCloud()
        subPCD.points = o3d.utility.Vector3dVector(points3d)

        labeledCluster = np.array(subPCD.cluster_dbscan(eps=_eps, min_points=_min_points, print_progress=False))

        if visualize:
            max_label = labeledCluster.max()
            for i in range(0,max_label+1):
                indx =  np.where(labeledCluster == i)[0]
                plt.scatter(points3d[indx,0], points3d[indx,1])
            plt.show()

        return subPCD, labeledCluster
    
    def findcenters(self, _min_points, _eps, className:str='vehicle', visualize:bool=False):
        subPCD, labeled_subPCD = self._clustering(_min_points, _eps, className, visualize)

        max_label = labeled_subPCD.max()
        if className=='vehicle':
            nodes = [np.array([0,0,-0.5])] #initialize nodes with the ego-vehicle
        else:
            nodes = []
        for i in range(0,max_label+1):
            indx =  np.where(labeled_subPCD == i)[0]
            points3d = np.array(subPCD.points)[indx,:]
            x = np.mean(points3d[:, 0])
            y = np.mean(points3d[:, 1])
            z = np.mean(points3d[:, 2])
            nodes.append(np.array([x, y, z]))
        
        nodes = np.array(nodes)

        if visualize:
            plt.scatter(nodes[:,0], nodes[:,1])
            plt.show()
        return nodes
    
    def generateGraph(self, _min_points:int=17, _eps:float=0.63, classNames:[str]=['vehicle']):
        nodes={}
        for className in classNames:
            nodes[className] = self.findcenters(className=className, visualize=False, _min_points=_min_points, _eps=_eps)
        return Graph(nodes, self.classes)

    
    # def visualizeVoxels(self):
    #     o3d.visualization.draw_geometries([self.voxel_grid])
    #     return
    
    # def getVoxels(self):
    #     return self.voxel_grid.get_voxels()
    
    # def _voxelize(self, voxelSize):
    #     egoPoints, egoColors = self._createEgoVehicle()

    #     oldPoints = np.array(self.pcd.points)
    #     oldColors = np.array(self.pcd.colors)
    #     newPoints = np.vstack([oldPoints, egoPoints])
    #     newColors = np.vstack([oldColors, egoColors])

    #     newPCD = o3d.geometry.PointCloud()
    #     newPCD.points = o3d.utility.Vector3dVector(newPoints)
    #     newPCD.colors = o3d.utility.Vector3dVector(newColors)

    #     voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(newPCD, voxel_size=voxelSize)
    #     return voxel_grid
        


    # def visualizePCD(self):
        # vis = o3d.visualization.Visualizer()
        # vis.create_window(
        #     window_name='Segmented Scene',
        #     width=960,
        #     height=540,
        #     left=480,
        #     top=270)
        # vis.get_render_option().background_color = [0.0, 0.0, 0.0]
        # vis.get_render_option().point_size = 5
        # vis.get_render_option().show_coordinate_frame = True
        # vis.add_geometry(self.pcd)
        # o3d.visualization.draw_plotly([self.pcd])
        # return

    # def _compute_AABB(self):
    #     min_x, min_y, min_z = np.min(self.pcd, axis=1)
    #     max_x, max_y, max_z = np.max(self.pcd, axis=1)
    #     aabb = {'min_x': min_x,
    #             'min_y': min_y,
    #             'min_z': min_z,
    #             'max_x': max_x,
    #             'max_y': max_y,
    #             'max_z': max_z}
    #     return aabb
    
    # def _findSteps(self, numVoxels):
    #     """
    #     find the step size (or voxel size) given the numbers of the voxels per axis.
    #     return: the values that "starts" each voxel and the "step"
    #     """
    #     xStarts, xStep = np.linspace(self.AABB['min_x'], self.AABB['max_x'], numVoxels[0], retstep=True)
    #     yStarts, yStep = np.linspace(self.AABB['min_y'], self.AABB['max_y'], numVoxels[1], retstep=True)
    #     zStarts, zStep = np.linspace(self.AABB['min_z'], self.AABB['max_z'], numVoxels[2], retstep=True)
    #     starts = (xStarts, yStarts, zStarts)
    #     steps = (xStep, yStep, zStep)
    #     return starts, steps
    
    # def _construct_Occupancy_LookUp(self):
    #     grid = {} #key the indexes, for example 2_4_1:list() ->A[2,4,1]=list() 
    #     for i,point in enumerate(self.pcd.T):
    #         ind_voxel = self._findVoxel(point)
    #         key = "%d_%d_%d" % tuple(ind_voxel)
    #         value = grid.get(key, None)
    #         if value is not None:
    #             grid[key] += [i]
    #         else:
    #              grid[key] = [i]
    #     return grid

    # def _findVoxel(self, point):
    #     pStart = np.array([self.AABB['min_x'], self.AABB['min_y'], self.AABB['min_z']])
    #     pFinish = np.array([self.AABB['max_x'], self.AABB['max_y'], self.AABB['max_z']])
    #     return (point - pStart)//np.array(self.steps)
    
    # def voxelize(self, voxelSize):
        # pcd = o3d.geometry.PointCloud( )
        # octree.convert_from_point_cloud(self.pc, voxel_size=10)
        # octree.v

        # kdtree = KDTree(self.pc)
        # # kdtree.q

        # xStarts, xStep = np.linspace(self.AABB['min_x'], self.AABB['max_x'], self.resolution[0], retstep=True)
        # yStarts, yStep = np.linspace(self.AABB['min_y'], self.AABB['max_y'], self.resolution[1], retstep=True)
        # zStarts, zStep = np.linspace(self.AABB['min_z'], self.AABB['max_z'], self.resolution[2], retstep=True)

        # data  = np.full(self.resolution, -1)
        # for voxel in self.occupancyLookUp:
        #     pointsIndx = self.occupancyLookUp[voxel]
        #     labels = self.labels[pointsIndx]
        #     appeareanse = Counter(labels)
        #     mostCommon = appeareanse.most_common(1)[0]
        #     # indxVoxel = voxel.split('_')
        #     indxVoxel = [int(i) for i in voxel.split('_')]
        #     data[indxVoxel[0], indxVoxel[1], indxVoxel[2]] = mostCommon[0]


    # def visualize(self):
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')

    #     colorArray = np.empty_like(self.voxelsData, dtype=object)

    #     data_ = self.voxelsData + 1 #elements with zero value dont plotting
    #     palette_ = [None] + self.palette
    #     for i in range(self.voxelsData.shape[0]):
    #         for j in range(self.voxelsData.shape[1]):
    #             for k in range(self.voxelsData.shape[2]):
    #                 voxelValue = data_[i, j, k]
    #                 if voxelValue == 0: #considering empty voxels have value: -1
    #                     colorArray[i, j, k] = '#00000000'
    #                 else:
    #                     color = palette_[voxelValue]
    #                     colorArray[i, j, k] = '#%02x%02x%02xff' % tuple(color)

    #     ax.voxels(data_, facecolors=colorArray)
    #     plt.xlabel('x-axis')
    #     plt.ylabel('y-axis')
       
    #     # plt.legend(labels=self.labels)
    #     plt.show()
    #     return fig, ax