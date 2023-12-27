import numpy as np
import matplotlib.pyplot as plt
from itertools import accumulate
import copy

class Graph():
    def __init__(self, nodes:dict(), allClassses):
        """
        nodes : className-> list of 3d points, ....
        """
        self.allClassses = allClassses
        self.nodes = nodes #used only in combine graphs
        self.indxName = [allClassses.index(x) for x in nodes.keys()]
        # self.startIndx = [0] + [nodes[x].shape[0] + self.startIndx[-1] for x in nodes.keys()]
        self.startIndx = [0] + [*accumulate([node.shape[0] for node in nodes.values()])]
        self.nodes3D = np.empty([0,3])
        for className in nodes.keys():
            self.nodes3D = np.vstack((self.nodes3D, nodes[className]))
             
        self.adjMatrix = self._generateAdjMatrix()
        return
    
    def tranform_Points(self, m:np.array):
        assert m.shape == (4,4)

        g = copy.deepcopy(self)

        points = np.hstack((self.nodes3D,np.ones((self.nodes3D.shape[0],1)))).T
        points_T = m@points
        points_T = (points_T[:3,:]/points_T[3,:]).T
        g.nodes3D = points_T

        for key, value in g.nodes.items():
            value = np.hstack((value,np.ones((value.shape[0],1)))).T
            points_T = m@value
            points_T = (points_T[:3,:]/points_T[3,:]).T
            g.nodes[key] = points_T

        return g
    

    def _generateAdjMatrix(self):
        points = self.nodes3D
        pairwise_distances = np.linalg.norm(points[:, np.newaxis] - points, axis=-1)
        return pairwise_distances
    
    def _generateNodeFeat(self):
        feat = np.zeros((self.nodes3D.shape[0], len(self.allClassses)))
        for i, idx in enumerate(self.indxName):
            start = self.startIndx[i]
            finish = self.startIndx[i+1]
            feat[start:finish, idx] = 1
        return feat
    
    def _createPlot(self, axis:plt.axes):
        marker_styles = ['.', '+', 'v', '^', '<', '>', 's', '*', 'x', 'D']

        for i, idx in enumerate(self.indxName):
            start = self.startIndx[i]
            finish = self.startIndx[i+1]
            axis.scatter(self.nodes3D[start:finish,0],
                         self.nodes3D[start:finish,1],
                         label=self.allClassses[idx],
                         marker=marker_styles[i%len(marker_styles)])
        return

    def plotGraph(self):
        # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
    
        fig, ax = plt.subplots()
        self._createPlot(ax)

        plt.legend()
        plt.show()

        return fig, ax
    
   
def combineGrapghs(g1, g2):
    """
     not good results DONT USE IT
    """
    assert g1.allClassses == g2.allClassses

    g1_copy = copy.deepcopy(g1)
    g2_copy = copy.deepcopy(g2)

    combined_nodes = {}
    for key, value in g1_copy.nodes.items():
        combined_nodes[key] = value

    # Adding items from the second dictionary
    for key, value in g2_copy.nodes.items():
        # If the key already exists, append the value to a list
        if key in combined_nodes:
            combined_nodes[key] = np.vstack((combined_nodes[key], value))
        else:
            combined_nodes[key] = value
    
    g = Graph(combined_nodes, g1_copy.allClassses)
    return g