import numpy as np
import matplotlib.pyplot as plt
from itertools import accumulate

class Graph():
    def __init__(self, nodes:dict(), allClassses):
        self.allClassses = allClassses
        self.indxName = [allClassses.index(x) for x in nodes.keys()]
        # self.startIndx = [0] + [nodes[x].shape[0] + self.startIndx[-1] for x in nodes.keys()]
        self.startIndx = [0] + [*accumulate([node.shape[0] for node in nodes.values()])]
        self.nodes3D = np.empty([0,3])
        for className in nodes.keys():
            self.nodes3D = np.vstack((self.nodes3D, nodes[className]))
             

        self.adjMatrix = self._generateAdjMatrix()

    def _generateAdjMatrix(self):
        points = self.nodes3D
        pairwise_distances = np.linalg.norm(points[:, np.newaxis] - points, axis=-1)
        return pairwise_distances
    
    def plotGraph(self):
        # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
        marker_styles = ['.', '+', 'v', '^', '<', '>', 's', '*', 'x', 'D']

        for i, idx in enumerate(self.indxName):
            start = self.startIndx[i]
            finish = self.startIndx[i+1]
            plt.scatter(self.nodes3D[start:finish,0], self.nodes3D[start:finish,1], label=self.allClassses[idx], marker=marker_styles[i%len(marker_styles)])
        plt.legend()
        plt.show()
        return
