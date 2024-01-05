import numpy as np
from Vehicle.Vehicle import Vehicle
from pyquaternion import Quaternion
from Scene.Graph import Graph
from Scene.Scene import Scene
import pygmtools as pygm
import functools
import matplotlib.pyplot as plt
import copy
import open3d as o3d
from scipy.spatial.transform import Rotation
from matplotlib.colors import LinearSegmentedColormap


def _RotTransl_2_4x4(rot:np.array, transl:np.array):
    assert rot.shape == (3,3)
    assert transl.size == 3
    M = np.eye(4)
    M[:3,:3] = rot
    M[:3,3] = transl.reshape(-1, 3)
    return M

def _4x4_2_RotTransl(a:np.array):
    assert a.shape == (4,4)
    r = a[:3,:3]
    t = a[:3,3]
    return r, t

def get_GT_tranfMatrix_Lidar2World(v1:Vehicle, time:int):
    lidar_V1_pos = np.array(v1.calibParam['LIDAR']['translation'])
    lidar_V1_rot = Quaternion(v1.calibParam['LIDAR']['rotation']).rotation_matrix

    v1_pos = np.array(v1.egoTranslation_Stream[time])
    v1_rot = Quaternion(v1.egoRotation_Stream[time]).rotation_matrix

    M = _RotTransl_2_4x4(v1_rot, v1_pos) @ _RotTransl_2_4x4(lidar_V1_rot, lidar_V1_pos) 
    return M

def rotMat2angle(rotMat:np.array, angularType:str='radians'):
    if angularType not in ['radians', 'degrees']:
        raise Exception("Wrong angularType value. Must be 'radians' or 'degrees'")
    assert rotMat.shape == (3,3)
    assert np.allclose(np.linalg.det(rotMat), 1.0)

    rotation = Rotation.from_matrix(rotMat)
    euler_angles = rotation.as_euler('xyz')

    if angularType=='degrees':
        euler_angles = np.degrees(euler_angles)

    return euler_angles

def compareTransformation(m1:np.array, m2:np.array, angularType:str='radians'):
    """
    returns the translation, angles difference
    """
    assert m1.shape == (4,4)
    assert m2.shape == (4,4)

    r1, t1= _4x4_2_RotTransl(m1)
    r2, t2= _4x4_2_RotTransl(m2)

    angles1 = rotMat2angle(r1, angularType)
    angles2 = rotMat2angle(r2, angularType)

    translation_difference = t2 - t1
    angles_difference = angles2 - angles1

    return translation_difference, angles_difference 

def get_GT_tranfMatrix_V1_2_V2(v1:Vehicle, v2:Vehicle, time:int):
    '''
    v1 is in the origin (0,0) and tranform the V2 in the coordinates of the V1
    '''
    m1 = get_GT_tranfMatrix_Lidar2World(v1, time)
    m2 = get_GT_tranfMatrix_Lidar2World(v2, time)
    
    return np.linalg.inv(m1) @ m2

def tranformPoints(points3D:np.array, matrix:np.array):
    '''
    PointsTransformed = matrix @ points\n
    points(n,3)
    '''
    assert matrix.shape == (4,4) and matrix.ndim == 2
    assert points3D.shape[1] == 3 and points3D.ndim == 2

    points3D = np.hstack((points3D, np.ones((points3D.shape[0],1)) )).T

    pointsTransformed = matrix @ points3D
    pointsTransformed = pointsTransformed[:3,:] / pointsTransformed[3,:]

    return pointsTransformed

def findTransforamtionMatrix(a:np.array, b:np.array, size_4x4:bool=True):
    a = a.T
    b = b.T
    assert a.shape == b.shape

    num_rows, num_cols = a.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = b.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(a, axis=1)
    centroid_B = np.mean(b, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = a - centroid_A
    Bm = b - centroid_B

    H = Am @ np.transpose(Bm)

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    if size_4x4:
        return _RotTransl_2_4x4(R,t)
    else:
        return R,t

def getTransforamtionMatrix_from_MatchingMatrix(g1:Graph, g2:Graph, matchMatrix:np.array, num_top_matches:int=3):
    """
    transform the coordinates of g2 to g1 : (g2->g1)
    return an 4x4 matrix
    """
    assert num_top_matches >= 3
    matchingIdnx = _matchMatrix_2_indxMatch(matchMatrix)
    matchingIdnx = np.asarray(matchingIdnx, dtype=int)

    points_1 = g1.nodes3D[matchingIdnx[:num_top_matches,0],:]
    points_2 = g2.nodes3D[matchingIdnx[:num_top_matches,1],:]
    T = findTransforamtionMatrix(points_2, points_1, True)
    return T

def matchGraph(g1:Graph, g2:Graph):
    pygm.BACKEND = 'numpy'

    n1 = np.array([g1.nodes3D.shape[0]])
    n2 = np.array([g2.nodes3D.shape[0]])

    conn1, edge1_feat = pygm.utils.dense_to_sparse(g1.adjMatrix)
    conn2, edge2_feat = pygm.utils.dense_to_sparse(g2.adjMatrix)

    node1_feat = g1._generateNodeFeat()
    node2_feat = g2._generateNodeFeat()

    gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.0) # set affinity function
    affinityMatrix = pygm.utils.build_aff_mat(node1_feat, edge1_feat, conn1, 
                                              node2_feat, edge2_feat, conn2, 
                                              n1, None, n2, None, edge_aff_fn=gaussian_aff)
    
    softMatch = pygm.rrwm(affinityMatrix, n1, n2, max_iter=50, sk_iter=30, alpha=0.2, beta=20)

    return affinityMatrix, softMatch

def _matchMatrix_2_indxMatch(matchMatrix:np.array):
    '''
    return an (n,3) array sorted by 'score' from high to low\n
    [nodei_g1, nodej_g2, score]
            ...
    '''
    m = hardMatch(matchMatrix)
    g1_nodes, g2_nodes = np.where(m == 1)
    scores = matchMatrix[g1_nodes, g2_nodes]

    match = np.vstack((g1_nodes, g2_nodes, scores)).T
    match = match[match[:, 2].argsort()[::-1]] #sort the results (descending order by score)
    return match

def hardMatch(matchMatrix:np.array):
    return pygm.hungarian(matchMatrix)

def plotMatching(g1:Graph, g2:Graph, matchMatrix:np.array, top_n=None):
    """
    pltAll: False=Render all matches, True=Only the top n 
    """
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))

    g1._createPlot(ax[0])
    g2._createPlot(ax[1])
    matchingIdnx = _matchMatrix_2_indxMatch(matchMatrix)
    
    maxConf = matchingIdnx[0,2]
    minConf = matchingIdnx[-1,2]

    green = (0,1,0)
    red = (1,0,0)
    cmap = LinearSegmentedColormap.from_list('custom_cmap', [red, green], N=256)
    cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=minConf, vmax=maxConf)),
                    cax=cax, orientation='vertical')
    cbar.set_label('Confidence')
    

    for n, (i,j, conf) in enumerate(matchingIdnx[:,:]):
        coeff = (conf - minConf)/(maxConf-minConf)
        _color = np.array(green)*(coeff) + np.array(red)*(1-coeff)

        if n > top_n and n!=None:
            break

        ax[1].annotate('',
                       xy=(g1.nodes3D[int(i),0], g1.nodes3D[int(i),1]), xycoords=ax[0].transData,
                       xytext=(g2.nodes3D[int(j),0], g2.nodes3D[int(j),1]), textcoords=ax[1].transData,
                       arrowprops=dict(arrowstyle="<->", color=_color))
        
    plt.show()
    
    return

def transform_Graph(ref:Graph, transformed:Graph, matchMatrix:np.array):
    T = getTransforamtionMatrix_from_MatchingMatrix(ref, transformed, matchMatrix)
    g = transformed.tranform_Points(T)
    return g

def draw_registration_result(target:Scene, source:Scene, transformation, annotations:bool=True):
    """
    the transformations is applied to source PCD
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    if annotations==False:
        source_temp.pcd.paint_uniform_color([1, 0.706, 0])
        target_temp.pcd.paint_uniform_color([0, 0.651, 0.929])
    source_temp.pcd.transform(transformation)

    o3d.visualization.draw_geometries([source_temp.pcd, target_temp.pcd])
    return

def performICP(source:Scene, target:Scene, trans_init:np.array, algorithm:str='p2p', threshold:float=0.5, sigma:float=0.1):
    if algorithm not in ['p2p', 'p2l']:
        raise Exception("Wrong algorith value. Must be 'p2p' or 'p2l'")
    
    classes = ['building', 'pole', 'traffic light', 'person', 'vehicle']
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    source_pcd = source_temp.createPCD(classes, cropDownLimits=[None, None, -1.2])
    target_pcd = target_temp.createPCD(classes, cropDownLimits=[None, None, -1.2])

    # o3d.visualization.draw_geometries([target_pcd])

    if algorithm == 'p2l':
        source_pcd.estimate_normals()
        target_pcd.estimate_normals()
        loss = o3d.pipelines.registration.TukeyLoss(k=sigma)
        function = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
    if algorithm == 'p2p':
        function = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    reg = o3d.pipelines.registration.registration_icp(source_pcd, target_pcd,
                                                      threshold, trans_init,
                                                      function)
    return reg

def printErrors(gt_TransformationMatrix:np.array, estimated_TransformationMatrix:np.array, angularType:str='radians'):
    assert gt_TransformationMatrix.shape == estimated_TransformationMatrix.shape
    assert gt_TransformationMatrix.shape == (4,4)

    t, r = compareTransformation(gt_TransformationMatrix, estimated_TransformationMatrix, angularType)

    errorT = np.linalg.norm(t)
    # r = np.abs(r)
    print(f'Translation Error:{errorT:.2f}\tAxis-X Error:{r[0]:.2f}\tAxis-Y Error:{r[1]:.2f}\tAxis-Z Error:{r[2]:.2f}')
    # print(np.linalg.norm(t))
    return errorT, r[0], r[1], r[2]