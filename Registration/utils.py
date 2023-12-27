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
    # """
    # a(N,3) and b(N,3)\n
    # b=M@a
    # """
    # assert a.shape[1] == 3 and a.ndim==2
    # assert b.shape[1] == 3 and b.ndim==2
    # assert a.shape[0] == b.shape[0]

    # ones = np.ones((a.shape[0], 1))

    # a = np.hstack((a, ones))
    # b = np.hstack((b, ones))
    # M = np.eye(4)
    # M[0,:] = np.linalg.lstsq(a, b[:,0], rcond=None)[0]
    # M[1,:] = np.linalg.lstsq(a, b[:,1], rcond=None)[0]
    # M[2,:] = np.linalg.lstsq(a, b[:,2], rcond=None)[0]
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
    matchingIdnx = matchMatrix_2_indxMatch(matchMatrix)
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

def matchMatrix_2_indxMatch(matchMatrix:np.array):
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

def plotMatching(g1:Graph, g2:Graph, matchMatrix:np.array, num_top_matches:int=3):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))

    g1._createPlot(ax[0])
    g2._createPlot(ax[1])
    matchingIdnx = matchMatrix_2_indxMatch(matchMatrix)
    for n, (i,j) in enumerate(matchingIdnx[:,:2]):
        if n<num_top_matches:
            _color = 'red'
        else:
            _color = 'black'

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

def draw_registration_result(target:Scene, source:Scene, transformation):
    """
    the transformations is applied to source PCD
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.pcd.paint_uniform_color([1, 0.706, 0])
    # target_temp.pcd.paint_uniform_color([0, 0.651, 0.929])
    source_temp.pcd.transform(transformation)

    o3d.visualization.draw_geometries([source_temp.pcd, target_temp.pcd])
    return

def performICP(source:Scene, target:Scene, trans_init:np.array, algorithm:str='p2p', threshold:float=0.5, sigma:float=0.1):
    if algorithm not in ['p2p', 'p2l']:
        raise Exception("Wrong algorith value. Must be 'p2p' or 'p2l'")
    
    classes = ['building', 'pole', 'traffic light', 'person', 'vehicle']
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    source_pcd = source_temp.createPCD(classes)
    target_pcd = target_temp.createPCD(classes)

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


# if __name__=='__main__':
#     testTransformation = np.array([[ 0.9914595, -0.0181429,  0.1291468, 10.       ],
#                                     [0.0329246,  0.9930198, -0.1132594,  5.       ],
#                                     [-0.1261905,  0.1165442,  0.9851362, 11.       ],
#                                     [ 0.       ,  0.       ,  0.       ,  1.       ]])
    
#     print(testTransformation)

#     p1_a = np.array([1,2,3])
#     p2_a = np.array([4,6,1])
#     p3_a = np.array([6,9,3])
#     p4_a = np.array([2,5,2])

#     p1_b_transformed_homogeneous = np.dot(testTransformation, np.append(p1_a, 1))
#     p2_b_transformed_homogeneous = np.dot(testTransformation, np.append(p2_a, 1))
#     p3_b_transformed_homogeneous = np.dot(testTransformation, np.append(p3_a, 1))
#     p4_b_transformed_homogeneous = np.dot(testTransformation, np.append(p4_a, 1))

#     p1_b = p1_b_transformed_homogeneous[:3] / p1_b_transformed_homogeneous[3]
#     p2_b = p2_b_transformed_homogeneous[:3] / p2_b_transformed_homogeneous[3]
#     p3_b = p3_b_transformed_homogeneous[:3] / p3_b_transformed_homogeneous[3]
#     p4_b = p4_b_transformed_homogeneous[:3] / p4_b_transformed_homogeneous[3]

#     a = np.vstack((p1_a,p2_a,p3_a,p4_a))
#     b = np.vstack((p1_b,p2_b,p3_b,p4_b))

#     M = findTransforamtionMatrix(a,b)
#     print(M)
#     print(1)
