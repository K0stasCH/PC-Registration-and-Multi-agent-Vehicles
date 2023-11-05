import numpy as np
# from Vehicle.Vehicle import Vehicle

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

# def get_GT_tranfMatrix_Lidar(v1:Vehicle, v2:Vehicle):
#     pass

def findTransforamtionMatrix(a:np.array, b:np.array, size_4x4:bool=True):
    """
    a(N,3) and b(N,3)\n
    b=M@a
    """
    assert a.shape[1] == 3 and a.ndim==2
    assert b.shape[1] == 3 and b.ndim==2
    assert a.shape[0] == b.shape[0]

    ones = np.ones((a.shape[0], 1))

    a = np.hstack((a, ones))
    b = np.hstack((b, ones))
    M = np.eye(4)
    M[0,:] = np.linalg.lstsq(a, b[:,0], rcond=None)[0]
    M[1,:] = np.linalg.lstsq(a, b[:,1], rcond=None)[0]
    M[2,:] = np.linalg.lstsq(a, b[:,2], rcond=None)[0]

    if size_4x4:
        return M
    else:
        return _4x4_2_RotTransl(M)
    
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
