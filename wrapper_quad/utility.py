import numpy as np
import math
# Return the rotation through Euler angles 
# R = Rx(alpha)*Ry(beta)*Rz(gamma)
def GetRotationMatrix(eulerAngles):
    alpha, beta, gamma = eulerAngles
    rx  =   np.array([[1,   0,  0],
                                        [0, math.cos(alpha), -math.sin(alpha)],
                                        [0, math.sin(alpha),  math.cos(alpha)]
                                    ])
    ry  =   np.array([[math.cos(beta), 0, math.sin(beta)],
                                        [0             , 1, 0             ],
                                        [-math.sin(beta),0, math.cos(beta)]
                                    ])
    rz  =   np.array([[math.cos(gamma), -math.sin(gamma), 0],
                                        [math.sin(gamma),  math.cos(gamma), 0],
                                        [0              , 0,                1]
                                    ])
    R   =   np.dot(rz, np.dot(ry, rx))

    return R


def DecodeRotationMatrix(flat_rotmat):
    """ 
            Decode flatten rotation matrix
            @flat_rotmat: Flatten rotation matrix:
            flat_rotmat[6]: -sin(b)
            flat_rotmat[6]: cos(b)sin(a)

            In hovering alpha & beta, must be between -90º to +90ª,
            so, if we apply arcsin ton sinbeta, this must be in I or IV quadrant [-90,+90ª]
            In case beta is greater than 90ª, other solution must be found
            
            Result interesting to just apply sin(angle) to compute the action of the MPC
    """
    sinbeta =   flat_rotmat[6]
    beta    =   np.arcsin(sinbeta)
    """ IF cos(beta) * cos(alpha) is < 0, 
                then the quadrotor is in reverse way 
            ELSE: 
                quadrotor is in hovery case, or else in extreme reverse way (probabli unrecoverably)
            
            so we consider just when quadrotor is in hover space
    """

    cbca    =   flat_rotmat[8]
    cbsa    =   flat_rotmat[7]
    if cbca > 0.0:
        #ca  =   cbca/np.cos(beta)
        sa  =   cbsa/np.cos(beta)
        alpha   =   np.arcsin(sa)
    elif cbca <=0.0:
        pass



def GetFlatRotationMatrix(eulerAngles):
    rotmat  =   GetRotationMatrix(eulerAngles)

    return np.reshape(rotmat, (9, ))
    #return np.append(rotmat[0,:], rotmat[1:3,:])