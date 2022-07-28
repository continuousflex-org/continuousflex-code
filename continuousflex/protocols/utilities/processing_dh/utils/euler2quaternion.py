import numpy as np
from math import cos, sin, atan2, acos, radians, degrees
import torch
def eul2quat(arr, i):
    rot = radians(arr[i,0])
    tilt = radians(arr[i,1])
    psi = radians(arr[i,2])

    quaternion = [(cos(rot/2)*cos(tilt/2)*cos(psi/2))-(sin(rot/2)*cos(tilt/2)*sin(psi/2)),
                  (cos(rot/2)*sin(tilt/2)*sin(psi/2))-(sin(rot/2)*sin(tilt/2)*cos(psi/2)),
                  (cos(rot/2)*sin(tilt/2)*cos(psi/2))+(sin(rot/2)*sin(tilt/2)*sin(psi/2)),
                  (cos(rot/2)*cos(tilt/2)*sin(psi/2))+(sin(rot/2)*cos(tilt/2)*cos(psi/2))]
    return quaternion



def quater2euler(arr):
    qw = arr[0] 
    qx = arr[1]
    qy = arr[2]
    qz = arr[3]
    
    tilt = (qw**2)-(qx**2)-(qy**2)+(qz**2)
    if tilt > 1:
        tilt = 1.0
    elif tilt <-1:
        tilt = -1.0
    else:
        pass 
    
    euler = [degrees(atan2(2*((qy*qz)-(qw*qx)),2*((qx*qz)+(qw*qy)))),
             degrees(acos(tilt)),
             degrees(atan2(2*((qy*qz)+(qw*qx)),-2*((qx*qz)-(qw*qy))))]
    
    return euler


def quat2rotm(arr):

    q0 = arr[0] 
    q1 = arr[1]
    q2 = arr[2]
    q3 = arr[3]

    I = np.array([[1,0,0],[0,1,0],[0,0,1]])
    A = np.array([[0, -q3, q2],[q3, 0, -q1],[-q2, q1, 0]])

    rot_mat = I + 2 * q0 * A + 2 * A * A
    return rot_mat

"""
a = np.array([[134.81444 , 131.236146, 356.849227],[90, 90, 90]], dtype='float32')    

example = eul2quat(a,0)
print("quaternions ",example)


example2 = quater2euler(example)

print("angles in degrees",example2)
"""

