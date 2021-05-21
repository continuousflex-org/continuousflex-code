# This function converts a TomBox table (motive list) into xmipp (scipion) metadata file
# given a metadata input containing a list of the subtomograms
from pwem.emlib import metadata as md
import numpy as np
from math import sin, cos
from xmippLib import Euler_matrix2angles


def TomboxRotationMatrix(phi, psi, theta):
    rotMat = np.zeros([3, 3])
    phi = np.deg2rad(phi)
    psi = np.deg2rad(psi)
    theta = np.deg2rad(theta)

    rotMat[0,0] = cos(psi) * cos(phi) - cos(theta) * sin(psi) * sin(phi)
    rotMat[0,1] = sin(psi) * cos(phi) + cos(theta) * cos(psi) * sin(phi)
    rotMat[0,2] = sin(theta) * sin(phi)
    rotMat[1,0] = -cos(psi) * sin(phi) - cos(theta) * sin(psi) * cos(phi)
    rotMat[1,1] = -sin(psi) * sin(phi) + cos(theta) * cos(psi) * cos(phi)
    rotMat[1,2] = sin(theta) * cos(phi)
    rotMat[2,0] = sin(theta) * sin(psi)
    rotMat[2,1] = -sin(theta) * cos(psi)
    rotMat[2,2] = cos(theta)
    return rotMat


def motivelist2metadata(mtlist, mdfi, mdfo):
    motlist = np.transpose(np.genfromtxt(mtlist, delimiter=','))
    md_motlist = md.MetaData(mdfi)
    counter = 0
    for line in motlist:
        counter += 1
        cc = float(line[0])
        # name = 'import_' + str(int(line[3])).zfill(3) + '.vol'
        shiftx = float(line[13])
        shifty = float(line[14])
        shiftz = float(line[15])
        phi = float(line[16])
        psi = float(line[17])
        theta = float(line[18])
        # Conversion of angles:
        a, b, c = Euler_matrix2angles(TomboxRotationMatrix(phi, psi, theta))
        rot = -c
        tilt = -b
        psi = -a
        # Writing the results:
        # md_motlist.setValue(md.MDL_IMAGE, name, md_motlist.addObject())
        md_motlist.setValue(md.MDL_MAXCC, cc, counter)
        md_motlist.setValue(md.MDL_SHIFT_X, shiftx, counter)
        md_motlist.setValue(md.MDL_SHIFT_Y, shifty, counter)
        md_motlist.setValue(md.MDL_SHIFT_Z, shiftz, counter)
        md_motlist.setValue(md.MDL_ANGLE_ROT, rot, counter)
        md_motlist.setValue(md.MDL_ANGLE_TILT, tilt, counter)
        md_motlist.setValue(md.MDL_ANGLE_PSI, psi, counter)

    md_motlist.write(mdfo)


# a test here:
# mtlist = '/home/guest/Downloads/motl1.csv'
# mdfni = '/home/guest/ScipionUserData/projects/TestXmipp2Mltomo/Runs/000172_FlexProtSubtomogramAveraging/extra/volumes.xmd'
# mdfno = '/home/guest/ScipionUserData/projects/TestXmipp2Mltomo/Runs/000172_FlexProtSubtomogramAveraging/extra/volumes2.xmd'
# motivelist2metadata(mtlist, mdfni,mdfno)
