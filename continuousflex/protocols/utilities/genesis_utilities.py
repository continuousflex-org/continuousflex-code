import numpy as np
from pyworkflow.utils import runCommand
import pwem.emlib.metadata as md
import re
import multiprocessing

NUMBER_OF_CPU = int(np.min([multiprocessing.cpu_count(),4]))

EMFIT_NONE = 0
EMFIT_VOLUMES = 1
EMFIT_IMAGES = 2

FORCEFIELD_CHARMM = 0
FORCEFIELD_AAGO = 1
FORCEFIELD_CAGO = 2

SIMULATION_MIN = 0
SIMULATION_MD = 1
SIMULATION_NMMD = 2
SIMULATION_REMD = 3
SIMULATION_RENMMD = 4

PROGRAM_ATDYN = 0
PROGRAM_SPDYN= 1

INTEGRATOR_VVERLET = 0
INTEGRATOR_LEAPFROG = 1

IMPLICIT_SOLVENT_GBSA = 0
IMPLICIT_SOLVENT_NONE = 1

TPCONTROL_NONE = 0
TPCONTROL_LANGEVIN = 1
TPCONTROL_BERENDSEN = 2
TPCONTROL_BUSSI = 3

ENSEMBLE_NVT = 0
ENSEMBLE_NVE = 1
ENSEMBLE_NPT = 2

BOUNDARY_NOBC = 0
BOUNDARY_PBC = 1

ELECTROSTATICS_PME = 0
ELECTROSTATICS_CUTOFF = 1

NUCLEIC_NO = 0
NUCLEIC_RNA =1
NUCLEIC_DNA = 2

RB_PROJMATCH = 0
RB_WAVELET = 1

INPUT_TOPOLOGY = 0
INPUT_RESTART = 1
INPUT_NEW_SIM = 2

PROJECTION_ANGLE_SAME=0
PROJECTION_ANGLE_XMIPP=1
PROJECTION_ANGLE_IMAGE=2

def save_dcd(mol, coords_list, prefix):
    print("> Saving DCD trajectory ...")
    n_frames = len(coords_list)

    # saving PDBs
    mol = mol.copy()
    for i in range(n_frames):
        mol.coords = coords_list[i]
        mol.write_pdb("%s_frame%i.pdb" % (prefix, i))

    # VMD command
    with open(prefix+"_cmd.tcl", "w") as f :
        f.write("mol new %s_frame0.pdb\n" % prefix)
        for i in range(1,n_frames):
            f.write("mol addfile %s_frame%i.pdb\n" % (prefix, i))
        f.write("animate write dcd %s.dcd\n" % prefix)
        f.write('exit\n')

    # Running VMD
    runCommand("vmd -dispdev text -e %s_cmd.tcl" % prefix)

    # Cleaning
    for i in range(n_frames):
        runCommand("rm -f %s_frame%i.pdb\n" % (prefix, i))
    runCommand("rm -f %s_cmd.tcl" % prefix)
    print("\t Done \n")


def lastPDBFromDCD(inputPDB,inputDCD,  outputPDB):

    # EXTRACT PDB from dcd file
    with open("%s_tmp_dcd2pdb.tcl" % outputPDB, "w") as f:
        s = ""
        s += "mol load pdb %s dcd %s\n" % (inputPDB, inputDCD)
        s += "set nf [molinfo top get numframes]\n"
        s += "[atomselect top all frame [expr $nf - 1]] writepdb %s\n" % outputPDB
        s += "exit\n"
        f.write(s)
    runCommand("vmd -dispdev text -e %s_tmp_dcd2pdb.tcl" % outputPDB)

    # CLEAN TMP FILES
    runCommand("rm -f %s_tmp_dcd2pdb.tcl" % (outputPDB))

def buildParallelScript(commands,numberOfThreads=1,  raiseError=True):
    """
    :param list commands: list of commands to run in parallel
    :param numberOfThreads: Number of openMP threads
    :param raiseError: raise error if fails
    :return None:
    """

    py_script =\
        """
from mpi4py import MPI
import sys
import os
from subprocess import Popen
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

env = os.environ
env["OMP_NUM_THREADS"] = str(%i)

        """%numberOfThreads
    for i in range(len(commands)):
        py_script +=\
            """
if rank == %i:
    p = Popen("%s", shell=True, stdout=sys.stdout, stderr = sys.stderr, env=env)
    exitcode = p.wait()
    if exitcode != 0:
        err_msg = "Command returned with errors : %s"
        if %s :
            raise RuntimeError(err_msg)
        else:
            print(err_msg)
            """ % (i, commands[i], commands[i], "True" if raiseError else "False")


    py_script +=\
    """
exit(0)
    """
    return py_script





def pdb2vol(inputPDB, outputVol, sampling_rate, image_size):
    """
    Create a density volume from a pdb
    :param str inputPDB: input pdb file name
    :param str outputVol: output vol file name
    :param float sampling_rate: Sampling rate
    :param int image_size: Size of the output volume
    :return str: the Xmipp command to run
    """
    cmd = "xmipp_volume_from_pdb"
    args = "-i %s  -o %s --sampling %f --size %i %i %i --centerPDB"%\
           (inputPDB, outputVol,sampling_rate,image_size,image_size,image_size)
    return cmd+ " "+ args

def projectVol(inputVol, outputProj, expImage, sampling_rate=5.0, angular_distance=-1, compute_neighbors=True):
    """
    Create a set of projections from an input volume
    :param str inputVol: Input volume file name
    :param str outputProj: Output set of proj file name
    :param str expImage: Experimental image to project in the neighborhood
    :param float sampling_rate: Samplign rate
    :param float angular_distance: Do not search a distance larger than...
    :param bool compute_neighbors: Compute projection nearby the experimental image
    :return str: the Xmipp command to run
    """
    cmd = "xmipp_angular_project_library"
    args = "-i %s.vol -o %s.stk --sampling_rate %f " % (inputVol, outputProj, sampling_rate)
    if compute_neighbors :
        args +="--compute_neighbors --angular_distance %f " % angular_distance
        args += "--experimental_images %s "%expImage
        if angular_distance != -1 :
            args += "--near_exp_data"
    return cmd+ " "+ args

def projectMatch(inputImage, inputProj, outputMeta):
    """
    Projection matching of an input experimental image with a set of projections
    :param str inputImage: File name of the input experimental image
    :param str inputProj: File name of the input set of projections
    :param str outputMeta: File name of the output Xmipp metadata file with the angles of the matching
    :return str: the Xmipp command to run
    """
    cmd = "xmipp_angular_projection_matching "
    args= "-i %s -o %s --ref %s.stk "%(inputImage, outputMeta, inputProj)
    args +="--search5d_shift 10.0 --search5d_step 1.0"
    return cmd + " "+ args

def waveletAssignement(inputImage, inputProj, outputMeta):
    """
    Make a discrete angular assignment of angles from a set of projections
    :param str inputImage: File name of input experimental image
    :param str inputProj: File name of the input set of projections
    :param str outputMeta: File name of the output Xmipp metadata file with the angles assigned
    :return str: the Xmipp command to run
    """
    cmd = "xmipp_angular_discrete_assign "
    args= "-i %s -o %s --ref %s.doc "%(inputImage, outputMeta, inputProj)
    args +="--psi_step 5.0 --max_shift_change 10.0 --search5D"
    return cmd + " "+ args

def continuousAssign(inputMeta, inputVol, outputMeta):
    """
    Make a continuous angular assignment of angles from a volume
    :param str inputMeta: File name of input Xmipp metadata file with angles to assign
    :param str inputVol: File name of the input Volume
    :param str outputMeta: File name of the output Xmipp metadata file with the angles
    :return str: the Xmipp command to run
    """
    cmd = "xmipp_angular_continuous_assign "
    args= "-i %s -o %s --ref %s.vol "%(inputMeta, outputMeta, inputVol)
    return cmd + " "+ args

def flipAngles(inputMeta, outputMeta):
    """
    Flip angles from Xmipp representation to Euler angles
    :param str inputMeta : File name of input Xmipp metadata file containing the angles to flip
    :param str outputMeta: file name of the output Xmipp matadata file
    :return None:
    """
    Md1 = md.MetaData(inputMeta)
    flip = Md1.getValue(md.MDL_FLIP, 1)
    tilt1 = Md1.getValue(md.MDL_ANGLE_TILT, 1)
    psi1 = Md1.getValue(md.MDL_ANGLE_PSI, 1)
    x1 = Md1.getValue(md.MDL_SHIFT_X, 1)
    if flip:
        Md1.setValue(md.MDL_SHIFT_X, -x1, 1)
        Md1.setValue(md.MDL_ANGLE_TILT, tilt1 + 180, 1)
        Md1.setValue(md.MDL_ANGLE_PSI, -psi1, 1)
    Md1.write(outputMeta)

# def getAngularDist(md1, md2, idx1=1, idx2=1):
#     rot1        = md1.getValue(md.MDL_ANGLE_ROT, int(idx1))
#     tilt1       = md1.getValue(md.MDL_ANGLE_TILT, int(idx1))
#     psi1        = md1.getValue(md.MDL_ANGLE_PSI, int(idx1))
#     rot2        = md2.getValue(md.MDL_ANGLE_ROT, int(idx2))
#     tilt2       = md2.getValue(md.MDL_ANGLE_TILT, int(idx2))
#     psi2        = md2.getValue(md.MDL_ANGLE_PSI, int(idx2))
#
#     return  SymList.computeDistanceAngles(SymList(), rot1, tilt1, psi1, rot2, tilt2, psi2, False, True, False)
#
#
# def getShiftDist(md1, md2, idx1=1, idx2=1):
#     shiftx1 = md1.getValue(md.MDL_SHIFT_X, int(idx1))
#     shifty1 = md1.getValue(md.MDL_SHIFT_Y, int(idx1))
#     shiftx2 = md2.getValue(md.MDL_SHIFT_X, int(idx2))
#     shifty2 = md2.getValue(md.MDL_SHIFT_Y, int(idx2))
#     return np.linalg.norm(np.array([shiftx1, shifty1, 0.0]) - np.array([shiftx2, shifty2, 0.0]))

def getAngularShiftDist(angle1MetaFile, angle2MetaData, angle2Idx, tmpPrefix, symmetry):

    mdImgTmp = md.MetaData()
    mdImgTmp.addObject()
    mdImgTmp.setValue(md.MDL_ANGLE_ROT, angle2MetaData.getValue(md.MDL_ANGLE_ROT, angle2Idx), 1)
    mdImgTmp.setValue(md.MDL_ANGLE_TILT,angle2MetaData.getValue(md.MDL_ANGLE_TILT,angle2Idx), 1)
    mdImgTmp.setValue(md.MDL_ANGLE_PSI, angle2MetaData.getValue(md.MDL_ANGLE_PSI, angle2Idx), 1)
    mdImgTmp.setValue(md.MDL_SHIFT_X,   angle2MetaData.getValue(md.MDL_SHIFT_X,   angle2Idx), 1)
    mdImgTmp.setValue(md.MDL_SHIFT_Y,   angle2MetaData.getValue(md.MDL_SHIFT_Y,   angle2Idx), 1)
    mdImgTmp.write(tmpPrefix + ".xmd")

    cmd = "xmipp_angular_distance --ang1 %s --ang2 %s.xmd --oroot %sDist --sym %s --check_mirrors > %s.log" % \
          (angle1MetaFile, tmpPrefix, tmpPrefix, symmetry, tmpPrefix)
    runCommand(cmd)
    with open(tmpPrefix + ".log", "r") as f:
        for line in f:
            if "angular" in line:
                angDist = float(re.findall("\d+\.\d+", line)[0])
            if "shift" in line:
                shftDist = float(re.findall("\d+\.\d+", line)[0])
    return angDist, shftDist


def readLogFile(log_file):
    with open(log_file,"r") as file:
        header = None
        dic = {}
        for line in file:
            if line.startswith("INFO:"):
                if header is None:
                    header = line.split()
                    for i in range(1,len(header)):
                        dic[header[i]] = []
                else:
                    splitline = line.split()
                    if len(splitline) == len(header):
                        for i in range(1,len(header)):
                            try :
                                dic[header[i]].append(float(splitline[i]))
                            except ValueError:
                                pass

    return dic

def dcd2numpyArr(filename):
    print("> Reading dcd file %s"%filename)
    BYTESIZE = 4
    with open(filename, 'rb') as f:

        # Header
        # ---------------- INIT

        start_size = int.from_bytes((f.read(BYTESIZE)), "little")
        crd_type = f.read(BYTESIZE).decode('ascii')
        nframe = int.from_bytes((f.read(BYTESIZE)), "little")
        start_frame = int.from_bytes((f.read(BYTESIZE)), "little")
        len_frame = int.from_bytes((f.read(BYTESIZE)), "little")
        len_total = int.from_bytes((f.read(BYTESIZE)), "little")
        for i in range(5):
            f.read(BYTESIZE)
        time_step = np.frombuffer(f.read(BYTESIZE), dtype=np.float32)
        for i in range(9):
            f.read(BYTESIZE)
        charmm_version = int.from_bytes((f.read(BYTESIZE)), "little")

        end_size = int.from_bytes((f.read(BYTESIZE)), "little")

        if end_size != start_size:
            raise RuntimeError("Can not read dcd file")

        # ---------------- TITLE
        start_size = int.from_bytes((f.read(BYTESIZE)), "little")
        ntitle = int.from_bytes((f.read(BYTESIZE)), "little")
        tilte_rd = f.read(BYTESIZE*20 * ntitle)
        try :
            title = tilte_rd.encode("ascii")
        except AttributeError:
            title = str(tilte_rd)
        end_size = int.from_bytes((f.read(BYTESIZE)), "little")

        if end_size != start_size:
            raise RuntimeError("Can not read dcd file")

        # ---------------- NATOM
        start_size = int.from_bytes((f.read(BYTESIZE)), "little")
        natom = int.from_bytes((f.read(BYTESIZE)), "little")
        end_size = int.from_bytes((f.read(BYTESIZE)), "little")

        if end_size != start_size:
            raise RuntimeError("Can not read dcd file")

        # ----------------- DCD COORD
        dcd_arr =  np.zeros((nframe, natom, 3), dtype=np.float32)
        for i in range(nframe):
            for j in range(3):

                start_size = int.from_bytes((f.read(BYTESIZE)), "little")
                while (start_size != BYTESIZE * natom and start_size != 0):
                    # print("\n-- UNKNOWN %s -- " % start_size)

                    f.read(start_size)
                    end_size = int.from_bytes((f.read(BYTESIZE)), "little")
                    if end_size != start_size:
                        raise RuntimeError("Can not read dcd file")
                    start_size = int.from_bytes((f.read(BYTESIZE)), "little")

                bin_arr = f.read(BYTESIZE * natom)
                if len(bin_arr) == BYTESIZE * natom:
                    dcd_arr[i, :, j] = np.frombuffer(bin_arr, dtype=np.float32)
                else:
                    break
                end_size = int.from_bytes((f.read(BYTESIZE)), "little")
                if end_size != start_size:
                    if i>1:
                        break
                    else:
                        # pass
                        raise RuntimeError("Can not read dcd file %i %i " % (start_size, end_size))

        print("\t -- Summary of DCD file -- ")
        print("\t\t crd_type  : %s"%crd_type)
        print("\t\t nframe  : %s"%nframe)
        print("\t\t len_frame  : %s"%len_frame)
        print("\t\t len_total  : %s"%len_total)
        print("\t\t time_step  : %s"%time_step)
        print("\t\t charmm_version  : %s"%charmm_version)
        print("\t\t title  : %s"%title)
        print("\t\t natom  : %s"%natom)
    print("\t Done \n")

    return dcd_arr


def numpyArr2dcd(arr, filename, start_frame=1, len_frame=1, time_step=1.0, title=None):
    print("> Wrinting dcd file %s"%filename)
    BYTESIZE = 4
    nframe, natom, _ = arr.shape
    len_total=nframe*len_frame
    charmm_version=24
    if title is None:
        title = "DCD file generated by Continuous Flex plugin"
    ntitle = (len(title)//(20*BYTESIZE)) + 1
    with open(filename, 'wb') as f:
        zeroByte = int.to_bytes(0, BYTESIZE, "little")

        # Header
        # ---------------- INIT
        f.write(int.to_bytes(21*BYTESIZE ,BYTESIZE, "little"))
        f.write(b'CORD')
        f.write(int.to_bytes(nframe, BYTESIZE, "little"))
        f.write(int.to_bytes(start_frame, BYTESIZE, "little"))
        f.write(int.to_bytes(len_frame, BYTESIZE, "little"))
        f.write(int.to_bytes(len_total, BYTESIZE, "little"))
        for i in range(5):
            f.write(zeroByte)
        f.write(np.float32(time_step).tobytes())
        for i in range(9):
            f.write(zeroByte)
        f.write(int.to_bytes(charmm_version, BYTESIZE, "little"))

        f.write(int.to_bytes(21*BYTESIZE,BYTESIZE, "little"))

        # ---------------- TITLE
        f.write(int.to_bytes((ntitle*20+1)*BYTESIZE ,BYTESIZE, "little"))
        f.write(int.to_bytes(ntitle ,BYTESIZE, "little"))
        f.write(title.ljust(20*BYTESIZE).encode("ascii"))
        f.write(int.to_bytes((ntitle*20+1)*BYTESIZE ,BYTESIZE, "little"))

        # ---------------- NATOM
        f.write(int.to_bytes(BYTESIZE ,BYTESIZE, "little"))
        f.write(int.to_bytes(natom ,BYTESIZE, "little"))
        f.write(int.to_bytes(BYTESIZE ,BYTESIZE, "little"))

        # ----------------- DCD COORD
        for i in range(nframe):
            for j in range(3):
                f.write(int.to_bytes(BYTESIZE*natom, BYTESIZE, "little"))
                f.write(np.float32(arr[i, :, j]).tobytes())
                f.write(int.to_bytes(BYTESIZE*natom, BYTESIZE, "little"))
    print("\t Done \n")

def existsCommand(name):
    from shutil import which
    return which(name) is not None
