import numpy as np
from struct import pack
import torch
import multiprocessing as multiprocessing

def readPDB(fnIn):
    with open(fnIn) as f:
        lines = f.readlines()
    return lines


def PDB2List(lines):
    newlines = []
    for line in lines:
        if line.startswith("ATOM "):
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                newline = [x, y, z]
                newlines.append(newline)
            except:
                pass
    return newlines


def pdb_to_array(fnIn):
    return torch.Tensor(PDB2List(readPDB(fnIn)))


def euler_matrix(rot, tilt, psi):
    from math import sin, cos, radians
    t1 = -torch.deg2rad(psi)
    t2 = -torch.deg2rad(tilt)
    t3 = -torch.deg2rad(rot)
    a11 = torch.cos(t1) * torch.cos(t2) * torch.cos(t3) - torch.sin(t1) * torch.sin(t3)
    a12 = -torch.cos(t3) * torch.sin(t1) - torch.cos(t1) * torch.cos(t2) * torch.sin(t3)
    a13 = torch.cos(t1) * torch.sin(t2)
    a21 = torch.cos(t1) * torch.sin(t3) + torch.cos(t2) * torch.cos(t3) * torch.sin(t1)
    a22 = torch.cos(t1) * torch.cos(t3) - torch.cos(t2) * torch.sin(t1) * torch.sin(t3)
    a23 = torch.sin(t1) * torch.sin(t2)
    a31 = -torch.cos(t3) * torch.sin(t2)
    a32 = torch.sin(t2) * torch.sin(t3)
    a33 = torch.cos(t2)
    T = torch.tensor([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
    return T


def projectPDBPixel(pdb_coordinates, s, t, sigma):
    # Auxilary function, not meant to be used standalone
    row_size, column_size = pdb_coordinates.shape
    ng = row_size
    # print(row_size,column_size)
    sum_of_gaussians = 0
    sl_v = pdb_coordinates[:,0]
    tl_v = pdb_coordinates[:,1]
    s_v = torch.ones_like(sl_v)*s
    t_v = torch.ones_like(tl_v)*t
    
    exp_arg = -((s_v-sl_v)**2 + (t_v-tl_v)**2)/(2*sigma**2)
    exp_vec = torch.exp(exp_arg)
    return torch.sum(exp_vec, dim = 2)
def projectPDB_NP(PDB, size, sampling_rate =1, sigma=1, rot=0, tilt=0, psi=0, shift_x=0, shift_y=0, shift_z=0):
    #T = torch.linalg.inv(euler_matrix(torch.tensor(rot, dtype=torch.float), torch.tensor(tilt, dtype=torch.float), torch.tensor(psi, dtype=torch.float)))
    #PDB = torch.matmul(PDB, T)/sampling_rate
    PDB = PDB/sampling_rate
    shifts = torch.ones_like(PDB)
    shifts[:, 0] = shifts[:, 0] * shift_x
    shifts[:, 1] = shifts[:, 1] * shift_y
    shifts[:, 2] = shifts[:, 2] * shift_z
    PDB += shifts
    
    PDB = (PDB - torch.mean(PDB) )
    
    projection = torch.zeros([size, size])
    
    limit = int(size/2)
    l = torch.arange(-limit, limit, 1)
    y, x = torch.meshgrid(l, l)
    x = x.unsqueeze(2)
    xx = x.repeat(1,1,PDB.shape[0])
    y = y.unsqueeze(2)
    yy = y.repeat(1,1,PDB.shape[0])
    res = projectPDBPixel(PDB, xx, yy, sigma)
    return res

def projectPDB2Image(PDB, size, sampling_rate =1, sigma=1, rot=0, tilt=0, psi=0, shift_x=0, shift_y=0, shift_z=0):
    # This function gives the same views as xmipp_phantom_project, however, it performs the projection by representing
    # atoms by 3D Gaussians

    # PDB: array of atomic coordinates
    # size: image size that you want
    # sampling rate: the sampling rate on the image
    # sigma: the Gaussian size for each atom (here the projection is with fixed Gaussian)
    # rot, tilt, psi: Euler angles of the view
    # shift_x, shift_y, shift_z: the shifting applied to the atomic structure before projection
    T = torch.linalg.inv(euler_matrix(torch.tensor(rot, dtype=torch.float), torch.tensor(tilt, dtype=torch.float), torch.tensor(psi, dtype=torch.float)))
    PDB = torch.matmul(PDB, T.to('cpu')) / sampling_rate
    shifts = torch.ones_like(PDB)
    shifts[:, 0] = shifts[:, 0] * shift_x
    shifts[:, 1] = shifts[:, 1] * shift_y
    shifts[:, 2] = shifts[:, 2] * shift_z
    PDB += shifts

    # projection = np.zeros([size, size])
    limit = int(size/2)
    l = torch.arange(-limit, limit, 1)
    x, y = torch.meshgrid(l, l)
    xv = torch.reshape(x, (-1,))
    yv = torch.reshape(y, (-1,))
    ps = [(xv[i], yv[i]) for i in range(len(xv))]
    
    global segment
    def segment(p):
        return projectPDBPixel(PDB, p[0], p[1], sigma)
    with multiprocessing.Pool(20) as pool:
        values = pool.map(segment,ps)
    values = torch.reshape(list(values),[size,size])
    return values


def PDBVoxel(pdb_coordinates,r, s, t, sigma, sampling_rate):
    # Auxilary function, not meant to be used standalone
    row_size, column_size = pdb_coordinates.shape
    ng = row_size
    # print(row_size,column_size)
    rl_v = pdb_coordinates[:,0]/sampling_rate
    sl_v = pdb_coordinates[:,1]/sampling_rate
    tl_v = pdb_coordinates[:,2]/sampling_rate
    r_v = np.ones_like(rl_v)*r
    s_v = np.ones_like(sl_v)*s
    t_v = np.ones_like(tl_v)*t
    exp_arg = -((r_v-rl_v)**2 + (s_v-sl_v)**2 + (t_v-tl_v)**2)/(2*sigma**2)
    exp_vec = np.exp(exp_arg)
    return np.sum(exp_vec)


def PDB2Volume(PDB, volume_size, sigma, sampling_rate):
    # This function gives the same as xmipp_volume_from_pdb, however, it represents atoms by 3D Gaussians

    # PDB: array of atomic coordinates
    # volume_size: output volume size
    # sampling rate: the sampling rate
    # sigma: the Gaussian size for each atom

    limit = int(volume_size/2)
    z, y, x = np.mgrid[-limit:limit, -limit:limit, -limit:limit]
    xv = np.reshape(x, -1)
    yv = np.reshape(y, -1)
    zv = np.reshape(z, -1)
    ps = [(xv[i], yv[i], zv[i]) for i in range(len(xv))]

    global segment

    def segment(p):
        return PDBVoxel(PDB, p[0], p[1], p[2], sigma, sampling_rate)
    with multiprocessing.Pool(processes = 8) as pool:
        values = pool.map(segment,ps)
    values = np.reshape(list(values),[volume_size,volume_size,volume_size])
    return values


def save_volume(vol, filename):
    vol = np.float32(vol)
    # From the spider format:
    labels = {i - 1: v for i, v in [
        (1, 'NZ'),
        (2, 'NY'),
        (3, 'IREC'),
        (5, 'IFORM'),
        (6, 'IMAMI'),
        (7, 'FMAX'),
        (8, 'FMIN'),
        (9, 'AV'),
        (10, 'SIG'),
        (12, 'NX'),
        (13, 'LABREC'),
        (14, 'IANGLE'),
        (15, 'PHI'),
        (16, 'THETA'),
        (17, 'GAMMA'),
        (18, 'XOFF'),
        (19, 'YOFF'),
        (20, 'ZOFF'),
        (21, 'SCALE'),
        (22, 'LABBYT'),
        (23, 'LENBYT'),
        (24, 'ISTACK/MAXINDX'),
        (26, 'MAXIM'),
        (27, 'IMGNUM'),
        (28, 'LASTINDX'),
        (31, 'KANGLE'),
        (32, 'PHI1'),
        (33, 'THETA1'),
        (34, 'PSI1'),
        (35, 'PHI2'),
        (36, 'THETA2'),
        (37, 'PSI2'),
        (38, 'PIXSIZ'),
        (39, 'EV'),
        (40, 'PROJ'),
        (41, 'MIC'),
        (42, 'NUM'),
        (43, 'GLONUM'),
        (101, 'PSI3'),
        (102, 'THETA3'),
        (103, 'PHI3'),
        (104, 'LANGLE')]}

    # Inverse.
    locations = {v: k for k, v in labels.items()}
    """Save volume vol into a file, with the spider format."""
    nx, ny, nz = vol.shape
    fields = [0.0] * nx
    values = {
        'NZ': nz, 'NY': ny,
        'IREC': 3,  # number of records (including header records)
        'IFORM': 3,  # 3D volume
        'FMAX': vol.max(), 'FMIN': vol.min(),
        'AV': vol.mean(), 'SIG': vol.std(),
        'NX': nx,
        'LABREC': 1,  # number of records in file header (label)
        'SCALE': 1,
        'LABBYT': 4 * nx,  # number of bytes in header
        'LENBYT': 4 * nx,  # record length in bytes (only 1 in our header)
    }
    for label, value in values.items():
        fields[locations[label]] = float(value)
    header = pack('%df' % nx, *fields)
    with open(filename, 'wb') as f:
        f.write(header)
        vol.tofile(f)
        
