import numpy as np
from sklearn.preprocessing import StandardScaler

def read_pdb(path, ca = False):
    with open(path, 'r') as f:
        lines = f.readlines()
    pdb_list = []
    for line in lines:
        line = line[:38] + " " + line[38:]
        line = line[:47] + " " + line[47:]
        line = line[:62] + " " + line[62:]
        if ca:
            if line.startswith("ATOM") and " CA " in line:
                line = line.split()
                pdb_list.append(line[6:9])
            else:
                pass
        else:
            if line.startswith("ATOM"):
                line = line.split()
                pdb_list.append(line[6:9])
            else:
                pass

    # pythonic way
    #lines = [line.split for line in lines]
    #lines = [line for line in lines if "ATOM" in line]
    #if ca:
    #    lines = [line for line in lines if " CA " in line]
    #lines
    pdb_array = np.array(pdb_list, dtype='float32')
    coords = pdb_array
    return coords
    
def parse_pdb(path, atom_p, i):
    """
    ----------
    This function generates the predicted pdb.

    Parameters
    ----------
    path : string path to the reference structure.
    atom_p: numpy array of the predicted atom positions.

    Returns
    -------
    """
    with open(path, 'r') as f:
            lines = f.readlines()
    lines = [line for line in lines if 'ATOM' in line]
    atom = [line[0:4] for line in lines]
    serial = ['{:>7}'.format(line[4:12].strip()) for line in lines]
    atom_name = ['{:>4}'.format(line[12:17].strip()) for line in lines]
    loc_ind = ['{:>5}'.format(line[17:20].strip()) for line in lines]
    residue_name = ['{:>2}'.format(line[21:22].strip()) for line in lines]
    chain_id = ['{:>4}'.format(line[23:28].strip()) for line in lines]
    occupancy = ['{:>6}'.format(line[56:60].strip()) for line in lines]
    temperature = ['{:>6}'.format(line[60:67].strip()) for line in lines]
    elem_symb = ['{:>12}'.format(line[77:80].strip()) for line in lines]
    with open('synth'+str(i)+'.pdb','w') as f:
        for i in range(len(lines)):
            pos = '{:>12.6}{:>8.6}{:>8.5}'.format(atom_p[i,0],atom_p[i,1],atom_p[i,2])
            line = atom[i]+serial[i]+atom_name[i]+loc_ind[i]+residue_name[i]+chain_id[i]+\
                   pos+occupancy[i]+temperature[i]+elem_symb[i]+'\n'
            f.write(line)
            
def standard_pdb(coords):
    return (coords-np.mean(coords,axis=(0)))/np.std(coords,axis=(0))
