import numpy as np
import os
import copy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from Bio.SVDSuperimposer import SVDSuperimposer
from pyworkflow.utils import runCommand
import pwem.emlib.metadata as md

class PDBMol:
    def __init__(self, pdb_file):
        """
        Contructor
        :param pdb_file: PDB file
        """
        atom = []
        atomNum = []
        atomName = []
        resName = []
        resAlter = []
        chainName = []
        resNum = []
        coords = []
        occ = []
        temp = []
        chainID = []
        elemName = []
        print("> Reading pdb file %s ..." % pdb_file)
        with open(pdb_file, "r") as f:
            for line in f:
                spl = line.split()
                if len(spl) > 0:
                    if (spl[0] == 'ATOM'):  # or (hetatm and spl[0] == 'HETATM'):
                        l = [line[:6], line[6:11], line[12:16], line[16], line[17:20], line[21], line[22:26],
                             line[30:38],
                             line[38:46], line[46:54], line[54:60], line[60:66], line[72:76], line[76:78]]
                        l = [i.strip() for i in l]
                        atom.append(l[0])
                        atomNum.append(l[1])
                        atomName.append(l[2])
                        resAlter.append(l[3])
                        resName.append(l[4])
                        chainName.append(l[5])
                        resNum.append(l[6])
                        coords.append([float(l[7]), float(l[8]), float(l[9])])
                        occ.append(l[10])
                        temp.append(l[11])
                        chainID.append(l[12])
                        elemName.append(l[13])
        print("\t Done \n")

        atomNum = np.array(atomNum)
        atomNum[np.where(atomNum == "*****")[0]] = "-1"

        self.atom = np.array(atom, dtype='<U6')
        self.n_atoms = len(self.atom)
        self.atomNum = np.array(atomNum).astype(int)
        self.atomName = np.array(atomName, dtype='<U4')
        self.resName = np.array(resName, dtype='<U4')
        self.resAlter = np.array(resAlter, dtype='<U1')
        self.chainName = np.array(chainName, dtype='<U1')
        self.resNum = np.array(resNum).astype(int)
        self.coords = np.array(coords).astype(float)
        self.occ = np.array(occ).astype(float)
        self.temp = np.array(temp).astype(float)
        self.chainID = np.array(chainID, dtype='<U4')
        self.elemName = np.array(elemName, dtype='<U2')

    def save(self, file):
        """
        Save to PDB Format
        :param file: pdb file path
        """
        print("> Saving pdb file %s ..." % file)
        with open(file, "w") as file:
            past_chainName = self.chainName[0]
            past_chainID = self.chainID[0]
            for i in range(len(self.atom)):
                if past_chainName != self.chainName[i] or past_chainID != self.chainID[i]:
                    past_chainName = self.chainName[i]
                    past_chainID = self.chainID[i]
                    file.write("TER\n")

                atom = self.atom[i].ljust(6)  # atom#6s
                if self.atomNum[i] == -1 or self.atomNum[i] >= 100000:
                    atomNum = "99999"  # aomnum#5d
                else:
                    atomNum = str(self.atomNum[i]).rjust(5)  # aomnum#5d
                atomName = self.atomName[i].ljust(4)  # atomname$#4s
                resAlter = self.resAlter[i].ljust(1)  # resAlter#1
                resName = self.resName[i].ljust(4)  # resname#1s
                chainName = self.chainName[i].rjust(1)  # Astring
                resNum = str(self.resNum[i]).rjust(4)  # resnum
                coordx = str('%8.3f' % (float(self.coords[i][0]))).rjust(8)  # x
                coordy = str('%8.3f' % (float(self.coords[i][1]))).rjust(8)  # y
                coordz = str('%8.3f' % (float(self.coords[i][2]))).rjust(8)  # z\
                occ = str('%6.2f' % self.occ[i]).rjust(6)  # occ
                temp = str('%6.2f' % self.temp[i]).rjust(6)  # temp
                chainID = str(self.chainID[i]).ljust(4)  # elname
                elemName = str(self.elemName[i]).rjust(2)  # elname
                file.write("%s%s %s%s%s%s%s    %s%s%s%s%s      %s%s\n" % (
                atom, atomNum, atomName, resAlter, resName, chainName, resNum,
                coordx, coordy, coordz, occ, temp, chainID, elemName))
            file.write("END\n")
        print("\t Done \n")

    def select_atoms(self, idx):
        self.coords = self.coords[idx]
        self.n_atoms = self.coords.shape[0]
        self.atom = self.atom[idx]
        self.atomNum = self.atomNum[idx]
        self.atomName = self.atomName[idx]
        self.resName = self.resName[idx]
        self.resAlter = self.resAlter[idx]
        self.chainName = self.chainName[idx]
        self.resNum = self.resNum[idx]
        self.elemName = self.elemName[idx]
        self.occ = self.occ[idx]
        self.temp = self.temp[idx]
        self.chainID = self.chainID[idx]

    def get_chain(self, chainName):
        if not isinstance(chainName, list):
            chainName=[chainName]
        chainidx =[]
        for i in chainName:
            idx = np.where(self.chainName == i)[0]
            if len(idx) == 0:
                idx= np.where(self.chainID == i)[0]
            chainidx = chainidx + list(idx)
        return np.array(chainidx)

    def select_chain(self, chainName):
        self.select_atoms(self.get_chain(chainName))

    def copy(self):
        return copy.deepcopy(self)

    def remove_alter_atom(self):
        idx = []
        for i in range(self.n_atoms):
            if self.resAlter[i] != "":
                print("!!! Alter residue %s for atom %i"%(self.resName[i], self.atomNum[i]))
                if self.resAlter[i] == "A":
                    idx.append(i)
                    self.resAlter[i]=""
            else:
                idx.append(i)
        self.select_atoms(idx)

    def remove_hydrogens(self):
        idx=[]
        for i in range(self.n_atoms):
            if not self.atomName[i].startswith("H"):
                idx.append(i)
        self.select_atoms(idx)

    def alias_atom(self, atomName, atomNew, resName=None):
        n_alias = 0
        for i in range(self.n_atoms):
            if self.atomName[i] == atomName:
                if resName is not None :
                    if self.resName[i] == resName :
                        self.atomName[i] = atomNew
                        n_alias+=1
                else:
                    self.atomName[i] = atomNew
                    n_alias+=1
        print("%s -> %s : %i lines changed"%(atomName, atomNew, n_alias))

    def alias_res(self, resName, resNew):
        n_alias=0
        for i in range(self.n_atoms):
            if self.resName[i] == resName :
                self.resName[i] = resNew
                n_alias+=1
        print("%s -> %s : %i lines changed"%(resName ,resNew,  n_alias))


    def add_terminal_res(self):
        aa = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU", "MET", "ASN", "PRO",
              "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"]
        past_chainName = self.chainName[0]
        past_chainID = self.chainID[0]
        for i in range(self.n_atoms-1):
            if past_chainName != self.chainName[i+1] or past_chainID != self.chainID[i+1]:
                if self.resName[i] in aa :
                    print("End of chain %s ; adding terminal residue to %s %i %s"%
                          (past_chainID,self.resName[i],self.resNum[i],self.atomName[i]))
                    resNum = self.resNum[i]
                    j=0
                    while self.resNum[i-j] ==resNum :
                        self.resName[i - j] += "T"
                        j+=1
                else:
                    print("End of chain %s %s %i"% (past_chainID,self.resName[i],self.resNum[i]))
                past_chainName = self.chainName[i+1]
                past_chainID = self.chainID[i+1]


        i = self.n_atoms-1
        if self.resName[i] in aa:
            print("End of chain %s ; adding terminal residue to %s %i %s" % (
            past_chainID, self.resName[i], self.resNum[i], self.atomName[i]))
            resNum = self.resNum[i]
            j = 0
            while self.resNum[i - j] == resNum:
                self.resName[i - j] += "T"
                j += 1
        else:
            print("End of chain %s %s %i" % (past_chainID, self.resName[i], self.resNum[i]))


    def check_res_order(self):
        chains = list(set(self.chainID))
        chains.sort()
        new_idx = []
        for c in chains:
            chain_idx = self.get_chain(c)
            resNumlist = list(set(self.resNum[chain_idx]))
            resNumlist.sort()
            for i in range(len(resNumlist)):
                idx = np.where(self.resNum[chain_idx] == resNumlist[i])[0]
                new_idx += list(chain_idx[idx])
        self.select_atoms(np.array(new_idx))

    def atom_res_reorder(self):
        chains = list(set(self.chainID))
        chains.sort()

        # reorder atoms and res
        for c in chains:
            chain_idx = self.get_chain(c)
            past_resNum = self.resNum[chain_idx[0]]
            resNum = 1
            for i in range(len(chain_idx)):
                if self.resNum[chain_idx[i]] != past_resNum:
                    past_resNum = self.resNum[chain_idx[i]]
                    resNum += 1
                self.resNum[chain_idx[i]] = resNum
                self.atomNum[chain_idx[i]] = i + 1

    def allatoms2ca(self):
        new_idx = []
        for i in range(self.n_atoms):
            if self.atomName[i] == "CA" or self.atomName[i] == "P":
                new_idx.append(i)
        return np.array(new_idx)

def matchPDBatoms(mols, ca_only=False):
    print("> Matching PDBs atoms ...")
    n_mols = len(mols)


    if mols[0].chainID[0] in mols[1].chainID:
        chaintype = 1
    elif mols[0].chainName[0] in mols[1].chainName:
        chaintype = 0
    else:
        raise RuntimeError("\t Warning : No matching chains")

    ids = []
    ids_idx = []
    for m in mols :
        id_tmp=[]
        id_idx_tmp=[]
        for i in range(m.n_atoms):
            if (not ca_only) or m.atomName[i] == "CA" or m.atomName[i] == "P":
                if chaintype == 0 :
                    id_tmp.append("%s_%i_%s_%s"%(m.chainName[i], m.resNum[i], m.resName[i] , m.atomName[i]))
                else:
                    id_tmp.append("%s_%i_%s_%s"%(m.chainID[i], m.resNum[i], m.resName[i] , m.atomName[i]))
                id_idx_tmp.append(i)
        ids.append(np.array(id_tmp))
        ids_idx.append(np.array(id_idx_tmp))

    idx = []
    for i in range(len(ids[0])):
        idx_line = [ids_idx[0][i]]
        for m in range(1,n_mols):
            idx_tmp = np.where(ids[0][i] == ids[m])[0]
            if len(idx_tmp) == 1:
                idx_line.append(ids_idx[m][idx_tmp[0]])
            elif len(idx_tmp) > 1:
                print("\t Warning : One atom in mol#0 is matching several atoms in mol#%i : "%m)

        if len(idx_line) == n_mols :
            idx.append(idx_line)

    if len(idx)==0:
        print("\t Warning : No matching coordinates")
    print("\t Done")

    return np.array(idx)

NUCLEIC_NO = 0
NUCLEIC_RNA =1
NUCLEIC_DNA = 2

FORCEFIELD_CHARMM = 0
FORCEFIELD_AAGO = 1
FORCEFIELD_CAGO = 2

def generatePSF(inputPDB, inputTopo, outputPrefix, nucleicChoice):
    fnPSFgen = outputPrefix+"psfgen.tcl"
    with open(fnPSFgen, "w") as psfgen:
        psfgen.write("mol load pdb %s\n" % inputPDB)
        psfgen.write("\n")
        psfgen.write("package require psfgen\n")
        psfgen.write("topology %s\n" % inputTopo)
        psfgen.write("pdbalias residue HIS HSE\n")
        psfgen.write("pdbalias residue MSE MET\n")
        psfgen.write("pdbalias atom ILE CD1 CD\n")
        if nucleicChoice == NUCLEIC_RNA:
            psfgen.write("pdbalias residue A ADE\n")
            psfgen.write("pdbalias residue G GUA\n")
            psfgen.write("pdbalias residue C CYT\n")
            psfgen.write("pdbalias residue U URA\n")
        elif nucleicChoice == NUCLEIC_DNA:
            psfgen.write("pdbalias residue DA ADE\n")
            psfgen.write("pdbalias residue DG GUA\n")
            psfgen.write("pdbalias residue DC CYT\n")
            psfgen.write("pdbalias residue DT THY\n")
        psfgen.write("\n")
        if nucleicChoice == NUCLEIC_RNA or nucleicChoice == NUCLEIC_DNA:
            psfgen.write("set nucleic [atomselect top nucleic]\n")
            psfgen.write("set chains [lsort -unique [$nucleic get chain]] ;\n")
            psfgen.write("foreach chain $chains {\n")
            psfgen.write("    set sel [atomselect top \"nucleic and chain $chain\"]\n")
            psfgen.write("    $sel writepdb %s_tmp.pdb\n" % outputPrefix)
            psfgen.write("    segment N${chain} { pdb %s_tmp.pdb }\n" % outputPrefix)
            psfgen.write("    coordpdb %s_tmp.pdb N${chain}\n" % outputPrefix)
            if nucleicChoice == NUCLEIC_DNA:
                psfgen.write("    set resids [lsort -unique [$sel get resid]]\n")
                psfgen.write("    foreach r $resids {\n")
                psfgen.write("        patch DEOX N${chain}:$r\n")
                psfgen.write("    }\n")
            psfgen.write("}\n")
            if nucleicChoice == NUCLEIC_DNA:
                psfgen.write("regenerate angles dihedrals\n")
            psfgen.write("\n")
        psfgen.write("set protein [atomselect top protein]\n")
        psfgen.write("set chains [lsort -unique [$protein get pfrag]]\n")
        psfgen.write("foreach chain $chains {\n")
        psfgen.write("    set sel [atomselect top \"protein and pfrag $chain\"]\n")
        psfgen.write("    $sel writepdb %s_tmp.pdb\n" % outputPrefix)
        psfgen.write("    segment P${chain} {pdb %s_tmp.pdb}\n" % outputPrefix)
        psfgen.write("    coordpdb %s_tmp.pdb P${chain}\n" % outputPrefix)
        psfgen.write("}\n")
        psfgen.write("rm -f %s_tmp.pdb\n" % outputPrefix)
        psfgen.write("\n")
        psfgen.write("guesscoord\n")
        psfgen.write("writepdb %s.pdb\n" % outputPrefix)
        psfgen.write("writepsf %s.psf\n" % outputPrefix)
        psfgen.write("exit\n")

    #Run VMD PSFGEN
    runCommand("vmd -dispdev text -e %s > %s.log " %(fnPSFgen,outputPrefix))

    #Clean
    os.system("rm -f " + fnPSFgen)


def generateGROTOP(inputPDB, outputPrefix, forcefield, smog_dir, nucleicChoice):
    mol = PDBMol(inputPDB)
    # mol.remove_alter_atom()
    mol.remove_hydrogens()
    mol.check_res_order()

    moltmp = mol.copy()

    moltmp.alias_atom("CD", "CD1", "ILE")
    moltmp.alias_atom("OT1", "O")
    moltmp.alias_atom("OT2", "OXT")
    moltmp.alias_res("HSE", "HIS")

    if nucleicChoice == NUCLEIC_RNA:
        moltmp.alias_res("CYT", "C")
        moltmp.alias_res("GUA", "G")
        moltmp.alias_res("ADE", "A")
        moltmp.alias_res("URA", "U")

    elif nucleicChoice == NUCLEIC_DNA:
        moltmp.alias_res("CYT", "DC")
        moltmp.alias_res("GUA", "DG")
        moltmp.alias_res("ADE", "DA")
        moltmp.alias_res("THY", "DT")

    moltmp.alias_atom("O1'", "O1*")
    moltmp.alias_atom("O2'", "O2*")
    moltmp.alias_atom("O3'", "O3*")
    moltmp.alias_atom("O4'", "O4*")
    moltmp.alias_atom("O5'", "O5*")
    moltmp.alias_atom("C1'", "C1*")
    moltmp.alias_atom("C2'", "C2*")
    moltmp.alias_atom("C3'", "C3*")
    moltmp.alias_atom("C4'", "C4*")
    moltmp.alias_atom("C5'", "C5*")
    moltmp.alias_atom("C5M", "C7")
    moltmp.add_terminal_res()
    moltmp.atom_res_reorder()
    moltmp.save(inputPDB)

    # Run Smog2
    runCommand("%s/bin/smog2" % smog_dir+\
               " -i %s -dname %s -%s -limitbondlength -limitcontactlength > %s.log" %
               (inputPDB, outputPrefix,
                "CA" if forcefield == FORCEFIELD_CAGO else "AA", outputPrefix))


    if forcefield == FORCEFIELD_CAGO:
        mol.select_atoms(mol.allatoms2ca())
    mol.save(inputPDB)

    # ADD CHARGE TO TOP FILE
    grotopFile = outputPrefix + ".top"
    with open(grotopFile, 'r') as f1:
        with open(grotopFile + ".tmp", 'w') as f2:
            atom_scope = False
            write_line = False
            for line in f1:
                if "[" in line and "]" in line:
                    if "atoms" in line:
                        atom_scope = True
                if atom_scope:
                    if "[" in line and "]" in line:
                        if not "atoms" in line:
                            atom_scope = False
                            write_line = False
                    elif not ";" in line and not (not line or line.isspace()):
                        write_line = True
                    else:
                        write_line = False
                if write_line:
                    f2.write("%s\t0.0\n" % line[:-1])
                else:
                    f2.write(line)
    os.system("cp %s.tmp %s" % (grotopFile, grotopFile))
    os.system("rm -f %s.tmp" % grotopFile)

def save_dcd(mol, coords_list, prefix):
    print("> Saving DCD trajectory ...")
    n_frames = len(coords_list)

    # saving PDBs
    mol = mol.copy()
    for i in range(n_frames):
        mol.coords = coords_list[i]
        mol.save("%s_frame%i.pdb" % (prefix, i))

    # VMD command
    with open(prefix+"_cmd.tcl", "w") as f :
        f.write("mol new %s_frame0.pdb\n" % prefix)
        for i in range(1,n_frames):
            f.write("mol addfile %s_frame%i.pdb\n" % (prefix, i))
        f.write("animate write dcd %s_traj.dcd\n" % prefix)
        f.write('exit\n')

    # Running VMD
    runCommand("vmd -dispdev text -e %s_cmd.tcl" % prefix)

    # Cleaning
    for i in range(n_frames):
        runCommand("rm -f %s_frame%i.pdb\n" % (prefix, i))
    runCommand("rm -f %s_cmd.tcl" % prefix)
    print("\t Done \n")



def compute_pca(data, length=None, labels=None, n_components=2, figsize=(5,5), colors=None, alphas=None,
                marker=None, traj=None, inv_pca=[], n_inv_pca=10, initdcd=None):
    print("Computing PCA ...")
    # plt.style.context("default")
    if length is None:
        length = [len(data)]
    if colors is None:
        if len(length)<=10:
            colors = ["tab:red", "tab:blue", "tab:orange", "tab:green",
                      "tab:brown", "tab:olive", "tab:pink", "tab:gray", "tab:cyan", "tab:purple"]
        else:
            colors = np.random.rand(len(length), 3)
    # Compute PCA
    arr = np.array(data)
    pca = PCA(n_components=n_components)

    components = pca.fit_transform(arr).T

    # Prepare plotting data
    idx = np.concatenate((np.array([0]),np.cumsum(length))).astype(int)
    if labels is None:
        pltlabels = ["#"+str(i) for i in range(len(length))]
    else:
        pltlabels=labels

    fig = plt.figure(figsize=figsize)
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlabel("PCA component 3")
    else:
        ax = fig.add_subplot(111)
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")

    if alphas is None:
        alphas = [1 for i in range(len(length))]
    if marker is None:
        marker = ["o" for i in range(len(length))]
    if traj is None:
        traj = [1 for i in range(len(length))]

    for i in range(len(length)):
        len_traj = length[i]//traj[i]
        for j in range(traj[i]):
            args = [
                components[0, idx[i] + j * len_traj:idx[i] + (j + 1) * len_traj],
                components[1, idx[i] + j * len_traj:idx[i] + (j + 1) * len_traj]
            ]
            if n_components==3:
                args.append(components[2, idx[i] + j * len_traj:idx[i] + (j + 1) * len_traj])

            ax.plot(*args, marker[i], label=pltlabels[i], markeredgecolor='black',
                        color = colors[i], alpha=alphas[i])
    if labels is not None :
        ax.legend()
    fig.tight_layout()

    annot = ax.annotate("", xy=(0, 0), xytext=(-40, 40), textcoords="offset points",
                        bbox=dict(boxstyle='round4', fc='linen', ec='k', lw=1),
                        arrowprops=dict(arrowstyle='-|>'))
    annot.set_visible(False)
    click_coord = []

    def onclick(event):
        if len(click_coord) < 2:
            click_coord.append((event.xdata, event.ydata))
            x = event.xdata
            y = event.ydata

            # printing the values of the selected point
            print([x, y])
            annot.xy = (x, y)
            text = "({:.2g}, {:.2g})".format(x, y)
            annot.set_text(text)
            annot.set_visible(True)
            fig.canvas.draw()

        if len(click_coord) == 2:
            click_sel = np.array([np.linspace(click_coord[0][0], click_coord[1][0], n_inv_pca),
                            np.linspace(click_coord[0][1], click_coord[1][1], n_inv_pca)
                            ])
            ax.plot(click_sel[0], click_sel[1], "-o", color="black")
            inv_pca.insert(0,pca.inverse_transform(click_sel.T))
            click_coord.clear()
            fig.canvas.draw()

            initdcdcp = initdcd.copy()
            coords_list = []
            for i in range(n_inv_pca):
                coords_list.append(inv_pca[0][i].reshape((initdcdcp.n_atoms, 3)))
            save_dcd(mol=initdcdcp, coords_list=coords_list, prefix="tmp")
            initdcdcp.coords = coords_list[0]
            initdcdcp.save("tmp.pdb")
            traj_viewer(pdb_file="tmp.pdb", dcd_file="tmp_traj.dcd")


    if n_components == 2:
        fig.canvas.mpl_connect('button_press_event', onclick)

    return fig, ax

def traj_viewer(pdb_file, dcd_file):
    runCommand("vmd %s %s" %(pdb_file,dcd_file))

def alignMol(mol1, mol2, idx=None):
    print("> Aligning PDB ...")

    sup = SVDSuperimposer()
    if idx is not None:
        c1 = mol1.coords[idx[:, 0]]
        c2 = mol2.coords[idx[:, 1]]
    else:
        c1 = mol1.coords
        c2 = mol2.coords
    sup.set(c1, c2)
    sup.run()
    rot, tran = sup.get_rotran()
    mol2.coords = np.dot(mol2.coords, rot) + tran
    print("\t Done \n")




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

def rmsdFromDCD(outputPrefix, inputPDB, targetPDB, align=False):

    # EXTRACT PDBs from dcd file
    with open("%s_tmp_dcd2pdb.tcl" % outputPrefix, "w") as f:
        s = ""
        s += "mol load pdb %s dcd %s.dcd\n" % (inputPDB, outputPrefix)
        s += "set nf [molinfo top get numframes]\n"
        s += "for {set i 0 } {$i < $nf} {incr i} {\n"
        s += "[atomselect top all frame $i] writepdb %stmp$i.pdb\n" % outputPrefix
        s += "}\n"
        s += "exit\n"
        f.write(s)
    runCommand("vmd -dispdev text -e %s_tmp_dcd2pdb.tcl > /dev/null" % outputPrefix)

    # DEF RMSD
    def RMSD(c1, c2):
        return np.sqrt(np.mean(np.square(np.linalg.norm(c1 - c2, axis=1))))

    # COMPUTE RMSD
    rmsd = []
    inputPDBmol = PDBMol(inputPDB)
    targetPDBmol = PDBMol(targetPDB)

    idx = matchPDBatoms([targetPDBmol, inputPDBmol], ca_only=True)
    if align:
        alignMol(targetPDBmol, inputPDBmol, idx=idx)
    rmsd.append(RMSD(inputPDBmol.coords[idx[:, 1]], targetPDBmol.coords[idx[:, 0]]))
    i=0
    while(os.path.exists("%stmp%i.pdb"%(outputPrefix,i+1))):
        f = "%stmp%i.pdb"%(outputPrefix,i+1)
        mol = PDBMol(f)
        if align:
            alignMol(targetPDBmol, mol, idx=idx)
        rmsd.append(RMSD(mol.coords[idx[:, 1]], targetPDBmol.coords[idx[:, 0]]))
        i+=1

    # CLEAN TMP FILES AND SAVE
    runCommand("rm -f %stmp*" % (outputPrefix))
    return rmsd

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
    runCommand("rm -f %stmp*" % (outputPDB))



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
    args +="--search5d_shift 7.0 --search5d_step 1.0"
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
    args +="--psi_step 5.0 --max_shift_change 7.0 --search5D"
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

