# **************************************************************************
# * Author:  Mohamad Harastani          (mohamad.harastani@upmc.fr)
# * IMPMC, UPMC Sorbonne University
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# **************************************************************************
from pyworkflow.object import String
from pyworkflow.protocol.params import (PointerParam, EnumParam, IntParam)
from pwem.protocols import ProtAnalysis3D
from pwem.convert import cifToPdb
from pyworkflow.utils.path import makePath, copyFile
from pyworkflow.protocol import params
from pwem.utils import runProgram
from pwem.emlib import MetaData, MDL_ENABLED, MDL_NMA_MODEFILE,MDL_ORDER
from pwem.objects import SetOfNormalModes, AtomStruct
from .convert import rowToMode
from xmipp3.base import XmippMdRow

from umap import UMAP

import numpy as np
import glob
from sklearn import decomposition
from joblib import dump

from .utilities.genesis_utilities import dcd2numpyArr
from .utilities.pdb_handler import ContinuousFlexPDBHandler

PDB_SOURCE_SUBTOMO = 0
PDB_SOURCE_PATTERN = 1
PDB_SOURCE_OBJECT = 2
PDB_SOURCE_TRAJECT = 3

REDUCE_METHOD_PCA = 0
REDUCE_METHOD_UMAP = 1

class FlexProtDimredPdb(ProtAnalysis3D):
    """ Protocol for applying dimentionality reduction on PDB files. """
    _label = 'pdb dimentionality reduction'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('pdbSource', EnumParam, default=0,
                      label='Source of PDBs',
                      choices=['Used for subtomogram synthesis', 'File pattern', 'Object', 'Trajectory Files'],
                      help='Use the file pattern as file location with /*.pdb')
        form.addParam('pdbs', params.PointerParam, pointerClass='FlexProtSynthesizeSubtomo',
                      condition='pdbSource == 0',
                      label="Subtomogram synthesis",
                      help='Point to a protocol of synthesizing subtomograms, the ground truth PDBs will be used as input')
        form.addParam('pdbs_file', params.PathParam,
                      condition='pdbSource == 1',
                      label="List of PDBs",
                      help='Use the file pattern as file location with /*.pdb')
        form.addParam('setOfPDBs', params.PointerParam, pointerClass='SetOfPDBs, SetOfAtomStructs',
                      condition='pdbSource == 2',
                      label="Set of PDBs",
                      help='Use a scipion object SetOfPDBs / SetOfAtomStructs')
        form.addParam('dcds_file', params.PathParam,
                      condition='pdbSource == 3',
                      label="List of trajectory DCD files",
                      help='Use the file pattern as file location with /*.dcd')
        form.addParam('dcd_start', params.IntParam, default=0,
                      condition='pdbSource == 3',
                      label="Beginning of the trajectory",
                      help='Index of the desired begining of the trajectory')
        form.addParam('dcd_end', params.IntParam, default=-1,
                      condition='pdbSource == 3',
                      label="Ending of the trajectory",
                      help='Index of the desired end of the trajectory')
        form.addParam('dcd_step', params.IntParam, default=1,
                      condition='pdbSource == 3',
                      label="Step of the trajectory",
                      help='Step to skip points in the trajectory')
        form.addParam('dcd_ref_pdb', params.PointerParam, pointerClass='AtomStruct',
                      condition='pdbSource == 3',
                      label="trajectory Reference PDB",
                      help='Reference PDB of the trajectory')
        form.addParam('method', params.EnumParam, label="Reduction method", default=REDUCE_METHOD_PCA,
                      choices=['PCA', 'UMAP'],help="")

        form.addParam('reducedDim', IntParam, default=2,
                      label='Number of Principal Components')
        form.addParam('alignPDBs', params.BooleanParam, default=False,
                      label="Align PDBs ?",
                      help='Perform rigid body alignement on the set of PDBs to a reference PDB')
        form.addParam('alignRefPDB', params.PointerParam, pointerClass='AtomStruct',
                      condition='alignPDBs',
                      label="Alignement Reference PDB",
                      help='Reference PDB to align the PDBs with')
        form.addParam('matchingType', params.EnumParam, label="Match structures ?", default=0,
                      choices=['All structures are matching', 'Match chain name/residue num/atom name',
                               'Match segment name/residue num/atom name'],
                      help="Method to find atomic coordinates correspondence between the trajectory "
                           "coordinates and the reference PDB. The method will select the matching atoms"
                           " and sort them in the corresponding order. If the structures in the files are"
                           " already matching, choose All structures are matching")

        # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('readInputFiles')
        self._insertFunctionStep('performDimred')

        if self.method.get() == REDUCE_METHOD_PCA:
            self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------
    def readInputFiles(self):
        inputFiles = self.getInputFiles()

        # Align PDBS if needed
        if self.pdbSource.get() != PDB_SOURCE_TRAJECT:
            if self.alignPDBs.get():
                ref = ContinuousFlexPDBHandler(self.alignRefPDB.get().getFileName())
                mol = ContinuousFlexPDBHandler(inputFiles[0])
                if self.matchingType.get() == 1:
                    idx_matching_atoms = mol.matchPDBatoms(reference_pdb=ref, matchingType=0)
                elif self.matchingType.get() == 2:
                    idx_matching_atoms = mol.matchPDBatoms(reference_pdb=ref, matchingType=1)
                else:
                    idx_matching_atoms = None

        # Get pdbs coordinates
        pdbs_matrix = []
        for pdbfn in inputFiles:
            if self.pdbSource.get() == PDB_SOURCE_TRAJECT:
                traj_arr= dcd2numpyArr(pdbfn)
                mol = ContinuousFlexPDBHandler(self.getPDBRef())
                traj_arr.shape
                for i in range(self.dcd_start.get(),
                               self.dcd_end.get() if self.dcd_end.get()!= -1 else traj_arr.shape[0],
                               self.dcd_step.get()):
                    pdbs_matrix.append(traj_arr[i].flatten())
                    mol.coords=traj_arr[i]
                    mol.write_pdb(self._getExtraPath("%s_traj.pdb"%str(i+1).zfill(5)))
            else:
                try :
                    # Read PDBs
                    mol = ContinuousFlexPDBHandler(pdbfn)

                    # Align PDBs
                    if self.alignPDBs.get():
                        mol= mol.alignMol(reference_pdb=ref, idx_matching_atoms=idx_matching_atoms)

                    pdbs_matrix.append(mol.coords.flatten())

                except RuntimeError:
                    print("Warning : Can not read PDB file %s "%pdbfn)

        self.pdbs_matrix = np.array(pdbs_matrix)


    def performDimred(self):

        if self.method.get() == REDUCE_METHOD_PCA:
            pca = decomposition.PCA(n_components=self.reducedDim.get())
            Y = pca.fit_transform(self.pdbs_matrix)
            dump(pca, self._getExtraPath('pca_pickled.joblib'))

            pathPC = self._getPath("modes")
            pdb = ContinuousFlexPDBHandler(self.getPDBRef())
            pdb.coords = pca.mean_.reshape(self.pdbs_matrix.shape[1] // 3, 3)
            pdb.write_pdb(self._getPath("atoms.pdb"))
            makePath(pathPC)
            matrix = pca.components_.reshape(self.reducedDim.get(),self.pdbs_matrix.shape[1]//3,3)
            self.writePrincipalComponents(prefix=pathPC, matrix = matrix)

        elif self.method.get() == REDUCE_METHOD_UMAP:
            umap = UMAP(n_components=self.reducedDim.get(), n_neighbors=15, n_epochs=1000).fit(self.pdbs_matrix)
            Y = umap.transform(self.pdbs_matrix)
            dump(umap, self._getExtraPath('pca_pickled.joblib'))

        np.savetxt(self.getOutputMatrixFile(),Y)

    def createOutputStep(self):
            # Metadata
            mdOut = MetaData()
            for i in range(self.reducedDim.get()):
                objId = mdOut.addObject()
                modefile = self._getPath("modes", "vec.%d" % (i + 1))
                mdOut.setValue(MDL_NMA_MODEFILE, modefile, objId)
                mdOut.setValue(MDL_ORDER, i + 1, objId)
                mdOut.setValue(MDL_ENABLED, 1, objId)
            mdOut.write(self._getPath("modes.xmd"))

            # Sqlite object
            pcSet =SetOfNormalModes(filename=self._getPath("modes.sqlite"))
            row = XmippMdRow()
            for objId in mdOut:
                row.readFromMd(mdOut, objId)
                pcSet.append(rowToMode(row))

            pdb = AtomStruct(self._getPath("atoms.pdb"))
            self._defineOutputs(outputMean=pdb)

            pcSet.setPdb(pdb)
            self._defineOutputs(outputPCA=pcSet)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        return summary

    def _validate(self):
        errors = []
        return errors

    def _citations(self):
        return ['harastani2020hybrid','Jin2014']

    def _methods(self):
        pass

    # --------------------------- UTILS functions --------------------------------------------
    def _printWarnings(self, *lines):
        """ Print some warning lines to 'warnings.xmd',
        the function should be called inside the working dir."""
        fWarn = open("warnings.xmd", 'w')
        for l in lines:
            print >> fWarn, l
        fWarn.close()

    def getInputFiles(self):
        if self.pdbSource.get()==PDB_SOURCE_SUBTOMO:
            l= [f for f in glob.glob(self.pdbs.get()._getExtraPath('*.pdb'))]
        elif self.pdbSource.get()==PDB_SOURCE_PATTERN:
            l= [f for f in glob.glob(self.pdbs_file.get())]
        elif self.pdbSource.get()==PDB_SOURCE_OBJECT:
            l= [i.getFileName() for i in self.setOfPDBs.get()]
        elif self.pdbSource.get()==PDB_SOURCE_TRAJECT:
            l= [f for f in glob.glob(self.dcds_file.get())]
        l.sort()
        return l

    def getPDBRef(self):
        if self.pdbSource.get()==PDB_SOURCE_TRAJECT:
            return self.dcd_ref_pdb.get().getFileName()
        else:
            return self.getInputFiles()[0]

    def getOutputMatrixFile(self):
        return self._getExtraPath('output_matrix.txt')

    def getDeformationFile(self):
        return self._getExtraPath('pdbs_mat.txt')

    def writePrincipalComponents(self, prefix, matrix):
        for i in range(self.reducedDim.get()):
            with open("%s/vec.%i"%(prefix,i+1), "w") as f:
                    for j in range(matrix.shape[1]):
                        f.write(" %e   %e   %e\n" % (matrix[i,j, 0], matrix[i,j, 1], matrix[i,j, 1]))