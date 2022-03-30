from pwem.protocols import EMProtocol
import pyworkflow.protocol.params as params
from .utilities.genesis_utilities import PDBMol
from sklearn.decomposition import PCA
import numpy as np
from pwem.objects.data import AtomStruct

class ProtPCAFromPDB(EMProtocol):
    """ Protocol to extract PCA space from set of PDBs """
    _label = 'PCAfromPDB'
    def _defineParams(self, form):
        form.addSection(label='Inputs')
        form.addParam('inputPDBs', params.PointerParam, label="Input set of PDBs",pointerClass="SetOfAtomStructs,SetOfPDBs",
                       help='TODO', important=True)
        form.addParam('n_pca', params.IntParam, default=10, label='Number of components',
                      help="TODO")

    def _insertAllSteps(self):
        self._insertFunctionStep("runPCAfromPDB")

    def runPCAfromPDB(self):

        pdbs = []
        for i in range(self.inputPDBs.get().getSize()):
            pdbs.append(self.inputPDBs.get()[i + 1].getFileName())

        cp = self.get_pca_space(pdbs = pdbs, outpdb=self._getExtraPath("output.pdb"),
                           outpca=self._getExtraPath("output.pca"), n_pca = self.n_pca.get())

        np.savetxt(fname=self._getExtraPath("output.crd"), X = cp)

        self._defineOutputs(outputPDB=AtomStruct(self._getExtraPath("output.pdb")))

    def save_pca(self,filename, arr, n_pca):
        with open(filename, "w") as f:
            for i in range(6 + n_pca):
                f.write(" VECTOR    %i       VALUE  0.0\n" % (i + 1))
                f.write(" -----------------------------------\n")
                if i < 6:
                    for j in range(arr.shape[1]):
                        f.write(" 0.0   0.0   0.0\n")
                else:
                    for j in range(arr.shape[1]):
                        f.write(" %e   %e   %e\n" % (arr[i - 6, j, 0], arr[i - 6, j, 1], arr[i - 6, j, 2]))

    def get_pca_space(self, pdbs, outpdb, outpca, n_pca):
        data = []
        n_pdbs = len(pdbs)
        for i in pdbs:
            mol = PDBMol(i)
            data.append(mol.coords.flatten())
        pca = PCA(n_components=n_pca)
        pca_coords = pca.fit_transform(X=np.array(data))
        pca_coord0 = pca.inverse_transform(np.zeros(n_pca)).reshape(mol.n_atoms, 3)
        mol.coords = pca_coord0
        mol.save(outpdb)
        components = pca.components_.reshape(n_pca, mol.n_atoms, 3)
        self.save_pca(outpca, components, n_pca)
        return pca_coords

    def _summary(self):
        summary = []
        return summary

    def _validate(self):
        errors = []
        return errors

    def _citations(self):
        pass

    def _methods(self):
        pass