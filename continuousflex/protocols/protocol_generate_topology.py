from pwem.protocols import EMProtocol
import pyworkflow.protocol.params as params
from pwem.objects.data import AtomStruct
from .utilities.pdb_handler import ContinuousFlexPDBHandler
from pyworkflow.utils import runCommand
import os


NUCLEIC_NO = 0
NUCLEIC_RNA =1
NUCLEIC_DNA = 2

FORCEFIELD_CHARMM = 0
FORCEFIELD_AAGO = 1
FORCEFIELD_CAGO = 2


class ProtGenerateTopology(EMProtocol):
    """ Protocol to generate topology files for GENESIS simulations """
    _label = 'generate topology'

    def _defineParams(self, form):

        form.addSection(label='Inputs')

        form.addParam('inputPDB', params.PointerParam,
                      pointerClass='AtomStruct', label="Input PDB",
                      help='Select the input PDB.', important=True)

        group = form.addGroup('Forcefield Inputs')
        group.addParam('forcefield', params.EnumParam, label="Forcefield type", default=FORCEFIELD_CHARMM, important=True,
                       choices=['CHARMM', 'AAGO', 'CAGO'],
                       help="Type of the force field used for energy and force calculation")
        group.addParam('nucleicChoice', params.EnumParam, label="Contains nucleic acids ?", default=NUCLEIC_NO,
                       choices=['NO', 'RNA', 'DNA'], help="Specify if the generator should consider nucleic residues as DNA or RNA")

        group.addParam('inputPRM', params.FileParam, label="CHARMM parameter file (prm)",
                       condition="forcefield==%i"%FORCEFIELD_CHARMM,
                       help='CHARMM parameter file containing force field parameters, e.g. force constants and librium'
                            ' geometries. Latest forcefields can be founded at http://mackerell.umaryland.edu/charmm_ff.shtml ')
        group.addParam('inputRTF', params.FileParam, label="CHARMM topology file (rtf)",
                       condition="forcefield==%i"%FORCEFIELD_CHARMM,
                       help='CHARMM topology file containing information about atom connectivity of residues and'
                            ' other molecules. Latest forcefields can be founded at http://mackerell.umaryland.edu/charmm_ff.shtml ')
        group.addParam('inputSTR', params.FileParam, label="CHARMM stream file (str, optional)",
                       condition="forcefield==%i"%FORCEFIELD_CHARMM, default="",
                       help='CHARMM stream file containing both topology information and parameters. '
                            'Latest forcefields can be founded at http://mackerell.umaryland.edu/charmm_ff.shtml ')


        group.addParam('smog_dir', params.FileParam, label="SMOG 2 install directory",
                       help="Path to SMOG2 install directory (For SMOG2 installation, see "
                            "https://smog-server.org/smog2/ , otherwise use the web GUI "
                            "https://smog-server.org/cgi-bin/GenTopGro.pl )",
                       condition="(forcefield==%i or forcefield==%i)"%(FORCEFIELD_CAGO, FORCEFIELD_AAGO))

    def _insertAllSteps(self):
        ff = self.forcefield.get()

        if ff == FORCEFIELD_CAGO or ff == FORCEFIELD_AAGO:
            self._insertFunctionStep("generateGROTOP")

        if ff == FORCEFIELD_CHARMM:
            self._insertFunctionStep("generatePSF")

        self._insertFunctionStep("createOutput")

    def createOutput(self):
        self._defineOutputs(outputPDB=AtomStruct(self._getExtraPath("output.pdb")))

    def generatePSF(self):
        inputPDB = self.inputPDB.get().getFileName()
        inputTopo = self.inputRTF.get()
        outputPrefix = self._getExtraPath("output")
        nucleicChoice = self.nucleicChoice.get()

        fnPSFgen = outputPrefix + "psfgen.tcl"
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

        # Run VMD PSFGEN
        runCommand("vmd -dispdev text -e %s > %s.log " % (fnPSFgen, outputPrefix))

        # Check PDB
        outMol = ContinuousFlexPDBHandler(outputPrefix + ".pdb")
        if outMol.n_atoms == 0:
            raise RuntimeError("VMD psfgen failed, check %s.log for details" % outputPrefix)


    def generateGROTOP(self):
        inputPDB = self.inputPDB.get().getFileName()
        outputPrefix = self._getExtraPath("output")
        forcefield = self.forcefield.get()

        mol = ContinuousFlexPDBHandler(inputPDB)
        # mol.remove_alter_atom()
        mol.remove_hydrogens()
        mol.check_res_order()

        moltmp = mol.copy()

        moltmp.alias_atom("CD", "CD1", "ILE")
        moltmp.alias_atom("OT1", "O")
        moltmp.alias_atom("OT2", "OXT")
        moltmp.alias_res("HSE", "HIS")
        moltmp.alias_res("HSD", "HIS")
        moltmp.alias_res("HSP", "HIS")

        if self.nucleicChoice.get() == NUCLEIC_RNA:
            moltmp.alias_res("CYT", "C")
            moltmp.alias_res("GUA", "G")
            moltmp.alias_res("ADE", "A")
            moltmp.alias_res("URA", "U")

        elif self.nucleicChoice.get() == NUCLEIC_DNA:
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
        moltmp.write_pdb(inputPDB)

        # Run Smog2
        runCommand("%s/bin/smog2" % self.smog_dir.get() + \
                   " -i %s -dname %s -%s -limitbondlength -limitcontactlength > %s.log" %
                   (inputPDB, outputPrefix,
                    "CA" if forcefield == FORCEFIELD_CAGO else "AA", outputPrefix))

        if forcefield == FORCEFIELD_CAGO:
            mol.select_atoms(mol.allatoms2ca())
        mol.write_pdb(outputPrefix + ".pdb")

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
        runCommand("cp %s.tmp %s" % (grotopFile, grotopFile))
        runCommand("rm -f %s.tmp" % grotopFile)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        return summary

    def _citations(self):
        return []

    def _methods(self):
        pass
