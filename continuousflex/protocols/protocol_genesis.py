# **************************************************************************
# * Authors: Rémi Vuillemot             (remi.vuillemot@upmc.fr)
# *
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


import pyworkflow.protocol.params as params
from pwem.protocols import EMProtocol
from pwem.objects.data import AtomStruct, SetOfAtomStructs, SetOfPDBs, SetOfVolumes,SetOfParticles

import numpy as np
import mrcfile
import os
import sys
import pwem.emlib.metadata as md
from pwem.utils import runProgram
from subprocess import Popen
from xmippLib import Euler_angles2matrix

from .utilities.genesis_utilities import *
from xmipp3 import Plugin
import pyworkflow.utils as pwutils
from pyworkflow.utils import runCommand

EMFIT_NONE = 0
EMFIT_VOLUMES = 1
EMFIT_IMAGES = 2

FORCEFIELD_CHARMM = 0
FORCEFIELD_AAGO = 1
FORCEFIELD_CAGO = 2

SIMULATION_MD = 0
SIMULATION_MIN = 1
SIMULATION_REMD = 2

PROGRAM_ATDYN = 0
PROGRAM_SPDYN= 1

INTEGRATOR_VVERLET = 0
INTEGRATOR_LEAPFROG = 1
INTEGRATOR_NMMD = 2

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

PREPROCESS_VOL_NONE = 0
PREPROCESS_VOL_NORM = 1
PREPROCESS_VOL_OPT = 2
PREPROCESS_VOL_MATCH = 3

RB_PROJMATCH = 0
RB_WAVELET = 1

class ProtGenesis(EMProtocol):
    """ Protocol to perform MD simulation using GENESIS. """
    _label = 'Genesis'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):

        # Inputs ============================================================================================
        form.addSection(label='Inputs')
        form.addParam('inputPDB', params.PointerParam,
                      pointerClass='AtomStruct, SetOfPDBs, SetOfAtomStructs', label="Input PDB (s)",
                      help='Select the input PDB or set of PDBs.')
        form.addParam('forcefield', params.EnumParam, label="Forcefield type", default=0,
                      choices=['CHARMM', 'AAGO', 'CAGO'], help="Type of the force field used for energy and force calculation")
        form.addParam('generateTop', params.BooleanParam, label="Generate topology files ?",
                      default=False, help="Use the GUI to generate topology files for you (PSF file for CHARMM and TOP file for AAGO/CAGO)."
                                          " Requires VMD psfgen for CHARMM forcefields "
                                          " and SMOG2 for GO models. Note that the generated topology files will not include"
                                          " solvent.")
        form.addParam('nucleicChoice', params.EnumParam, label="Contains nucleic acids ?", default=0,
                      choices=['NO', 'RNA', 'DNA'], condition ="generateTop",help="TODo")
        form.addParam('smog_dir', params.FileParam, label="SMOG2 directory",
                      help='Path to SMOG2 directory', condition="(forcefield==1 or forcefield==2) and generateTop")
        form.addParam('inputTOP', params.FileParam, label="GROMACS Topology File",
                      condition="(forcefield==1 or forcefield==2) and not generateTop",
                      help='Gromacs ‘top’ file containing information of the system such as atomic masses, charges,'
                           ' atom connectivities. For details about this format, see the Gromacs web site')
        form.addParam('inputPRM', params.FileParam, label="CHARMM parameter file",
                      condition = "forcefield==0",
                      help='CHARMM parameter file containing force field parameters, e.g. force constants and librium'
                            ' geometries' )
        form.addParam('inputRTF', params.FileParam, label="CHARMM topology file",
                      condition="forcefield==0 or ((forcefield==1 or forcefield==2) and generateTop)",
                      help='CHARMM topology file containing information about atom connectivity of residues and'
                           ' other molecules. For details on the format, see the CHARMM web site')
        form.addParam('inputPSF', params.FileParam, label="CHARMM Structure File",
                      condition="forcefield==0 and not generateTop",
                      help='CHARMM/X-PLOR psf file containing information of the system such as atomic masses,'
                            ' charges, and atom connectivities. To generate this file, you can either use the option'
                           '\" generate topology files\", VMD psfgen, or online CHARMM GUI.')
        form.addParam('inputSTR', params.FileParam, label="CHARMM stream file (optional)",
                      condition="forcefield==0", default="",
                      help='CHARMM stream file containing both topology information and parameters')


        form.addParam('inputRST', params.FileParam, label="GENESIS Restart File (optional)",
                       help='Restart a previous GENESIS run with a .rst file', default="")


        # Simulation =================================================================================================
        form.addSection(label='Simulation')
        form.addParam('md_program', params.EnumParam, label="MD program", default=0,
                      choices=['ATDYN', 'SPDYN'],
                      help="SPDYN (Spatial decomposition dynamics) and ATDYN (Atomic decomposition dynamics)"
                    " share almost the same data structures, subroutines, and modules, but differ in"
                    " their parallelization schemes. In SPDYN, the spatial decomposition scheme is implemented with new"
                    " parallel algorithms and GPGPU calculation. In ATDYN, the atomic decomposition scheme"
                    " is introduced for simplicity. The performance of ATDYN is not comparable to SPDYN due to the"
                    " simple parallelization scheme but contains new methods and features.", important=True)
        form.addParam('simulationType', params.EnumParam, label="Simulation type", default=0,
                      choices=['Molecular Dynamics', 'Minimization', 'Replica-Exchange Molecular Dynamics'],
                      help="Type of simulation to be performed by GENESIS", important=True)
        form.addParam('integrator', params.EnumParam, label="Integrator", default=0,
                      choices=['Velocity Verlet', 'Leapfrog', 'NMMD'],
                      help="Type of integrator for the MD simulation", condition="simulationType!=1")
        form.addParam('time_step', params.FloatParam, default=0.002, label='Time step (ps)',
                      help="Time step in the MD run", condition="simulationType!=1")
        form.addParam('n_steps', params.IntParam, default=10000, label='Number of steps',
                      help="Total number of steps in one MD run")
        form.addParam('eneout_period', params.IntParam, default=100, label='Energy output period',
                      help="Output frequency for the energy data")
        form.addParam('crdout_period', params.IntParam, default=100, label='Coordinate output period',
                      help="Output frequency for the coordinates data")
        form.addParam('nbupdate_period', params.IntParam, default=10, label='Non-bonded update period',
                      help="Update frequency of the non-bonded pairlist",
                      expertLevel=params.LEVEL_ADVANCED)

        form.addParam('nm_number', params.IntParam, default=10, label='[NMMD] Number of normal modes',
                      help="Number of normal modes for NMMD. 10 should work in most cases. Avoid "
                           " using too much NM (>50).",
                      condition="integrator==2 and simulationType!=1")
        form.addParam('nm_mass', params.FloatParam, default=10.0, label='[NMMD] NM mass',
                      help="Mass value of Normal modes for NMMD", condition="integrator==2 and simulationType!=1",
                      expertLevel=params.LEVEL_ADVANCED)
        form.addParam('nm_limit', params.FloatParam, default=1000.0, label='[NMMD] NM amplitude threshold',
                      help="Threshold of normal mode amplitude above which the normal modes are updated",
                      condition="integrator==2 and simulationType!=1",expertLevel=params.LEVEL_ADVANCED)
        form.addParam('elnemo_cutoff', params.FloatParam, default=8.0, label='[NMMD] NMA cutoff (A)',
                      help="Cutoff distance for elastic network model", condition="integrator==2 and simulationType!=1",
                      expertLevel=params.LEVEL_ADVANCED)
        form.addParam('elnemo_rtb_block', params.IntParam, default=10, label='[NMMD] NMA Number of residue RTB',
                      help="Number of residue per RTB block in the NMA computation",
                      condition="integrator==2 and simulationType!=1",expertLevel=params.LEVEL_ADVANCED)

        form.addParam('exchange_period', params.IntParam, default=1000, label='[REMD] Exchange Period',
                      help="Number of MD steps between replica exchanges", condition="simulationType==2")
        form.addParam('nreplica', params.IntParam, default=1, label='[REMD] Number of replicas',
                      help="Number of replicas for REMD", condition="simulationType==2")
        # ENERGY =================================================================================================
        form.addSection(label='Energy')
        form.addParam('implicitSolvent', params.EnumParam, label="Implicit Solvent", default=1,
                      choices=['GBSA', 'NONE'],
                      help="Turn on Generalized Born/Solvent accessible surface area model. Boundary condition must be NO."
                           " ATDYN only.")

        form.addParam('electrostatics', params.EnumParam, label="Non-bonded interactions", default=1,
                      choices=['PME', 'Cutoff'],
                      help="Type of Non-bonded interactions. "
                           " CUTOFF: Non-bonded interactions including the van der Waals interaction are just"
                           " truncated at cutoffdist; "
                           " PME : Particle mesh Ewald (PME) method is employed for long-range interactions."
                            " This option is only availabe in the periodic boundary condition")
        form.addParam('vdw_force_switch', params.BooleanParam, label="Switch function Van der Waals", default=True,
                      help="This paramter determines whether the force switch function for van der Waals interactions is"
                        " employed or not. The users must take care about this parameter, when the CHARMM"
                        " force field is used. Typically, vdw_force_switch=YES should be specified in the case of"
                        " CHARMM36",expertLevel=params.LEVEL_ADVANCED)
        form.addParam('switch_dist', params.FloatParam, default=10.0, label='Switch Distance',
                      help="Switch-on distance for nonbonded interaction energy/force quenching")
        form.addParam('cutoff_dist', params.FloatParam, default=12.0, label='Cutoff Distance',
                      help="Cut-off distance for the non-bonded interactions. This distance must be larger than"
                            " switchdist, while smaller than pairlistdist")
        form.addParam('pairlist_dist', params.FloatParam, default=15.0, label='Pairlist Distance',
                      help="Distance used to make a Verlet pair list for non-bonded interactions . This distance"
                            " must be larger than cutoffdist")

        # Ensemble =================================================================================================
        form.addSection(label='Ensemble')
        form.addParam('ensemble', params.EnumParam, label="Ensemble", default=0,
                      choices=['NVT', 'NVE', 'NPT'],
                      help="Type of ensemble, NVE: Microcanonical ensemble, NVT: Canonical ensemble,"
                           " NPT: Isothermal-isobaric ensemble")
        form.addParam('tpcontrol', params.EnumParam, label="Temperature control", default=1,
                      choices=['NO', 'LANGEVIN', 'BERENDSEN', 'BUSSI'],
                      help="Type of thermostat and barostat. The availabe algorithm depends on the integrator :"
                           " LEAP : BERENDSEN, LANGEVIN;  VVER : BERENDSEN (NVT only), LANGEVIN, BUSSI; "
                           " NMMD : LANGEVIN (NVT only)")
        form.addParam('temperature', params.FloatParam, default=300.0, label='Temperature (K)',
                      help="Initial and target temperature")
        form.addParam('pressure', params.FloatParam, default=1.0, label='Pressure (atm)',
                      help="Target pressure in the NPT ensemble", condition="ensemble==2")
        # Boundary =================================================================================================
        form.addSection(label='Boundary')
        form.addParam('boundary', params.EnumParam, label="Boundary", default=0,
                      choices=['No boundary', 'Periodic Boundary Condition'], important=True,
                      help="Type of boundary condition")
        form.addParam('box_size_x', params.FloatParam, label='Box size X',
                      help="Box size along the x dimension", condition="boundary==1")
        form.addParam('box_size_y', params.FloatParam, label='Box size Y',
                      help="Box size along the y dimension", condition="boundary==1")
        form.addParam('box_size_z', params.FloatParam, label='Box size Z',
                      help="Box size along the z dimension", condition="boundary==1")
        # Experiments =================================================================================================
        form.addSection(label='Experiments')
        form.addParam('EMfitChoice', params.EnumParam, label="Cryo-EM Flexible Fitting", default=0,
                      choices=['None', 'Volume (s)', 'Image (s)'], important=True,
                      help="Type of cryo-EM data to be processed")
        form.addParam('constantK', params.StringParam, default="10000", label='Force constant (kcal/mol)',
                      help="Force constant in Eem = k*(1 - c.c.). Note that in the case of REUS, the number of "
                           " force constant value must be equal to the number of replicas, for example for 4 replicas,"
                           " a valid force constant is \"1000 2000 3000 4000\" "
                      , condition="EMfitChoice!=0")
        form.addParam('emfit_sigma', params.FloatParam, default=2.0, label="EMfit Sigma",
                      help="Resolution parameter of the simulated map. This is usually set to the half of the resolution"
                        " of the target map. For example, if the target map resolution is 5 Å, emfit_sigma=2.5",
                      condition="EMfitChoice!=0",expertLevel=params.LEVEL_ADVANCED)
        form.addParam('emfit_tolerance', params.FloatParam, default=0.01, label='EMfit Tolerance',
                      help="This variable determines the tail length of the Gaussian function. For example, if em-"
                        " fit_tolerance=0.001 is specified, the Gaussian function is truncated to zero when it is less"
                        " than 0.1% of the maximum value. Smaller value requires large computational cost",
                      condition="EMfitChoice!=0",expertLevel=params.LEVEL_ADVANCED)

        # Volumes
        form.addParam('inputVolume', params.PointerParam, pointerClass="Volume, SetOfVolumes",
                      label="Input volume (s)", help='Select the target EM density volume',
                      condition="EMfitChoice==1")
        form.addParam('voxel_size', params.FloatParam, default=1.0, label='Voxel size (A)',
                      help="Voxel size in ANgstrom of the target volume", condition="EMfitChoice==1")
        form.addParam('centerOrigin', params.BooleanParam, label="Center Origin", default=False,
                      help="Center the volume to the origin", condition="EMfitChoice==1")
        form.addParam('preprocessingVol', params.EnumParam, label="Volume preprocessing", default=0,
                      choices=['None', 'Standard Normal', 'Match values range'],#, 'Match Histograms'],
                      help="Pre-process the input volume to match gray-values of the simulated map"
                           " used in the cryo-EM flexible fitting algorithm. Standard normal will normalize the "
                           " mean and standard deviation of the gray values to match the simulated map. Match values range"
                           " will linearly rescale the gray values range to match the simulated map range. Match histograms"
                           " will match histograms of the target EM and the simulated EM maps", condition="EMfitChoice==1")

        # Images
        form.addParam('inputImage', params.PointerParam, pointerClass="Particle, SetOfParticles",
                      label="Input image (s)", help='Select the target EM density map',
                      condition="EMfitChoice==2")
        form.addParam('image_size', params.IntParam, default=64, label='Image Size',
                      help="TODO", condition="EMfitChoice==2")
        form.addParam('estimateAngleShift', params.BooleanParam, label="Estimate rigid body ?",
                      default=False,  condition="EMfitChoice==2", help="If set, the GUI will perform rigid body alignement. "
                            "Otherwise, you must provide a set of alignement parameters for each image")
        form.addParam('rb_n_iter', params.IntParam, default=1, label='Number of iterations for rigid body fitting',
                      help="Number of rigid body alignement during the simulation. If 1 is set, the rigid body alignement "
                           "will be performed once at the begining of the simulation",
                      condition="EMfitChoice==2 and estimateAngleShift")
        form.addParam('rb_method', params.EnumParam, label="Rigid body alignement method", default=1,
                      choices=['Projection Matching', 'Wavelet'], help="Type of rigid body alignement. "
                                                                       "Wavelet method is recommended",
                      condition="EMfitChoice==2 and estimateAngleShift")
        form.addParam('imageAngleShift', params.FileParam, label="Rigid body parameters (.xmd)",
                      condition="EMfitChoice==2 and not estimateAngleShift",
                      help='Xmipp metadata file of rigid body parameters for each image (3 euler angles, 2 shift)')
        form.addParam('pixel_size', params.FloatParam, default=1.0, label='Pixel size (A)',
                      help="Pixel size of the EM data in Angstrom", condition="EMfitChoice==2")
        # Constraints =================================================================================================
        form.addSection(label='Constraints')
        form.addParam('rigid_bond', params.BooleanParam, label="Rigid bonds",
                      default=False,
                      help="Turn on or off the SHAKE/RATTLE algorithms for covalent bonds involving hydrogen")
        form.addParam('fast_water', params.BooleanParam, label="Fast water",
                      default=False,
                      help="Turn on or off the SETTLE algorithm for the constraints of the water molecules")
        form.addParam('water_model', params.StringParam, label='Water model', default="TIP3",
                      help="Residue name of the water molecule to be rigidified in the SETTLE algorithm", condition="fast_water")

        form.addParallelSection(threads=1, mpi=1)
        # --------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
        self._insertFunctionStep("convertInputPDBStep")
        if self.EMfitChoice.get() != EMFIT_NONE:
            self._insertFunctionStep("convertInputEMStep")
        self._insertFunctionStep("runGenesisStep")
        self._insertFunctionStep("createOutputStep")

    ################################################################################
    ##                 CONVERT INPUT PDB
    ################################################################################

    def convertInputPDBStep(self):

        inputPDBfn = self.getInputPDBfn()
        n_pdb = self.getNumberOfInputPDB()

        # Copy PDBs :
        for i in range(n_pdb):
            runCommand("cp %s %s.pdb"%(inputPDBfn[i],self.getInputPDBprefix(i)))

        # GENERATE TOPOLOGY FILES
        if self.generateTop.get():
            #CHARMM
            if self.forcefield.get() == FORCEFIELD_CHARMM:
                for i in range(n_pdb):
                    prefix = self.getInputPDBprefix(i)
                    generatePSF(inputPDB=prefix+".pdb",inputTopo=self.inputRTF.get(),
                        outputPrefix=prefix, nucleicChoice=self.nucleicChoice.get())

            # GROMACS
            elif self.forcefield.get() == FORCEFIELD_AAGO\
                    or self.forcefield.get() == FORCEFIELD_CAGO:
                self.inputTOPfn = []
                for i in range(n_pdb):
                    prefix = self.getInputPDBprefix(i)
                    generatePSF(inputPDB=prefix+".pdb", inputTopo=self.inputRTF.get(),
                                outputPrefix=prefix, nucleicChoice=self.nucleicChoice.get())
                    generateGROTOP(inputPDB=prefix+".pdb", outputPrefix=prefix,
                                   forcefield=self.forcefield.get(), smog_dir=self.smog_dir.get(),
					nucleicChoice=self.nucleicChoice.get())

        else:
            # CHARMM
            if self.forcefield.get() == FORCEFIELD_CHARMM:
                for i in range(n_pdb):
                    runCommand("cp %s %s.psf" % (self.inputPSF.get(), self.getInputPDBprefix(i)))

            # GROMACS
            elif self.forcefield.get() == FORCEFIELD_AAGO\
                    or self.forcefield.get() == FORCEFIELD_CAGO:
                runCommand("cp %s %s.top" % (self.inputTOP.get(), self.getInputPDBprefix(i)))

        # Center PDB in case of images
        if self.EMfitChoice.get() ==  EMFIT_IMAGES:
            for i in range(n_pdb):
                mol = PDBMol(self.getInputPDBprefix(i)+".pdb")
                mol.center()
                mol.save(self.getInputPDBprefix(i)+".pdb")

    ################################################################################
    ##                 CONVERT INPUT VOLUME/IMAGE
    ################################################################################

    def convertInputEMStep(self):
        # SETUP INPUT VOLUMES / IMAGES

        inputEMfn = self.getInputEMfn()
        n_em = self.getNumberOfInputEM()

        # CONVERT VOLUMES
        if self.EMfitChoice.get() == EMFIT_VOLUMES:
            for i in range(n_em):
                self.convertVolum2Situs(fnInput=inputEMfn[i],
                                   volPrefix = self.getInputEMprefix(i), fnPDB=self.getInputPDBprefix(i)+".pdb")

        # Initialize rigid body fitting parameters
        elif self.EMfitChoice.get() == EMFIT_IMAGES:
            for i in range(n_em):
                runCommand("cp %s %s.spi"%(inputEMfn[i], self.getInputEMprefix(i)))
                if self.estimateAngleShift.get():
                    currentAngles = md.MetaData()
                    currentAngles.setValue(md.MDL_IMAGE, self.getInputEMprefix(i), currentAngles.addObject())
                    currentAngles.setValue(md.MDL_ANGLE_ROT, 0.0, 1)
                    currentAngles.setValue(md.MDL_ANGLE_TILT, 0.0, 1)
                    currentAngles.setValue(md.MDL_ANGLE_PSI, 0.0, 1)
                    currentAngles.setValue(md.MDL_SHIFT_X, 0.0, 1)
                    currentAngles.setValue(md.MDL_SHIFT_Y, 0.0, 1)
                    currentAngles.write(self._getExtraPath("%s_current_angles.xmd" % str(i + 1).zfill(5)))

    def convertVolum2Situs(self,fnInput,volPrefix, fnPDB):

        # CONVERT TO MRC
        pre, ext = os.path.splitext(os.path.basename(fnInput))
        if ext != ".mrc":
            runProgram("xmipp_image_convert", "-i %s --oext mrc -o %s.mrc" %
                        (fnInput,volPrefix))
        else:
            runProgram("cp","%s %s.mrc" %(fnInput,volPrefix))

        # READ INPUT MRC
        with mrcfile.open("%s.mrc" % volPrefix) as input_mrc:
            inputMRCData = input_mrc.data
            inputMRCShape = inputMRCData.shape
            if self.centerOrigin.get():
                origin = -self.voxel_size.get() * (np.array(inputMRCData.shape)) / 2
            else:
                origin = np.zeros(3)
        if self.preprocessingVol.get() == PREPROCESS_VOL_NORM or self.preprocessingVol.get() == PREPROCESS_VOL_OPT:
            # CONVERT PDB TO SITUS VOLUME USING EMMAP GENERATOR
            fnTmpVol = self._getExtraPath("tmp")
            s ="\n[INPUT] \n"
            s +="pdbfile = %s\n" % fnPDB
            s +="\n[OUTPUT] \n"
            s +="mapfile = %s.sit\n" % fnTmpVol
            s +="\n[OPTION] \n"
            s +="map_format = SITUS \n"
            s +="voxel_size = %f \n" % self.voxel_size.get()
            s +="sigma = %f  \n" % self.emfit_sigma.get()
            s +="tolerance = %f  \n"% self.emfit_tolerance.get()
            s +="auto_margin    = NO\n"
            s +="x0             = %f \n" % origin[0]
            s +="y0             = %f \n" % origin[1]
            s +="z0             = %f \n" % origin[2]
            s +="box_size_x     =  %f \n" % (inputMRCShape[0]*self.voxel_size.get())
            s +="box_size_y     =  %f \n" % (inputMRCShape[1]*self.voxel_size.get())
            s +="box_size_z     =  %f \n" % (inputMRCShape[2]*self.voxel_size.get())
            with open("%s_INP_emmap" % fnTmpVol, "w") as f:
                f.write(s)

            runCommand("emmap_generator %s_INP_emmap"% fnTmpVol, env=self.getGenesisEnv())

            # CONVERT SITUS TMP FILE TO MRC
            with open(self._getExtraPath("runconvert.sh"), "w") as f:
                f.write("#!/bin/bash \n")
                f.write("echo $PATH \n")
                f.write("map2map %s.sit %s.mrc <<< \'1\'\n" % (fnTmpVol, fnTmpVol))
                f.write("exit")
            runCommand("/bin/bash %s" %self._getExtraPath("runconvert.sh"), env=self.getGenesisEnv())

            # READ GENERATED MRC
            with mrcfile.open(fnTmpVol+".mrc") as tmp_mrc:
                tmpMRCData = tmp_mrc.data

            # PREPROCESS VOLUME
            if self.preprocessingVol.get() == PREPROCESS_VOL_NORM:
                mrc_data =  ((inputMRCData-inputMRCData.mean())/inputMRCData.std())\
                            *tmpMRCData.std() + tmpMRCData.mean()
            elif self.preprocessingVol.get() == PREPROCESS_VOL_OPT:
                min1 = tmpMRCData.min()
                min2 = inputMRCData.min()
                max1 = tmpMRCData.max()
                max2 = inputMRCData.max()
                mrc_data = ((inputMRCData - (min2 + min1))*(max1 - min1) )/ (max2 - min2)
            # elif self.preprocessingVol.get() == PREPROCESS_VOL_MATCH:
            #     mrc_data = match_histograms(inputMRCData, tmpMRCData)

            # CLEANING
            runProgram("rm", "-f %s.sit" % fnTmpVol)
            runProgram("rm", "-f %s.mrc" % fnTmpVol)
            runProgram("rm", "-f %s_INP_emmap" % fnTmpVol)
        else:
            mrc_data = inputMRCData

        # SAVE TO MRC
        with mrcfile.new("%sConv.mrc"%volPrefix, overwrite=True) as mrc:
            mrc.set_data(np.float32(mrc_data))
            mrc.voxel_size = self.voxel_size.get()
            mrc.header['origin']['x'] = origin[0]
            mrc.header['origin']['y'] = origin[1]
            mrc.header['origin']['z'] = origin[2]
            mrc.update_header_from_data()
            mrc.update_header_stats()

        # CONVERT MRC TO SITUS
        with open(self._getExtraPath("runconvert.sh"), "w") as f:
            f.write("#!/bin/bash \n")
            f.write("map2map %sConv.mrc %s.sit <<< \'1\'\n" % (volPrefix,volPrefix))
            f.write("exit")
        runCommand( "/bin/bash %s " %self._getExtraPath("runconvert.sh"), env=self.getGenesisEnv())


        runProgram("rm","-f %s"%self._getExtraPath("runconvert.sh"))
        runProgram("rm","-f %sConv.mrc"%volPrefix)
        runProgram("rm","-f %s.mrc" % volPrefix)


    ################################################################################
    ##                 GENESIS STEP
    ################################################################################

    def runGenesisStep(self):
        rb_condition = self.EMfitChoice.get() == EMFIT_IMAGES and self.estimateAngleShift.get()

        # Parallel Genesis simulation
        if not(rb_condition):
            self.runParallelGenesis()

        # Parallel rigid body fitting for EMFIT images
        else:
            self.runParallelGenesisRBFitting()

    def runParallelGenesis(self):

        # SETUP MPI parameters
        numMpiPerFit, numLinearFit, numParallelFit, numLastIter = self.getMPIParams()

        for i1 in range(numLinearFit + 1):
            cmds = []
            n_parallel = numParallelFit if i1 < numLinearFit else numLastIter
            for i2 in range(n_parallel):
                indexFit = i2 + i1 * numParallelFit
                prefix = self.getOutputPrefix(indexFit)

                # Create INP file
                self.createINP(inputPDB=self.getInputPDBprefix(indexFit) + ".pdb",
                               outputPrefix=prefix, indexFit=indexFit)

                # Create Genesis command
                cmds.append(self.getGenesisCmd(prefix=prefix, n_mpi=numMpiPerFit))

            # Run Genesis
            self.runParallelJobs(cmds)

    def runParallelGenesisRBFitting(self):

        # SETUP MPI parameters
        numMpiPerFit, numLinearFit, numParallelFit, numLastIter = self.getMPIParams()

        for i1 in range(numLinearFit + 1):
            n_parallel = numParallelFit if i1 < numLinearFit else numLastIter

            # Loop rigidbody align / GENESIS fitting
            for iterFit in range(self.rb_n_iter.get()):

                # ------   ALIGN PDBs---------
                # Transform PDBs to volume
                cmds_pdb2vol = []
                for i2 in range(n_parallel):
                    indexFit = i2 + i1 * numParallelFit
                    inputPDB = self.getInputPDBprefix(indexFit) + ".pdb" if iterFit == 0 \
                        else self.getOutputPrefix(indexFit) + ".pdb"

                    tmpPrefix = self._getExtraPath("%s_tmp" % str(indexFit + 1).zfill(5))
                    cmds_pdb2vol.append(pdb2vol(inputPDB=inputPDB, outputVol=tmpPrefix,
                                                sampling_rate=self.pixel_size.get(),
                                                image_size=self.image_size.get()))
                self.runParallelJobs(cmds_pdb2vol)

                # Loop 4 times to refine the angles
                # sampling_rate = [10.0, 5.0, 3.0, 2.0]
                # angular_distance = [-1, 20, 10, 5]
                sampling_rate = [10.0]
                angular_distance = [-1]
                for i_align in range(len(sampling_rate)):
                    cmds_projectVol = []
                    cmds_alignement = []
                    for i2 in range(n_parallel):
                        indexFit = i2 + i1 * numParallelFit
                        tmpPrefix = self._getExtraPath("%s_tmp" % str(indexFit + 1).zfill(5))
                        inputImage = self.getInputEMprefix(indexFit) + ".spi"
                        tmpMeta = self._getExtraPath("%s_tmp_angles.xmd" % str(indexFit + 1).zfill(5))
                        currentAngles = self._getExtraPath("%s_current_angles.xmd" % str(indexFit + 1).zfill(5))

                        # get commands
                        if self.rb_method.get() == RB_PROJMATCH:
                            cmds_projectVol.append(projectVol(inputVol=tmpPrefix,
                                                              outputProj=tmpPrefix, expImage=inputImage,
                                                              sampling_rate=sampling_rate[i_align],
                                                              angular_distance=angular_distance[i_align]))
                            cmds_alignement.append(projectMatch(inputImage=inputImage,
                                                                inputProj=tmpPrefix, outputMeta=tmpMeta))
                        else:
                            cmds_projectVol.append(projectVol(inputVol=tmpPrefix,
                                                              outputProj=tmpPrefix, expImage=inputImage,
                                                              sampling_rate=sampling_rate[i_align],
                                                              angular_distance=angular_distance[i_align],
                                                              compute_neighbors=False))
                            cmds_alignement.append(waveletAssignement(inputImage=inputImage,
                                                                      inputProj=tmpPrefix, outputMeta=tmpMeta))
                    # run parallel jobs
                    self.runParallelJobs(cmds_projectVol)
                    self.runParallelJobs(cmds_alignement)

                    cmds_continuousAssign = []
                    for i2 in range(n_parallel):
                        indexFit = i2 + i1 * numParallelFit
                        tmpPrefix = self._getExtraPath("%s_tmp" % str(indexFit + 1).zfill(5))
                        tmpMeta = self._getExtraPath("%s_tmp_angles.xmd" % str(indexFit + 1).zfill(5))
                        currentAngles = self._getExtraPath("%s_current_angles.xmd" % str(indexFit + 1).zfill(5))
                        if self.rb_method.get() == RB_PROJMATCH:
                            flipAngles(inputMeta=tmpMeta, outputMeta=tmpMeta)
                        cmds_continuousAssign.append(continuousAssign(inputMeta=tmpMeta,
                                                                      inputVol=tmpPrefix,
                                                                      outputMeta=currentAngles))
                    self.runParallelJobs(cmds_continuousAssign)

                # Cleaning volumes and projections
                for i2 in range(n_parallel):
                    indexFit = i2 + i1 * numParallelFit
                    tmpPrefix = self._getExtraPath("%s_tmp" % str(indexFit + 1).zfill(5))
                    runCommand("rm -f %s*" % tmpPrefix)

                # ------   Run Genesis ---------
                cmds = []
                for i2 in range(n_parallel):
                    indexFit = i2 + i1 * numParallelFit
                    if iterFit == 0:
                        prefix = self.getOutputPrefix(indexFit)
                        inputPDB = self.getInputPDBprefix(indexFit) + ".pdb"
                    else:
                        prefix = self._getExtraPath("%s_tmp" % str(indexFit + 1).zfill(5))
                        inputPDB = self.getOutputPrefix(indexFit) + ".pdb"

                    # Create INP file
                    self.createINP(inputPDB=inputPDB,
                                   outputPrefix=prefix, indexFit=indexFit)

                    # run GENESIS
                    cmds.append(self.getGenesisCmd(prefix=prefix, n_mpi=numMpiPerFit))
                self.runParallelJobs(cmds)

                # append files
                if iterFit != 0:
                    for i2 in range(n_parallel):
                        indexFit = i2 + i1 * numParallelFit
                        tmpPrefix = self._getExtraPath("%s_tmp" % str(indexFit + 1).zfill(5))
                        newPrefix = self.getOutputPrefix(indexFit)

                        cat_cmd = "cat %s.log >> %s.log" % (tmpPrefix, newPrefix)
                        tcl_cmd = "animate read dcd %s.dcd waitfor all\n" % (newPrefix)
                        tcl_cmd += "animate read dcd %s.dcd waitfor all\n" % (tmpPrefix)
                        tcl_cmd += "animate write dcd %s.dcd \nexit \n" % newPrefix
                        with open("%s.tcl" % tmpPrefix, "w") as f:
                            f.write(tcl_cmd)
                        cp_cmd = "cp %s.pdb %s.pdb" % (tmpPrefix, newPrefix)
                        runCommand(cat_cmd)
                        runCommand(cp_cmd)
                        runCommand("vmd -dispdev text -e %s.tcl" % tmpPrefix)

                rstfile = ""
                for i2 in range(n_parallel):
                    indexFit = i2 + i1 * numParallelFit
                    newPrefix = self.getOutputPrefix(indexFit)
                    if iterFit != 0:
                        tmpPrefix = self._getExtraPath("%s_tmp" % str(indexFit + 1).zfill(5))
                    else:
                        tmpPrefix = self.getOutputPrefix(indexFit)

                    runCommand("cp %s.rst %s.tmp.rst" % (tmpPrefix, newPrefix))
                    rstfile += "%s.tmp.rst "%newPrefix
                    #save angles
                    angles = self._getExtraPath("%s_current_angles.xmd" % str(indexFit + 1).zfill(5))
                    saved_angles = self._getExtraPath("%s_iter%i_angles.xmd" % (str(indexFit + 1).zfill(5), iterFit))
                    runCommand("cp %s %s" % (angles, saved_angles))

                    #cleaning
                    runCommand("rm -rf %s" %self._getExtraPath("%s_tmp" % str(indexFit + 1).zfill(5)))
                self.inputRST.set(rstfile)
        self.inputRST.set("")




    def runParallelJobs(self, cmds):
        # Set env
        env = self.getGenesisEnv()
        env["OMP_NUM_THREADS"] = str(self.numberOfThreads)

        # run process
        processes = []
        for cmd in cmds:
            print("Running command : %s" %cmd)
            processes.append(Popen(cmd, shell=True, env=env, stdout=sys.stdout, stderr = sys.stderr))

        # Wait for processes
        for p in processes:
            exitcode = p.wait()
            print("Process done %s" %str(exitcode))
            if exitcode != 0:
                raise RuntimeError("Command returned with errors : %s" %str(cmds))

    def createINP(self,inputPDB, outputPrefix, indexFit):
        """
        Create INP input file for GENESIS
        :param str inputPDB: input PDB file name
        :param str outputPrefix: output prefix
        :param int indexFit: index of the simulation
        :return None:
        """
        inputPDBprefix = self.getInputPDBprefix(indexFit)
        inputEMprefix = self.getInputEMprefix(indexFit)
        inp_file = "%s_INP"% outputPrefix

        s = "\n[INPUT] \n" #-----------------------------------------------------------
        s += "pdbfile = %s\n" % inputPDB
        if self.forcefield.get() == FORCEFIELD_CHARMM:
            s += "topfile = %s\n" % self.inputRTF.get()
            s += "parfile = %s\n" % self.inputPRM.get()
            s += "psffile = %s.psf\n" % inputPDBprefix
            if self.inputSTR.get() != "" and self.inputSTR.get() is not None:
                s += "strfile = %s\n" % self.inputSTR.get()
        elif self.forcefield.get() == FORCEFIELD_AAGO\
                or self.forcefield.get() == FORCEFIELD_CAGO:
            s += "grotopfile = %s.top\n" % inputPDBprefix
        if self.inputRST.get() != "" and self.inputRST.get() is not None:
            s += "rstfile = %s\n" % self.getRestartFile(indexFit)

        s += "\n[OUTPUT] \n" #-----------------------------------------------------------
        if self.simulationType.get() == SIMULATION_REMD:
            s += "remfile = %s_remd{}.rem\n" %outputPrefix
            s += "logfile = %s_remd{}.log\n" %outputPrefix
            s += "dcdfile = %s_remd{}.dcd\n" %outputPrefix
            s += "rstfile = %s_remd{}.rst\n" %outputPrefix
            s += "pdbfile = %s_remd{}.pdb\n" %outputPrefix
        else:
            s += "dcdfile = %s.dcd\n" %outputPrefix
            s += "rstfile = %s.rst\n" %outputPrefix
            s += "pdbfile = %s.pdb\n" %outputPrefix

        s += "\n[ENERGY] \n" #-----------------------------------------------------------
        if self.forcefield.get() == FORCEFIELD_CHARMM:
            s += "forcefield = CHARMM \n"
        elif self.forcefield.get() == FORCEFIELD_AAGO:
            s += "forcefield = AAGO  \n"
        elif self.forcefield.get() == FORCEFIELD_CAGO:
            s += "forcefield = CAGO  \n"

        if self.electrostatics.get() == ELECTROSTATICS_CUTOFF :
            s += "electrostatic = CUTOFF  \n"
        else:
            s += "electrostatic = PME  \n"
        s += "switchdist   = %.2f \n" % self.switch_dist.get()
        s += "cutoffdist   = %.2f \n" % self.cutoff_dist.get()
        s += "pairlistdist = %.2f \n" % self.pairlist_dist.get()
        if self.vdw_force_switch.get():
            s += "vdw_force_switch = YES \n"
        if self.implicitSolvent.get() == IMPLICIT_SOLVENT_GBSA:
            s += "implicit_solvent = GBSA \n"
            s += "gbsa_eps_solvent = 78.5 \n"
            s += "gbsa_eps_solute  = 1.0 \n"
            s += "gbsa_salt_cons   = 0.2 \n"
            s += "gbsa_surf_tens   = 0.005 \n"

        if self.simulationType.get() == SIMULATION_MIN:
            s += "\n[MINIMIZE]\n" #-----------------------------------------------------------
            s += "method = SD\n"
        else:
            s += "\n[DYNAMICS] \n" #-----------------------------------------------------------
            if self.integrator.get() == INTEGRATOR_VVERLET:
                s += "integrator = VVER  \n"
            elif self.integrator.get() == INTEGRATOR_LEAPFROG:
                s += "integrator = LEAP  \n"
            elif self.integrator.get() == INTEGRATOR_NMMD:
                s += "integrator = NMMD  \n"
            s += "timestep = %f \n" % self.time_step.get()
        s += "nsteps = %i \n" % self.n_steps.get()
        s += "eneout_period = %i \n" % self.eneout_period.get()
        s += "crdout_period = %i \n" % self.crdout_period.get()
        s += "rstout_period = %i \n" % self.n_steps.get()
        s += "nbupdate_period = %i \n" % self.nbupdate_period.get()

        if self.integrator.get() == INTEGRATOR_NMMD:
            s += "\n[NMMD] \n" #-----------------------------------------------------------
            s+= "nm_number = %i \n" % self.nm_number.get()
            s+= "nm_mass = %f \n" % self.nm_mass.get()
            s+= "nm_limit = %f \n" % self.nm_limit.get()
            s+= "elnemo_cutoff = %f \n" % self.elnemo_cutoff.get()
            s+= "elnemo_rtb_block = %i \n" % self.elnemo_rtb_block.get()
            s+= "elnemo_path = %s \n" %  Plugin.getVar("NMA_HOME")
            if self.simulationType.get() == SIMULATION_REMD:
                s+= "nm_prefix = %s_remd{} \n" % outputPrefix
            else:
                s += "nm_prefix = %s \n" % outputPrefix

        s += "\n[CONSTRAINTS] \n" #-----------------------------------------------------------
        if self.rigid_bond.get()        : s += "rigid_bond = YES \n"
        else                            : s += "rigid_bond = NO \n"
        if self.fast_water.get()      :
            s += "fast_water = YES \n"
            s += "water_model = %s \n" %self.water_model.get()
        else                            : s += "fast_water = NO \n"

        s += "\n[BOUNDARY] \n" #-----------------------------------------------------------
        if self.boundary.get() == BOUNDARY_PBC:
            s += "type = PBC \n"
            s += "box_size_x = %f \n" % self.box_size_x.get()
            s += "box_size_y = %f \n" % self.box_size_y.get()
            s += "box_size_z = %f \n" % self.box_size_z.get()
        else :
            s += "type = NOBC \n"

        s += "\n[ENSEMBLE] \n" #-----------------------------------------------------------
        if self.ensemble.get() == ENSEMBLE_NVE:
            s += "ensemble = NVE  \n"
        elif self.ensemble.get() == ENSEMBLE_NPT:
            s += "ensemble = NPT  \n"
        else:
            s += "ensemble = NVT  \n"
        if self.tpcontrol.get() == TPCONTROL_LANGEVIN:
            s += "tpcontrol = LANGEVIN  \n"
        elif self.tpcontrol.get() == TPCONTROL_BERENDSEN:
            s += "tpcontrol = BERENDSEN  \n"
        elif self.tpcontrol.get() == TPCONTROL_BUSSI:
            s += "tpcontrol = BUSSI  \n"
        else:
            s += "tpcontrol = NO  \n"
        s += "temperature = %.2f \n" % self.temperature.get()
        if self.ensemble.get() == ENSEMBLE_NPT:
            s += "pressure = %.2f \n" % self.pressure.get()

        if (self.EMfitChoice.get()==EMFIT_VOLUMES or self.EMfitChoice.get()==EMFIT_IMAGES)\
                and self.simulationType.get() != SIMULATION_MIN:
            s += "\n[SELECTION] \n" #-----------------------------------------------------------
            s += "group1 = all and not hydrogen\n"

            s += "\n[RESTRAINTS] \n" #-----------------------------------------------------------
            s += "nfunctions = 1 \n"
            s += "function1 = EM \n"
            s += "constant1 = %s \n" % self.constantK.get()
            s += "select_index1 = 1 \n"

            s += "\n[EXPERIMENTS] \n" #-----------------------------------------------------------
            s += "emfit = YES  \n"
            s += "emfit_sigma = %.4f \n" % self.emfit_sigma.get()
            s += "emfit_tolerance = %.6f \n" % self.emfit_tolerance.get()
            s += "emfit_period = 1  \n"
            if self.EMfitChoice.get() == EMFIT_VOLUMES:
                s += "emfit_target = %s.sit \n" % inputEMprefix
            elif self.EMfitChoice.get()==EMFIT_IMAGES :
                s += "emfit_type = IMAGE \n"
                s += "emfit_target = %s.spi \n" % inputEMprefix
                s += "emfit_pixel_size =  %f\n" % self.pixel_size.get()
                rigid_body_params = self.getRigidBodyParams(indexFit)
                s += "emfit_roll_angle = %f\n" % rigid_body_params[0]
                s += "emfit_tilt_angle = %f\n" % rigid_body_params[1]
                s += "emfit_yaw_angle =  %f\n" % rigid_body_params[2]
                s += "emfit_shift_x = %f\n" % rigid_body_params[3]
                s += "emfit_shift_y =  %f\n" % rigid_body_params[4]

            if self.simulationType.get() == SIMULATION_REMD:
                s += "\n[REMD] \n" #-----------------------------------------------------------
                s += "dimension = 1 \n"
                s += "exchange_period = %i \n" % self.exchange_period.get()
                s += "type1 = RESTRAINT \n"
                s += "nreplica1 = %i \n" % self.nreplica.get()
                s += "rest_function1 = 1 \n"

        with open(inp_file, "w") as f:
            f.write(s)

    ################################################################################
    ##                 CREATE OUTPUT STEP
    ################################################################################

    def createOutputStep(self):
        """
        Create output set of PDBs
        :return None:
        """
        # CREATE SET OF PDBs
        pdbset = self._createSetOfPDBs("outputPDBs")

        # Add each output PDB to the Set
        for i in range(self.getNumberOfFitting()):

            # Extract the pdb from the DCD file in case of SPDYN
            if self.md_program.get() == PROGRAM_SPDYN:
                lastPDBFromDCD(
                    inputDCD=self.getOutputPrefix(i)+ ".dcd",
                    outputPDB=self.getOutputPrefix(i)+ ".pdb",
                    inputPDB=self.getInputPDBprefix(i)+".pdb")

            outputPrefix =self.getOutputPrefixAll(i)
            for j in outputPrefix:
                pdbset.append(AtomStruct(j + ".pdb"))
        self._defineOutputs(outputPDBs=pdbset)

    # --------------------------- STEPS functions --------------------------------------------
    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        return summary

    def _validate(self):
        errors = []
        print(self.getGenesisEnv()["PATH"])
        if not os.path.exists(os.path.join(
                Plugin.getVar("GENESIS_HOME"), 'bin/atdyn')):
            errors.append("Missing GENESIS program : atdyn ")

        if not os.path.exists(os.path.join(
                Plugin.getVar("GENESIS_HOME"), 'bin/spdyn')):
            errors.append("Missing GENESIS program : spdyn ")

        return errors

    def _citations(self):
        pass

    def _methods(self):
        pass

    # --------------------------- UTILS functions --------------------------------------------


    def getNumberOfInputPDB(self):
        """
        Get the number of input PDBs
        :return int: number of input PDBs
        """
        if isinstance(self.inputPDB.get(), SetOfAtomStructs) or \
                isinstance(self.inputPDB.get(), SetOfPDBs):
            return self.inputPDB.get().getSize()
        else: return 1

    def getNumberOfInputEM(self):
        """
        Get the number of input EM data to analyze
        :return int : number of input EM data
        """
        if self.EMfitChoice.get() == EMFIT_VOLUMES:
            if isinstance(self.inputVolume.get(), SetOfVolumes): return self.inputVolume.get().getSize()
            else: return 1
        elif self.EMfitChoice.get() == EMFIT_IMAGES:
            if isinstance(self.inputImage.get(), SetOfParticles): return self.inputImage.get().getSize()
            else: return 1
        else: return 0

    def getNumberOfFitting(self):
        """
        Get the number of simulations to perform
        :return int: Number of simulations
        """
        numberOfInputPDB = self.getNumberOfInputPDB()
        numberOfInputEM = self.getNumberOfInputEM()

        # Check input volumes/images correspond to input PDBs
        if numberOfInputPDB != numberOfInputEM and \
                numberOfInputEM != 1 and numberOfInputPDB != 1:
            raise RuntimeError("Number of input volumes and PDBs must be the same.")
        return np.max([numberOfInputEM, numberOfInputPDB])

    def getInputPDBfn(self):
        """
        Get the input PDB file names
        :return list : list of input PDB file names
        """
        initFn = []
        if isinstance(self.inputPDB.get(), SetOfAtomStructs) or \
                isinstance(self.inputPDB.get(), SetOfPDBs):
            for i in range(self.inputPDB.get().getSize()):
                initFn.append(self.inputPDB.get()[i+1].getFileName())

        else:
            initFn.append(self.inputPDB.get().getFileName())
        return initFn

    def getInputEMfn(self):
        """
        Get the input EM data file names
        :return list: list of input EM data file names
        """
        inputEMfn = []
        if self.EMfitChoice.get() == EMFIT_VOLUMES:
            if isinstance(self.inputVolume.get(), SetOfVolumes) :
                for i in self.inputVolume.get():
                    inputEMfn.append(i.getFileName())
            else:
                inputEMfn.append(self.inputVolume.get().getFileName())
        elif self.EMfitChoice.get() == EMFIT_IMAGES:
            if isinstance(self.inputImage.get(), SetOfParticles) :
                for i in self.inputImage.get():
                    inputEMfn.append(i.getFileName())
            else:
                inputEMfn.append(self.inputImage.get().getFileName())
        return inputEMfn

    def getInputPDBprefix(self, index=0):
        """
        Get the input PDB prefix of the specified index
        :param int index: index of input PDB
        :return str: Input PDB prefix
        """
        prefix = self._getExtraPath("%s_inputPDB")
        if self.getNumberOfInputPDB() == 1:
            return prefix % str(1).zfill(5)
        else:
            return prefix % str(index + 1).zfill(5)

    def getInputEMprefix(self, index=0):
        """
        Get the input EM data prefix of the specified index
        :param int index: index of the EM data
        :return str: Input EM data prefix
        """
        prefix = self._getExtraPath("%s_inputEM")
        if self.getNumberOfInputEM() == 0:
            return ""
        elif self.getNumberOfInputEM() == 1:
            return prefix % str(1).zfill(5)
        else:
            return prefix % str(index + 1).zfill(5)


    def getOutputPrefix(self, index=0):
        """
        Output prefix of the specified index
        :param int index: index of the simulation to get
        :return string : Output prefix of the specified index
        """
        return self._getExtraPath("%s_output" % str(index + 1).zfill(5))

    def getOutputPrefixAll(self, index=0):
        """
        All output prefix of the specified index including multiple replicas in case of REUS
        :param int index: index of the simulation to get
        :return list: list of all output prefix of the specified index
        """
        outputPrefix=[]
        if self.simulationType.get() == SIMULATION_REMD:
            for i in range(self.nreplica.get()):
                outputPrefix.append(self._getExtraPath("%s_output_remd%i" %
                                (str(index + 1).zfill(5), i + 1)))
        else:
            outputPrefix.append(self._getExtraPath("%s_output" % str(index + 1).zfill(5)))
        return outputPrefix

    def getMPIParams(self):
        """
        Get mpi parameters for the simulation
        :return tuple: numberOfMpiPerFit, numberOfLinearFit, numberOfParallelFit, numberOflastIter
        """
        n_fit = self.getNumberOfFitting()
        if n_fit <= self.numberOfMpi.get():
            return self.numberOfMpi.get()//n_fit, 1, n_fit, 0
        else:
            return 1, n_fit//self.numberOfMpi.get(),  self.numberOfMpi.get(), n_fit % self.numberOfMpi.get()

    def getRigidBodyParams(self, index=0):
        """
        Get the current rigid body parameters for the specified index in case of EMFIT with iamges
        :param int index: Index of the simulation
        :return list: angle_rot, angle_tilt, angle_psi, shift_x, shift_y
        """
        if not self.estimateAngleShift.get():
            mdImg = md.MetaData(self.imageAngleShift.get())
            idx = int(index + 1)
        else:
            mdImg = md.MetaData(self._getExtraPath("%s_current_angles.xmd" % str(index + 1).zfill(5)))
            idx=1
        return [
            mdImg.getValue(md.MDL_ANGLE_ROT, idx),
            mdImg.getValue(md.MDL_ANGLE_TILT, idx),
            mdImg.getValue(md.MDL_ANGLE_PSI, idx),
            mdImg.getValue(md.MDL_SHIFT_X, idx),
            mdImg.getValue(md.MDL_SHIFT_Y, idx),
        ]

    def getGenesisEnv(self):
        """
        Get environnement for running GENESIS
        :return Environ: environnement
        """
        environ = pwutils.Environ(os.environ)
        environ.set('PATH', os.path.join(Plugin.getVar("GENESIS_HOME"), 'bin'),
                    position=pwutils.Environ.BEGIN)
        environ.set('PATH', os.path.join(Plugin.getVar("SITUS_HOME"), 'bin'),
                    position=pwutils.Environ.BEGIN)
        return environ

    def getGenesisCmd(self, prefix,n_mpi):
        """
        Get GENESIS cmd to run
        :param str prefix: prefix of the simulation
        :param int n_mpi: number of MPI processes
        :return str : GENESIS commadn to run
        """
        cmd=""
        if (n_mpi != 1):
            cmd += "mpirun -np %s " % n_mpi
        if self.md_program.get() == PROGRAM_ATDYN:
            cmd +=  "atdyn %s " % ("%s_INP" % prefix)
        else:
            cmd += "spdyn %s " % ("%s_INP" % prefix)
        cmd += "  > %s.log" % prefix
        return cmd

    def getRestartFile(self, index=0):
        rstfile = self.inputRST.get()
        rstList = rstfile.split(" ")
        if len(rstList) >1:
            return rstList[index]
        else:
            return rstList[0]