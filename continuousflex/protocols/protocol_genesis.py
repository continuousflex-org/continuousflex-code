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
from skimage.exposure import match_histograms
import pwem.emlib.metadata as md
from pwem.utils import runProgram
from subprocess import Popen
from xmippLib import Euler_angles2matrix

from .utilities.genesis_utilities import PDBMol,generatePSF, generateGROTOP

EMFIT_NONE = 0
EMFIT_VOLUMES = 1
EMFIT_IMAGES = 2

FORCEFIELD_CHARMM = 0
FORCEFIELD_AAGO = 1
FORCEFIELD_CAGO = 2

SIMULATION_MD = 0
SIMULATION_MIN = 1

INTEGRATOR_VVERLET = 0
INTEGRATOR_LEAPFROG = 1

IMPLICIT_SOLVENT_GBSA = 0
IMPLICIT_SOLVENT_NONE = 1

TPCONTROL_LANGEVIN = 0
TPCONTROL_BERENDSEN = 1
TPCONTROL_NONE = 2

NUCLEIC_NO = 0
NUCLEIC_RNA =1
NUCLEIC_DNA = 2


class ProtGenesis(EMProtocol):
    """ Protocol for the molecular dynamics software GENESIS. """
    _label = 'Genesis'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):

        # Inputs ============================================================================================
        form.addSection(label='Inputs')
        form.addParam('genesisDir', params.FileParam, label="Genesis install path",
                      help='Path to genesis installation', important=True)
        form.addParam('inputPDB', params.PointerParam,
                      pointerClass='AtomStruct, SetOfPDBs, SetOfAtomStructs', label="Input PDB (s)",
                      help='Select the input PDB or set of PDBs.')
        form.addParam('forcefield', params.EnumParam, label="Forcefield type", default=0,
                      choices=['CHARMM', 'AAGO', 'CAGO'], help="TODo")
        form.addParam('generateTop', params.BooleanParam, label="Generate topology files ?",
                      default=False, help="TODo")
        form.addParam('smog_dir', params.FileParam, label="SMOG2 directory",
                      help='TODO', condition="(forcefield==1 or forcefield==2) and generateTop")
        form.addParam('inputTOP', params.FileParam, label="GROMACS Topology File (.top)",
                      condition="(forcefield==1 or forcefield==2) and not generateTop",
                      help='TODO')
        form.addParam('inputPRM', params.FileParam, label="CHARMM Parameter File (.prm)",
                      condition = "forcefield==0",
                      help='CHARMM force field parameter file (.prm). Can be founded at ' +
                           'http://mackerell.umaryland.edu/charmm_ff.shtml#charmm')
        form.addParam('inputRTF', params.FileParam, label="CHARMM Topology File (.rtf)",
                      condition="forcefield==0 or ((forcefield==1 or forcefield==2) and generateTop)",
                      help='CHARMM force field topology file (.rtf). Can be founded at ' +
                           'http://mackerell.umaryland.edu/charmm_ff.shtml#charmm. '+
                           'In the case of AAGO/CAGO model, used for completing the missing structure')
        form.addParam('nucleicChoice', params.EnumParam, label="Contains nucleic acids ?", default=0,
                      choices=['NO', 'RNA', 'DNA'], condition ="generateTop",help="TODo")

        form.addParam('inputPSF', params.FileParam, label="Protein Structure File (.psf)",
                      condition="forcefield==0 and not generateTop",
                      help='TODO')

        form.addParam('restartchoice', params.BooleanParam, label="Restart previous run ?", default=False,
                     help="TODo")
        form.addParam('inputRST', params.FileParam, label="GENESIS Restart File (.rst)",
                       help='Restart file from previous minimisation or MD run '
                      , condition="restartchoice")


        # Simulation =================================================================================================
        form.addSection(label='Simulation')
        form.addParam('simulationType', params.EnumParam, label="Simulation type", default=0,
                      choices=['Molecular Dynamics', 'Minimization'],  help="TODO", important=True)
        form.addParam('integrator', params.EnumParam, label="Integrator", default=0,
                      choices=['Velocity Verlet', 'Leapfrog'],  help="TODO", condition="simulationType==0")
        form.addParam('time_step', params.FloatParam, default=0.002, label='Time step (ps)',
                      help="TODO", condition="simulationType==0")
        form.addParam('n_steps', params.IntParam, default=10000, label='Number of steps',
                      help="Select the number of steps in the MD fitting")
        form.addParam('eneout_period', params.IntParam, default=100, label='Energy output period',
                      help="TODO")
        form.addParam('crdout_period', params.IntParam, default=100, label='Coordinates output period',
                      help="TODO")
        form.addParam('nbupdate_period', params.IntParam, default=10, label='Non-bonded update period',
                      help="TODO")
        # ENERGY =================================================================================================
        form.addSection(label='Energy')
        form.addParam('implicitSolvent', params.EnumParam, label="Implicit Solvent", default=1,
                      choices=['GBSA', 'NONE'],
                      help="TODo")
        form.addParam('switch_dist', params.FloatParam, default=10.0, label='Switch Distance', help="TODO")
        form.addParam('cutoff_dist', params.FloatParam, default=12.0, label='Cutoff Distance', help="TODO")
        form.addParam('pairlist_dist', params.FloatParam, default=15.0, label='Pairlist Distance', help="TODO")
        form.addParam('tpcontrol', params.EnumParam, label="Temperature control", default=0,
                      choices=['LANGEVIN', 'BERENDSEN', 'NO'],
                      help="TODo")
        form.addParam('temperature', params.FloatParam, default=300.0, label='Temperature (K)',
                      help="TODO")
        # EM fit =================================================================================================
        form.addSection(label='EM fit')
        form.addParam('EMfitChoice', params.EnumParam, label="Cryo-EM Flexible Fitting", default=0,
                      choices=['None', 'Volume (s)', 'Image (s)'], important=True,
                      help="TODO")
        form.addParam('constantK', params.IntParam, default=10000, label='Force constant K',
                      help="TODO", condition="EMfitChoice!=0")
        form.addParam('emfit_sigma', params.FloatParam, default=2.0, label="EMfit Sigma",
                      help="TODO", condition="EMfitChoice!=0")
        form.addParam('emfit_tolerance', params.FloatParam, default=0.01, label='EMfit Tolerance',
                      help="TODO", condition="EMfitChoice!=0")

        # Volumes
        form.addParam('inputVolume', params.PointerParam, pointerClass="Volume, SetOfVolumes",
                      label="Input volume (s)", help='Select the target EM density volume',
                      condition="EMfitChoice==1")
        form.addParam('voxel_size', params.FloatParam, default=1.0, label='Voxel size (A)',
                      help="TODO", condition="EMfitChoice==1")
        form.addParam('situs_dir', params.FileParam,
                      label="Situs install path", help='Select the root directory of Situs installation'
                      , condition="EMfitChoice==1")
        form.addParam('centerOrigin', params.BooleanParam, label="Center Origin", default=False,
                      help="TODo", condition="EMfitChoice==1")

        # Images
        form.addParam('inputImage', params.PointerParam, pointerClass="Particle, SetOfParticles",
                      label="Input image (s)", help='Select the target EM density map',
                      condition="EMfitChoice==2")
        form.addParam('image_size', params.IntParam, default=64, label='Image Size',
                      help="TODO", condition="EMfitChoice==2")
        form.addParam('estimateAngleShift', params.BooleanParam, label="Estimate rigid body ?",
                      default=False,  condition="EMfitChoice==2", help="TODO")
        form.addParam('n_iter', params.IntParam, default=10, label='Number of iterations for rigid body fitting',
                      help="TODO", condition="EMfitChoice==2 and estimateAngleShift")
        form.addParam('imageAngleShift', params.FileParam, label="Rigid body parameters (.xmd)",
                      condition="EMfitChoice==2 and not estimateAngleShift",
                      help='TODO')
        form.addParam('pixel_size', params.FloatParam, default=1.0, label='Pixel size (A)',
                      help="TODO", condition="EMfitChoice==2")

        # NMMD =================================================================================================
        form.addSection(label='NMMD')
        form.addParam('normalModesChoice', params.BooleanParam, label="Normal Mode Molecular Dynamics",
                      default=False, important=True, help="TODO")
        form.addParam('n_modes', params.IntParam, default=10, label='Number of normal modes',
                      help="TODO", condition="normalModesChoice")
        form.addParam('global_mass', params.FloatParam, default=1.0, label='Normal modes amplitude mass',
                      help="TODO", condition="normalModesChoice")
        form.addParam('global_limit', params.FloatParam, default=300.0, label='Normal mode amplitude threshold',
                      help="TODO", condition="normalModesChoice")
        # REMD =================================================================================================
        form.addSection(label='REMD')
        form.addParam('replica_exchange', params.BooleanParam, label="Replica Exchange",
                      default=False, important=True,
                      help="TODO")
        form.addParam('exchange_period', params.IntParam, default=1000, label='Exchange Period',
                      help="TODO", condition="replica_exchange")
        form.addParam('nreplica', params.IntParam, default=1, label='Number of replicas',
                      help="TODO", condition="replica_exchange")
        form.addParam('constantKREMD', params.StringParam, label='K values ',
                      help="TODO", condition="replica_exchange")

        form.addParallelSection(threads=1, mpi=1)
        # --------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
        self._insertFunctionStep("convertInputPDBStep")
        if self.EMfitChoice.get() == EMFIT_VOLUMES or self.EMfitChoice.get() == EMFIT_IMAGES:
            self._insertFunctionStep("convertInputEMStep")
        self._insertFunctionStep("fittingStep")
        self._insertFunctionStep("createOutputStep")

    ################################################################################
    ##                 CONVERT INPUT PDB
    ################################################################################

    def convertInputPDBStep(self):

        inputPDBfn = self.getInputPDBfn()
        n_pdb = self.getNumberOfInputPDB()

        # Copy PDBs :
        for i in range(n_pdb):
            os.system("cp %s %s.pdb"%(inputPDBfn[i],self.getInputPDBprefix(i)))

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
                    os.system("cp %s %s.psf" % (self.inputPSF.get(), self.getInputPDBprefix(i)))

            # GROMACS
            elif self.forcefield.get() == FORCEFIELD_AAGO\
                    or self.forcefield.get() == FORCEFIELD_CAGO:
                os.system("cp %s %s.top" % (self.inputTOP.get(), self.getInputPDBprefix(i)))

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
                                   volPrefix = self.getInputEMprefix(i), fnPDB=self.getInputPDBprefix(i))

        # Initialize rigid body fitting parameters
        elif self.EMfitChoice.get() == EMFIT_IMAGES:
            for i in range(n_em):
                os.system("cp %s %s.spi"%(inputEMfn[i], self.getInputEMprefix(i)))
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
        runProgram("%s/bin/emmap_generator" % self.genesisDir.get(), "%s_INP_emmap" % fnTmpVol)

        # CONVERT SITUS TMP FILE TO MRC
        with open(self._getExtraPath("runconvert.sh"), "w") as f:
            f.write("#!/bin/bash \n")
            f.write("%s/bin/map2map %s %s <<< \'1\'\n" % (self.situs_dir.get(), fnTmpVol+".sit", fnTmpVol+".mrc"))
            f.write("exit")
        os.system("/bin/bash "+self._getExtraPath("runconvert.sh"))

        # READ GENERATED MRC
        with mrcfile.open(fnTmpVol+".mrc") as tmp_mrc:
            tmpMRCData = tmp_mrc.data

        # MATCH HISTOGRAMS
        mrc_data = match_histograms(inputMRCData, tmpMRCData)

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
            f.write("%s/bin/map2map %s %s <<< \'1\'\n" % (self.situs_dir.get(),
                                                          "%sConv.mrc"%volPrefix, "%s.sit"%volPrefix))
            f.write("exit")
        os.system("/bin/bash " + self._getExtraPath("runconvert.sh"))

        # CLEANING
        runProgram("rm","-f %s.sit"%fnTmpVol)
        runProgram("rm","-f %s.mrc"%fnTmpVol)
        runProgram("rm","-f %s"%self._getExtraPath("runconvert.sh"))
        runProgram("rm","-f %s_INP_emmap" % fnTmpVol)
        runProgram("rm","-f %sConv.mrc"%volPrefix)
        runProgram("rm","-f %s.mrc" % volPrefix)


    ################################################################################
    ##                 FITTING STEP
    ################################################################################

    def fittingStep(self):
        # SETUP MPI parameters
        numMpiPerFit, numLinearFit, numParallelFit, numLastIter = self.getMPIParams()

        # RUN PARALLEL FITTING
        if not(self.EMfitChoice.get() == EMFIT_IMAGES and self.estimateAngleShift.get()):
            for i1 in range(numLinearFit+1):
                cmds= []
                n_parallel = numParallelFit if i1<numLinearFit else numLastIter
                for i2 in range(n_parallel):
                    indexFit = i2 + i1*numParallelFit
                    prefix = self._getExtraPath(str(indexFit + 1).zfill(5))

                    # Create INP file
                    self.createINP(prefix=prefix, indexFit=indexFit)

                    # Create Genesis command
                    cmds.append(self.getGenesisCmd(prefix=prefix, n_mpi=numMpiPerFit))

                # Run Genesis
                self.runParallelJobs(cmds)


        # RUN PARALLEL FITTING FOR IMAGES
        else:
            for i1 in range(numLinearFit + 1):
                n_parallel = numParallelFit if i1 < numLinearFit else numLastIter

                # Loop rigidbody align / GENESIS fitting
                for iterFit in range(self.n_iter.get()):

                # ------   ALIGN PDBs---------
                    # Transform PDBs to volume
                    cmds_pdb2vol = []
                    for i2 in range(n_parallel):
                        indexFit = i2 + i1 * numParallelFit
                        inputPDB = self.getInputPDBprefix(indexFit)+".pdb" if iterFit ==0 \
                            else  self._getExtraPath("%s_iter%i_output.pdb" % (str(indexFit + 1).zfill(5), iterFit-1))

                        tmpPrefix = self._getExtraPath("%s_tmp" % str(indexFit + 1).zfill(5))
                        cmds_pdb2vol.append(self.pdb2vol(inputPDB=inputPDB, outputVol=tmpPrefix))
                    self.runParallelJobs(cmds_pdb2vol)

                    # Loop 4 times to refine the angles
                    sampling_rate = [10.0, 5.0, 3.0, 2.0]
                    angular_distance = [-1, 20, 10, 5]
                    for i_align in range(4):
                        cmds_projectVol = []
                        cmds_projectMatch = []
                        for i2 in range(n_parallel):
                            indexFit = i2 + i1 * numParallelFit
                            tmpPrefix = self._getExtraPath("%s_tmp" % str(indexFit + 1).zfill(5))
                            inputImage = self.getInputEMprefix(indexFit)+".spi"
                            currentAngles = self._getExtraPath("%s_current_angles.xmd" % str(indexFit + 1).zfill(5))

                            # get commands
                            cmds_projectVol.append(self.projectVol(inputVol=tmpPrefix, outputProj=tmpPrefix, expImage=currentAngles,
                                                sampling_rate=sampling_rate[i_align], angular_distance=angular_distance[i_align]))
                            cmds_projectMatch.append(self.projectMatch(inputImage= inputImage, inputProj=tmpPrefix, outputMeta=currentAngles))

                        # run parallel jobs
                        self.runParallelJobs(cmds_projectVol)
                        self.runParallelJobs(cmds_projectMatch)

                    # Cleaning volumes and projections
                    for i2 in range(n_parallel):
                        indexFit = i2 + i1 * numParallelFit
                        tmpPrefix = self._getExtraPath("%s_tmp" % str(indexFit + 1).zfill(5))
                        os.system("rm -f %s*"%tmpPrefix)


                # ------   Run Genesis ---------
                    cmds = []
                    for i2 in range(n_parallel):
                        indexFit = i2 + i1 * numParallelFit
                        prefix = self._getExtraPath("%s_iter%i" % (str(indexFit + 1).zfill(5), iterFit))

                        # Create INP file
                        self.createINP(prefix=prefix, indexFit=indexFit)

                        # run GENESIS
                        cmds.append(self.getGenesisCmd(prefix=prefix, n_mpi=numMpiPerFit))
                    self.runParallelJobs(cmds)

                for i2 in range(n_parallel):
                    idx = str(i2 + i1 * numParallelFit+ 1).zfill(5)
                    os.system("cp %s %s" % (self._getExtraPath("%s_iter%i_output.pdb" % (idx, self.n_iter.get()-1)),
                                            self._getExtraPath("%s_output.pdb" % idx)))




    def runParallelJobs(self, cmds):

        # Set env
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(self.numberOfThreads)

        # run process
        processes = []
        for cmd in cmds:
            print("Running command : %s" %cmd)
            processes.append(Popen(cmd, shell=True, env=env))

        # Wait for processes
        for p in processes:
            exitcode = p.wait()
            print("Process done %s" %str(exitcode))
            if exitcode != 0:
                raise RuntimeError("GENESIS exit with errors, check .log file ")
                # print("Warning : GENESIS exit with errors, check .log file ")


    def getGenesisCmd(self, prefix,n_mpi):
        cmd=""
        if (n_mpi != 1):
            cmd += "mpirun -np %s " % n_mpi
        cmd +=  "%s/bin/atdyn %s " % (self.genesisDir.get(),"%s_INP" % prefix)
        if self.normalModesChoice.get():
            cmd += "%s/ %i %f %f" % (self.genesisDir.get(), self.n_modes.get(),
                                      self.global_mass.get(), self.global_limit.get())
        cmd += " > %s_output.log" % prefix
        return cmd

    def createINP(self,prefix, indexFit):
        # CREATE INPUT FILE FOR GENESIS
        outputPrefix = "%s_output"%prefix
        inputPDBprefix = self.getInputPDBprefix(indexFit)
        inputEMprefix = self.getInputEMprefix(indexFit)

        s = "\n[INPUT] \n" #-----------------------------------------------------------
        s += "pdbfile = %s.pdb\n" % inputPDBprefix
        if self.forcefield.get() == FORCEFIELD_CHARMM:
            s += "topfile = %s\n" % self.inputRTF.get()
            s += "parfile = %s\n" % self.inputPRM.get()
            s += "psffile = %s.psf\n" % inputPDBprefix
        elif self.forcefield.get() == FORCEFIELD_AAGO\
                or self.forcefield.get() == FORCEFIELD_CAGO:
            s += "grotopfile = %s.top\n" % inputPDBprefix
        if self.restartchoice.get():
            s += "rstfile = %s\n" % self.inputRST.get()

        s += "\n[OUTPUT] \n" #-----------------------------------------------------------
        if self.replica_exchange.get():
            outputPrefix += "_remd{}"
            s += "remfile = %s.rem\n" %outputPrefix
            s += "logfile = %s.log\n" %outputPrefix
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
        s += "electrostatic = CUTOFF  \n"
        s += "switchdist   = %.2f \n" % self.switch_dist.get()
        s += "cutoffdist   = %.2f \n" % self.cutoff_dist.get()
        s += "pairlistdist = %.2f \n" % self.pairlist_dist.get()
        s += "vdw_force_switch = YES \n"
        if self.implicitSolvent.get() == IMPLICIT_SOLVENT_GBSA:
            s += "implicit_solvent = GBSA \n"
            s += "gbsa_eps_solvent = 78.5 \n"
            s += "gbsa_eps_solute  = 1.0 \n"
            s += "gbsa_salt_cons   = 0.2 \n"
            s += "gbsa_surf_tens   = 0.005 \n"
        else:
            s += "implicit_solvent = NONE  \n"

        if self.simulationType.get() == SIMULATION_MIN:
            s += "\n[MINIMIZE]\n" #-----------------------------------------------------------
            s += "method = SD\n"
        else:
            s += "\n[DYNAMICS] \n" #-----------------------------------------------------------
            if self.integrator.get() == INTEGRATOR_VVERLET:
                s += "integrator = VVER  \n"
            else:
                s += "integrator = LEAP  \n"
            s += "timestep = %f \n" % self.time_step.get()
        s += "nsteps = %i \n" % self.n_steps.get()
        s += "eneout_period = %i \n" % self.eneout_period.get()
        s += "crdout_period = %i \n" % self.crdout_period.get()
        s += "rstout_period = %i \n" % self.n_steps.get()
        s += "nbupdate_period = %i \n" % self.nbupdate_period.get()

        s += "\n[CONSTRAINTS] \n" #-----------------------------------------------------------
        s += "rigid_bond = NO \n"

        s += "\n[ENSEMBLE] \n" #-----------------------------------------------------------
        s += "ensemble = NVT \n"
        if self.tpcontrol.get() == TPCONTROL_LANGEVIN:
            s += "tpcontrol = LANGEVIN  \n"
        elif self.tpcontrol.get() == TPCONTROL_BERENDSEN:
            s += "tpcontrol = BERENDSEN  \n"
        else:
            s += "tpcontrol = NO  \n"
        s += "temperature = %.2f \n" % self.temperature.get()

        s += "\n[BOUNDARY] \n" #-----------------------------------------------------------
        s += "type = NOBC  \n"

        if (self.EMfitChoice.get()==EMFIT_VOLUMES or self.EMfitChoice.get()==EMFIT_IMAGES)\
                and self.simulationType.get() == SIMULATION_MD:
            s += "\n[SELECTION] \n" #-----------------------------------------------------------
            s += "group1 = all and not hydrogen\n"

            s += "\n[RESTRAINTS] \n" #-----------------------------------------------------------
            s += "nfunctions = 1 \n"
            s += "function1 = EM \n"
            if self.replica_exchange.get():
                s += "constant1 = %s \n" % self.constantKREMD.get()
            else:
                s += "constant1 = %.2f \n" % self.constantK.get()
            s += "select_index1 = 1 \n"

            s += "\n[EXPERIMENTS] \n" #-----------------------------------------------------------
            s += "emfit = YES  \n"
            s += "emfit_sigma = %.4f \n" % self.emfit_sigma.get()
            s += "emfit_tolerance = %.6f \n" % self.emfit_tolerance.get()
            s += "emfit_period = 1  \n"
            if self.EMfitChoice.get() == EMFIT_VOLUMES:
                s += "emfit_target = %s.sit \n" % inputEMprefix
            elif self.EMfitChoice.get()==EMFIT_IMAGES :
                s += "emfit_exp_image = %s.spi \n" % inputEMprefix
                s += "emfit_image_size =  %i\n" %self.image_size.get()
                s += "emfit_pixel_size =  %i\n" % self.pixel_size.get()
                rigid_body_params = self.getRigidBodyParams(indexFit)
                s += "emfit_roll_angle = %f\n" % rigid_body_params[0]
                s += "emfit_tilt_angle = %f\n" % rigid_body_params[1]
                s += "emfit_yaw_angle =  %f\n" % rigid_body_params[2]
                s += "emfit_shift_x = %f\n" % rigid_body_params[3]
                s += "emfit_shift_y =  %f\n" % rigid_body_params[4]

            if self.replica_exchange.get():
                s += "\n[REMD] \n" #-----------------------------------------------------------
                s += "dimension = 1 \n"
                s += "exchange_period = %i \n" % self.exchange_period.get()
                s += "type1 = RESTRAINT \n"
                s += "nreplica1 = %i \n" % self.nreplica.get()
                s += "rest_function1 = 1 \n"

        with open("%s_INP"% prefix, "w") as f:
            f.write(s)


    def pdb2vol(self, inputPDB, outputVol):
        cmd = "xmipp_volume_from_pdb"
        args = "-i %s  -o %s --sampling %f --size %i %i %i"%\
               (inputPDB, outputVol,self.pixel_size.get(),
                self.image_size.get(),self.image_size.get(),self.image_size.get())
        return cmd+ " "+ args

    def projectVol(self, inputVol, outputProj, expImage, sampling_rate=5.0, angular_distance=-1):
        cmd = "xmipp_angular_project_library"
        args = "-i %s.vol -o %s.stk --sampling_rate %f " % (inputVol, outputProj, sampling_rate)
        args +="--sym c1 --compute_neighbors --angular_distance %f --method real_space " % angular_distance
        args += "--experimental_images %s "%expImage
        if angular_distance != -1 :
            args += "--near_exp_data"
        return cmd+ " "+ args

    def projectMatch(self, inputImage, inputProj, outputMeta, max_shift = 1000.0):
        cmd = "xmipp_angular_projection_matching "
        args= "-i %s -o %s --ref %s.stk "%(inputImage, outputMeta, inputProj)
        args +="--max_shift %f --search5d_shift 10.0 --search5d_step 2.0" %max_shift
        return cmd + " "+ args



    # def applyTransform2PDB(self, inputPDB, outputPDB, inputMeta, tmpPrefix):
    #
    #     mdImgs = md.MetaData("%s.xmd"%inputMeta)
    #     Ts = self.voxel_size.get()
    #     for objId in mdImgs:
    #         rot = str(mdImgs.getValue(md.MDL_ANGLE_ROT, objId))
    #         tilt = str(mdImgs.getValue(md.MDL_ANGLE_TILT, objId))
    #         psi = str(mdImgs.getValue(md.MDL_ANGLE_PSI, objId))
    #
    #         shiftx = str(-mdImgs.getValue(md.MDL_SHIFT_X, objId)*Ts)
    #         shifty = str(-mdImgs.getValue(md.MDL_SHIFT_Y, objId)*Ts)
    #
    #         cmd = "xmipp_phantom_transform "
    #         args = "-i %s -o %s.pdb --operation rotate_euler %s %s %s" % \
    #                (inputPDB, tmpPrefix, rot, tilt,psi)
    #         runProgram(cmd, args)
    #
    #         cmd = "xmipp_phantom_transform "
    #         args = "-i %s.pdb -o %s --operation shift %s %s 0.0" % \
    #                (tmpPrefix, outputPDB, shiftx, shifty)
    #         runProgram(cmd, args)

    ################################################################################
    ##                 CREATE OUTPUT STEP
    ################################################################################

    def createOutputStep(self):
        # CREATE SET OF PDBs
        pdbset = self._createSetOfPDBs("outputPDBs")
        numberOfReplicas = self.nreplica.get() \
            if self.replica_exchange.get() else 1

        for i in range(self.getNumberOfFitting()):
            outputPrefix = self._getExtraPath("%s_output" % str(i + 1).zfill(5))
            for j in range(numberOfReplicas):
                if self.replica_exchange.get():
                    outputPrefix = self._getExtraPath("%s_output_remd%i" % (str(i + 1).zfill(5), j + 1))
                pdbset.append(AtomStruct(outputPrefix + ".pdb"))

        self._defineOutputs(outputPDBs=pdbset)

    # --------------------------- STEPS functions --------------------------------------------
    # --------------------------- INFO functions --------------------------------------------
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

    # --------------------------- UTILS functions --------------------------------------------

    def getNumberOfInputPDB(self):
        if isinstance(self.inputPDB.get(), SetOfAtomStructs) or \
                isinstance(self.inputPDB.get(), SetOfPDBs):
            return self.inputPDB.get().getSize()
        else: return 1

    def getNumberOfInputEM(self):
        if self.EMfitChoice.get() == EMFIT_VOLUMES:
            if isinstance(self.inputVolume.get(), SetOfVolumes): return self.inputVolume.get().getSize()
            else: return 1
        elif self.EMfitChoice.get() == EMFIT_IMAGES:
            if isinstance(self.inputImage.get(), SetOfParticles): return self.inputImage.get().getSize()
            else: return 1
        else: return 0

    def getNumberOfFitting(self):
        numberOfInputPDB = self.getNumberOfInputPDB()
        numberOfInputEM = self.getNumberOfInputEM()

        # Check input volumes/images correspond to input PDBs
        if numberOfInputPDB != numberOfInputEM and \
                numberOfInputEM != 1 and numberOfInputPDB != 1:
            raise RuntimeError("Number of input volumes and PDBs must be the same.")
        return np.max([numberOfInputEM, numberOfInputPDB])

    def getInputPDBfn(self):
        initFn = []
        if isinstance(self.inputPDB.get(), SetOfAtomStructs) or \
                isinstance(self.inputPDB.get(), SetOfPDBs):
            for i in range(self.inputPDB.get().getSize()):
                initFn.append(self.inputPDB.get()[i+1].getFileName())

        else:
            initFn.append(self.inputPDB.get().getFileName())
        return initFn

    def getInputEMfn(self):
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

    def getInputPDBprefix(self, index):
        prefix = self._getExtraPath("%s_inputPDB")
        if self.getNumberOfInputPDB() == 1:
            return prefix % str(1).zfill(5)
        else:
            return prefix % str(index + 1).zfill(5)

    def getInputEMprefix(self, index):
        prefix = self._getExtraPath("%s_inputEM")
        if self.getNumberOfInputEM() == 0:
            return ""
        elif self.getNumberOfInputEM() == 1:
            return prefix % str(1).zfill(5)
        else:
            return prefix % str(index + 1).zfill(5)

    def getMPIParams(self):
        """
        return numberOfMpiPerFit, numberOfLinearFit, numberOfParallelFit, numberOflastIter
        """
        n_fit = self.getNumberOfFitting()
        if n_fit <= self.numberOfMpi.get():
            return self.numberOfMpi.get()//n_fit, 1, n_fit, 0
        else:
            return 1, n_fit//self.numberOfMpi.get(),  self.numberOfMpi.get(), n_fit % self.numberOfMpi.get()

    def getRigidBodyParams(self, index):
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


