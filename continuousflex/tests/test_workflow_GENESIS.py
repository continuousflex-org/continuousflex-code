# **************************************************************************
# * Authors:     Rémi Vuillemot (remi.vuillemot@upmc.fr)
# * IMPMC, Sorbonne University
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

from pwem.protocols import ProtImportPdb #, ProtImportParticles, ProtImportVolumes
from pwem.tests.workflows import TestWorkflow
from pyworkflow.tests import setupTestProject, DataSet

from continuousflex.protocols.protocol_genesis import *
from continuousflex.viewers.viewer_genesis import *
from xmipp3.protocols import XmippProtConvertPdb
import os
import multiprocessing


class TestGENESIS_1(TestWorkflow):
    """ Test protocol for GENESIS. """
    @classmethod
    def setUpClass(cls):
        # Create a new project
        setupTestProject(cls)
        cls.ds = DataSet.getDataSet('nma_V2.0')

    def test_GENESIS_EMFIT_VOL_CHARMM(self):
        # Import a initial PDB
        protPdb4ake = self.newProtocol(ProtImportPdb, inputPdbData=1,
                                         pdbFile=self.ds.getFile('4ake'))
        protPdb4ake.setObjLabel('4ake.pdb')
        self.launchProtocol(protPdb4ake)

        # import target PDB
        protPdb1ake = self.newProtocol(ProtImportPdb, inputPdbData=1,
                                         pdbFile=self.ds.getFile('1ake'))
        protPdb1ake.setObjLabel('1ake.pdb')
        self.launchProtocol(protPdb1ake)

        protVolFromPdb = self.newProtocol(XmippProtConvertPdb, inputPdbData=1,
                                       pdbObj=protPdb1ake.outputPdb, setSize=True,
                                          size_x=64,size_y=64,size_z=64,
                                          sampling=2.0)
        self.launchProtocol(protVolFromPdb)

        protGenesisMin = self.newProtocol(ProtGenesis,
            inputPDB = protPdb4ake.outputPdb,
            forcefield = FORCEFIELD_CHARMM,
            generateTop = True,
            inputPRM = self.ds.getFile('charmm_prm'),
            inputRTF = self.ds.getFile('charmm_top'),

            simulationType = SIMULATION_MIN,
            time_step = 0.002,
            n_steps = 100,
            eneout_period = 10,
            crdout_period = 10,
            nbupdate_period = 10,

            implicitSolvent = IMPLICIT_SOLVENT_NONE,
            electrostatics = ELECTROSTATICS_CUTOFF,
            switch_dist = 10.0,
            cutoff_dist = 12.0,
            pairlist_dist = 15.0,

            numberOfThreads = multiprocessing.cpu_count(),

       )
        # Launch minimisation
        self.launchProtocol(protGenesisMin)

        # Get GENESIS log file
        output_prefix = protGenesisMin.getOutputPrefix()
        log_file = output_prefix+".log"

        # Get the potential energy from the log file
        potential_ene = readLogFile(log_file)["POTENTIAL_ENE"]

        # Assert that the potential energy is decreasing
        print("\n\n//////////////////////////////////////////////")
        print("Initial potential energy : %.2f kcal/mol"%potential_ene[0])
        print("Final potential energy : %.2f kcal/mol"%potential_ene[-1])
        print("//////////////////////////////////////////////\n\n")

        assert(potential_ene[0] > potential_ene[-1])

        protGenesisFit = self.newProtocol(ProtGenesis,

            inputPDB = protGenesisMin.outputPDBs,
            forcefield = FORCEFIELD_CHARMM,
            generateTop = False,
            inputPRM = self.ds.getFile('charmm_prm'),
            inputRTF = self.ds.getFile('charmm_top'),
            inputPSF = protGenesisMin.getInputPDBprefix()+".psf",
            restartchoice = True,
            inputRST = protGenesisMin.getOutputPrefix()+".rst",

            simulationType = SIMULATION_MD,
            integrator = INTEGRATOR_VVERLET,
            time_step = 0.002,
            n_steps = 5000,
            eneout_period = 100,
            crdout_period = 100,
            nbupdate_period = 10,

            implicitSolvent = IMPLICIT_SOLVENT_NONE,
            electrostatics = ELECTROSTATICS_CUTOFF,
            switch_dist = 10.0,
            cutoff_dist = 12.0,
            pairlist_dist = 15.0,

            ensemble = ENSEMBLE_NVT,
            tpcontrol = TPCONTROL_LANGEVIN,
            temperature = 300.0,

            boundary = BOUNDARY_NOBC,
            EMfitChoice = EMFIT_VOLUMES,
            constantK = 10000,
            emfit_sigma = 2.0,
            emfit_tolerance = 0.1,
            inputVolume = protVolFromPdb.outputVolume,
            voxel_size = 2.0,
            centerOrigin = True,
            preprocessingVol = PREPROCESS_VOL_MATCH,

            numberOfThreads = multiprocessing.cpu_count(),
        )

        # Launch minimisation
        self.launchProtocol(protGenesisFit)

        # Get GENESIS log file
        log_file = protGenesisFit.getOutputPrefix()+".log"

        # Get the CC from the log file
        cc = readLogFile(log_file)["RESTR_CVS001"]

        # Assert that the CC is increasing
        print("\n\n//////////////////////////////////////////////")
        print("Initial CC : %.2f"%cc[0])
        print("Final CC : %.2f"%cc[-1])
        print("//////////////////////////////////////////////\n\n")

        assert(cc[0] < cc[-1])

        # Get the RMSD from the dcd file
        rmsd = rmsdFromDCD(outputPrefix = protGenesisFit.getOutputPrefix(),
                           inputPDB = protGenesisFit.getInputPDBprefix()+".pdb",
                           targetPDB=protPdb1ake.outputPdb.getFileName(),
                           align=False)

        # Assert that the RMSD is decreasing
        print("\n\n//////////////////////////////////////////////")
        print("Initial rmsd : %.2f Ang"%rmsd[0])
        print("Final rmsd : %.2f Ang"%rmsd[-1])
        print("//////////////////////////////////////////////\n\n")

        assert(rmsd[0] > rmsd[-1])
        assert(rmsd[-1] < 3.0)

    def test_GENESIS_EMFIT_VOL_CAGO(self):
        # Import a initial PDB
        protPdb4ake = self.newProtocol(ProtImportPdb, inputPdbData=1,
                                         pdbFile=self.ds.getFile('4ake'))
        protPdb4ake.setObjLabel('4ake.pdb')
        self.launchProtocol(protPdb4ake)

        # import target PDB
        protPdb1ake = self.newProtocol(ProtImportPdb, inputPdbData=1,
                                         pdbFile=self.ds.getFile('1ake'))
        protPdb1ake.setObjLabel('1ake.pdb')
        self.launchProtocol(protPdb1ake)

        protVolFromPdb = self.newProtocol(XmippProtConvertPdb, inputPdbData=1,
                                       pdbObj=protPdb1ake.outputPdb, setSize=True,
                                          size_x=64,size_y=64,size_z=64,
                                          sampling=2.0)
        self.launchProtocol(protVolFromPdb)

        protGenesisMin = self.newProtocol(ProtGenesis,
            inputPDB = protPdb4ake.outputPdb,
            forcefield = FORCEFIELD_CAGO,
            generateTop = True,
            inputRTF = self.ds.getFile('charmm_top'),
            smog_dir = "/home/guest/Smog/",

            simulationType = SIMULATION_MIN,
            time_step = 0.001,
            n_steps = 100,
            eneout_period = 10,
            crdout_period = 10,
            nbupdate_period = 10,

            implicitSolvent = IMPLICIT_SOLVENT_NONE,
            electrostatics = ELECTROSTATICS_CUTOFF,
            switch_dist = 10.0,
            cutoff_dist = 12.0,
            pairlist_dist = 15.0,

            numberOfThreads = multiprocessing.cpu_count(),

       )
        # Launch minimisation
        self.launchProtocol(protGenesisMin)

        protGenesisFit = self.newProtocol(ProtGenesis,

            inputPDB = protGenesisMin.outputPDBs,
            forcefield = FORCEFIELD_CAGO,
            generateTop = False,
            inputTOP = protGenesisMin.getInputPDBprefix()+".top",
            restartchoice = True,
            inputRST = protGenesisMin.getOutputPrefix()+".rst",

            simulationType = SIMULATION_MD,
            integrator = INTEGRATOR_VVERLET,
            time_step = 0.0005,
            n_steps = 10000,
            eneout_period = 100,
            crdout_period = 100,
            nbupdate_period = 10,

            implicitSolvent = IMPLICIT_SOLVENT_NONE,
            electrostatics = ELECTROSTATICS_CUTOFF,
            switch_dist = 10.0,
            cutoff_dist = 12.0,
            pairlist_dist = 15.0,

            ensemble = ENSEMBLE_NVT,
            tpcontrol = TPCONTROL_LANGEVIN,
            temperature = 100.0,

            boundary = BOUNDARY_NOBC,
            EMfitChoice = EMFIT_VOLUMES,
            constantK = 100,
            emfit_sigma = 2.0,
            emfit_tolerance = 0.1,
            inputVolume = protVolFromPdb.outputVolume,
            voxel_size = 2.0,
            centerOrigin = True,
            preprocessingVol = PREPROCESS_VOL_MATCH,

            numberOfThreads = multiprocessing.cpu_count(),
        )

        # Launch minimisation
        self.launchProtocol(protGenesisFit)

        # Get GENESIS log file
        log_file = protGenesisFit.getOutputPrefix()+".log"

        # Get the CC from the log file
        cc = readLogFile(log_file)["RESTR_CVS001"]

        # Assert that the CC is increasing
        print("\n\n//////////////////////////////////////////////")
        print("Initial CC : %.2f"%cc[0])
        print("Final CC : %.2f"%cc[-1])
        print("//////////////////////////////////////////////\n\n")

        assert(cc[0] < cc[-1])

        # Get the RMSD from the dcd file
        rmsd = rmsdFromDCD(outputPrefix = protGenesisFit.getOutputPrefix(),
                           inputPDB = protGenesisFit.getInputPDBprefix()+".pdb",
                           targetPDB=protPdb1ake.outputPdb.getFileName(),
                           align=False)

        # Assert that the RMSD is decreasing
        print("\n\n//////////////////////////////////////////////")
        print("Initial rmsd : %.2f Ang"%rmsd[0])
        print("Final rmsd : %.2f Ang"%rmsd[-1])
        print("//////////////////////////////////////////////\n\n")

        assert(rmsd[0] > rmsd[-1])
        assert(rmsd[-1] < 3.0)

    def test_GENESIS_CHARMM_MD(self):
        # Import PDB
        protPdbIonize = self.newProtocol(ProtImportPdb, inputPdbData=1,
                                       pdbFile=self.ds.getFile('ionize_pdb'))
        protPdbIonize.setObjLabel('ionize.pdb')
        self.launchProtocol(protPdbIonize)

        # Minimize energy
        protGenesisMin = self.newProtocol(ProtGenesis,
              inputPDB = protPdbIonize.outputPdb,
              forcefield = FORCEFIELD_CHARMM,
              inputPRM = self.ds.getFile('charmm_prm'),
              inputRTF = self.ds.getFile('charmm_top'),
              inputPSF = self.ds.getFile('ionize_psf'),
              inputSTR = self.ds.getFile('charmm_str'),

              md_program = PROGRAM_SPDYN,
              simulationType = SIMULATION_MIN,
              time_step = 0.002,
              n_steps = 1000,
              eneout_period = 10,
              crdout_period = 10,
              nbupdate_period = 10,

              electrostatics = ELECTROSTATICS_PME,
              switch_dist = 10.0,
              cutoff_dist = 12.0,
              pairlist_dist = 15.0,

              boundary = BOUNDARY_PBC,
              box_size_x = 101.4,
              box_size_y = 113.6,
              box_size_z = 81.4,

              rigid_bond = True,
              fast_water = True,
              water_model = "TIP3",

              numberOfThreads=multiprocessing.cpu_count(),
          )
        # Launch minimisation
        self.launchProtocol(protGenesisMin)

        # Get GENESIS log file
        output_prefix = protGenesisMin.getOutputPrefix()
        log_file = output_prefix + ".log"

        # Get the potential energy from the log file
        potential_ene = readLogFile(log_file)["POTENTIAL_ENE"]

        # Assert that the potential energy is decreasing
        print("\n\n//////////////////////////////////////////////")
        print("Initial potential energy : %.2f kcal/mol" % potential_ene[0])
        print("Final potential energy : %.2f kcal/mol" % potential_ene[-1])
        print("//////////////////////////////////////////////\n\n")

        assert (potential_ene[0] > potential_ene[-1])

        protGenesisMDRun = self.newProtocol(ProtGenesis,
                    inputPDB=protPdbIonize.outputPdb,
                    forcefield=FORCEFIELD_CHARMM,
                    inputPRM=self.ds.getFile('charmm_prm'),
                    inputRTF=self.ds.getFile('charmm_top'),
                    inputPSF=self.ds.getFile('ionize_psf'),
                    inputSTR=self.ds.getFile('charmm_str'),

                    md_program=PROGRAM_SPDYN,
                    integrator=INTEGRATOR_VVERLET,
                    time_step=0.002,
                    n_steps=100,
                    eneout_period=100,
                    crdout_period=100,
                    nbupdate_period=10,

                    electrostatics=ELECTROSTATICS_PME,
                    switch_dist=10.0,
                    cutoff_dist=12.0,
                    pairlist_dist=15.0,

                    ensemble=ENSEMBLE_NPT,
                    tpcontrol=TPCONTROL_LANGEVIN,
                    temperature=300.0,
                    pressure=1.0,

                    boundary=BOUNDARY_PBC,
                    box_size_x=101.4,
                    box_size_y=113.6,
                    box_size_z=81.4,

                    rigid_bond=True,
                    fast_water=True,
                    water_model="TIP3",

                    numberOfThreads=multiprocessing.cpu_count(),
                                          )

        # Launch minimisation
        self.launchProtocol(protGenesisMDRun)

        # Get GENESIS log file
        log_file = protGenesisMDRun.getOutputPrefix() + ".log"









        #     inputPDB = protImportPdb.outputPdb.get(),
        #     forcefield = FORCEFIELD_CHARMM,
        #     generateTop = True,
        #     # smog_dir = ,
        #     # inputTOP = ,
        #     inputPRM = "/home/guest/toppar/",
        #     inputRTF = "/home/guest/toppar/",
        #     nucleicChoice = NUCLEIC_NO,
        #     # inputPSF = ,
        #     restartchoice = False,
        #     # inputRST = ,
        #
        #     simulationType = SIMULATION_MIN,
        #     # integrator = ,
        #     time_step = 0.002,
        #     n_steps = 1000,
        #
        #     eneout_period = 100,
        #     crdout_period = 100,
        #     nbupdate_period = 10,
        #     # nm_number = ,
        #     # nm_mass = ,
        #     # nm_limit = ,
        #     # elnemo_cutoff = ,
        #     # elnemo_rtb_block = ,
        #     # elnemo_path = ,
        #
        #     implicitSolvent = IMPLICIT_SOLVENT_NONE,
        #     electrostatics = ELECTROSTATICS_CUTOFF,
        #     switch_dist = 10.0,
        #     cutoff_dist = 12.0,
        #     pairlist_dist = 15.0,
        #
        #     ensemble = ENSEMBLE_NVT,
        #     tpcontrol = TPCONTROL_LANGEVIN,
        #     temperature = 300.0,
        #     # pressure = ,
        #
        #     boundary = BOUNDARY_NOBC,
        #     # box_size_x = ,
        #     # box_size_y = ,
        #     # box_size_z = ,
        #
        #     EMfitChoice = EMFIT_NONE,
        #     # constantK = ,
        #     # emfit_sigma = ,
        #     # emfit_tolerance = ,
        #     # inputVolume = ,
        #     # voxel_size = ,
        #     # situs_dir = ,
        #     # centerOrigin = ,
        #     # preprocessingVol = ,
        #     # inputImage = ,
        #     # image_size = ,
        #     # estimateAngleShift = ,
        #     # rb_n_iter = ,
        #     # rb_method = ,
        #     # imageAngleShift = ,
        #     # pixel_size = ,
        #
        #     rigid_bond = False,
        #     fast_water = False,
        #     # water_model = ,
        #
        #     replica_exchange = False,
        #     # exchange_period = ,
        #     # nreplica = ,
        #     # constantKREMD =
        #     numberOfThreads = multiprocessing.cpu_count(),
