# **************************************************************************
# * Author:  Mohamad Harastani          (mohamad.harastani@upmc.fr)
# * IMPMC, UPMC, Sorbonne University
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
from pwem.protocols import ProtImportPdb, ProtImportParticles, ProtImportVolumes
from pwem.tests.workflows import TestWorkflow
from pwem import Domain
from pyworkflow.tests import setupTestProject, DataSet

from continuousflex.protocols import (FlexProtNMA, FlexProtSynthesizeSubtomo,NMA_CUTOFF_ABS)
from continuousflex.protocols.protocol_subtomogrmas_synthesize import MODE_RELATION_LINEAR, MODE_RELATION_3CLUSTERS,\
    MODE_RELATION_MESH, MODE_RELATION_RANDOM
from continuousflex.protocols.protocol_pdb_dimred import FlexProtDimredPdb
from continuousflex.protocols.protocol_subtomograms_classify import FlexProtSubtomoClassify
from continuousflex.protocols.protocol_subtomogram_averaging import FlexProtSubtomogramAveraging

class TestStA(TestWorkflow):
    """ Check subtomograms are generated propoerly """
    
    @classmethod
    def setUpClass(cls):    
        # Create a new project
        setupTestProject(cls)
        cls.ds = DataSet.getDataSet('nma')
    
    def test_synthesize_all(self):
        """ Run NMA then synthesize sybtomograms"""
        
        #------------------------------------------------
        # Import a Pdb -> NMA
        #------------------------------------------------
        # Import a PDB
        protImportPdb = self.newProtocol(ProtImportPdb, inputPdbData=1,
                                         pdbFile=self.ds.getFile('pdb'))
        protImportPdb.setObjLabel('AK.pdb')
        self.launchProtocol(protImportPdb)
        
        # Launch NMA for PDB imported
        protNMA = self.newProtocol(FlexProtNMA,
                                    cutoffMode=NMA_CUTOFF_ABS)
        protNMA.inputStructure.set(protImportPdb.outputPdb)
        protNMA.setObjLabel('NMA')
        self.launchProtocol(protNMA)
        #------------------------------------------------------------------------------------
        # Synthesize subtomograms with linear relationship
        protSynthesize = self.newProtocol(FlexProtSynthesizeSubtomo,
                                         modeList='7-8',
                                         numberOfVolumes=36,
                                         modeRelationChoice=MODE_RELATION_LINEAR)
        protSynthesize.inputModes.set(protNMA.outputModes)
        protSynthesize.setObjLabel('synthesized linear')
        self.launchProtocol(protSynthesize)
        # Perform StA
        protStA = self.newProtocol(FlexProtSubtomogramAveraging,
                                    NumOfIters=3)
        protStA.inputVolumes.set(protSynthesize.outputVolumes)
        protStA.setObjLabel('StA')
        self.launchProtocol(protStA)

