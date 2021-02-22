# **************************************************************************
# *
# * Authors:     P. Conesa (pconesa@cnb.csic.es) [1]
# *              J.M. De la Rosa Trevin (delarosatrevin@scilifelab.se) [2]
# *
# * [1] Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# * [2] SciLifeLab, Stockholm University
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
# *
# **************************************************************************
from pwem.protocols import ProtImportPdb, ProtImportParticles, ProtImportVolumes
from pwem.tests.workflows import TestWorkflow
from pwem import Domain
from pyworkflow.tests import setupTestProject, DataSet

from continuousflex.protocols import (FlexProtNMA, FlexProtSynthesizeSubtomo,NMA_CUTOFF_ABS)
from continuousflex.protocols.protocol_subtomogrmas_synthesize import MODE_RELATION_LINEAR, MODE_RELATION_3CLUSTERS,\
    MODE_RELATION_MESH, MODE_RELATION_RANDOM

class TestSubtomogramSynthesize(TestWorkflow):
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
        self.launchProtocol(protImportPdb)
        
        # Launch NMA for PDB imported
        protNMA = self.newProtocol(FlexProtNMA,
                                    cutoffMode=NMA_CUTOFF_ABS)
        protNMA.inputStructure.set(protImportPdb.outputPdb)
        self.launchProtocol(protNMA)
        
        # Synthesize subtomograms with linear relationship
        protSynthesize1 = self.newProtocol(FlexProtSynthesizeSubtomo,
                                         modeList='7-8',
                                         modeRelationChoice=MODE_RELATION_LINEAR)
        protSynthesize1.inputModes.set(protNMA.outputModes)
        self.launchProtocol(protSynthesize1)


        # Synthesize subtomograms with clusters relationship
        protSynthesize2 = self.newProtocol(FlexProtSynthesizeSubtomo,
                                         modeList='7-8',
                                         modeRelationChoice=MODE_RELATION_3CLUSTERS)
        protSynthesize2.inputModes.set(protNMA.outputModes)
        self.launchProtocol(protSynthesize2)

        # Synthesize subtomograms with Mesh relationship
        protSynthesize3 = self.newProtocol(FlexProtSynthesizeSubtomo,
                                         modeList='7-8',
                                         modeRelationChoice=MODE_RELATION_MESH)
        protSynthesize3.inputModes.set(protNMA.outputModes)
        self.launchProtocol(protSynthesize3)

        # Synthesize subtomograms with random relationship
        protSynthesize4 = self.newProtocol(FlexProtSynthesizeSubtomo,
                                         modeList='7-8',
                                         modeRelationChoice=MODE_RELATION_RANDOM)
        protSynthesize4.inputModes.set(protNMA.outputModes)
        self.launchProtocol(protSynthesize4)