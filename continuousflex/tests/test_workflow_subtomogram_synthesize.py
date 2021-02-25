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
from continuousflex.protocols.protocol_pdb_dimred import FlexProtDimredPdb
from continuousflex.protocols.protocol_subtomograms_classify import FlexProtSubtomoClassify

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
        protSynthesize1 = self.newProtocol(FlexProtSynthesizeSubtomo,
                                         modeList='7-8',
                                         modeRelationChoice=MODE_RELATION_LINEAR)
        protSynthesize1.inputModes.set(protNMA.outputModes)
        protSynthesize1.setObjLabel('synthesized linear')
        self.launchProtocol(protSynthesize1)

        protpdbdimred1 = self.newProtocol(FlexProtDimredPdb,
                                         reducedDim=3)
        protpdbdimred1.pdbs.set(protSynthesize1)
        protpdbdimred1.setObjLabel('pdb dimred')
        self.launchProtocol(protpdbdimred1)

        protclassifyhierarchical1= self.newProtocol(FlexProtSubtomoClassify,
                                                   numOfClasses=3)
        protclassifyhierarchical1.ProtSynthesize.set(protSynthesize1)
        protclassifyhierarchical1.setObjLabel('hierarchical')
        self.launchProtocol(protclassifyhierarchical1)
        protclassifyKmeans1 = self.newProtocol(FlexProtSubtomoClassify,
                                                     numOfClasses=3,
                                                     classifyTechnique=1,
                                                     reducedDim=3)
        protclassifyKmeans1.ProtSynthesize.set(protSynthesize1)
        protclassifyKmeans1.setObjLabel('Kmeans')
        self.launchProtocol(protclassifyKmeans1)
        # ------------------------------------------------------------------------------------
        # Synthesize subtomograms with clusters relationship
        protSynthesize2 = self.newProtocol(FlexProtSynthesizeSubtomo,
                                         modeList='7-8',
                                         modeRelationChoice=MODE_RELATION_3CLUSTERS)
        protSynthesize2.inputModes.set(protNMA.outputModes)
        protSynthesize2.setObjLabel('synthesized 3 clusters')
        self.launchProtocol(protSynthesize2)

        protpdbdimred2 = self.newProtocol(FlexProtDimredPdb,
                                         reducedDim=3)
        protpdbdimred2.pdbs.set(protSynthesize2)
        protpdbdimred2.setObjLabel('pdb dimred')
        self.launchProtocol(protpdbdimred2)

        protclassifyhierarchical2= self.newProtocol(FlexProtSubtomoClassify,
                                                   numOfClasses=3)
        protclassifyhierarchical2.ProtSynthesize.set(protSynthesize2)
        protclassifyhierarchical2.setObjLabel('hierarchical')
        self.launchProtocol(protclassifyhierarchical2)
        protclassifyKmeans2 = self.newProtocol(FlexProtSubtomoClassify,
                                                     numOfClasses=3,
                                                     classifyTechnique=1,
                                                     reducedDim=3)
        protclassifyKmeans2.ProtSynthesize.set(protSynthesize2)
        protclassifyKmeans2.setObjLabel('Kmeans')
        self.launchProtocol(protclassifyKmeans2)
        # ------------------------------------------------------------------------------------
        # Synthesize subtomograms with Mesh relationship
        protSynthesize3 = self.newProtocol(FlexProtSynthesizeSubtomo,
                                         modeList='7-8',
                                         modeRelationChoice=MODE_RELATION_MESH)
        protSynthesize3.inputModes.set(protNMA.outputModes)
        protSynthesize3.setObjLabel('synthesized mesh')
        self.launchProtocol(protSynthesize3)

        protpdbdimred3 = self.newProtocol(FlexProtDimredPdb,
                                         reducedDim=3)
        protpdbdimred3.pdbs.set(protSynthesize3)
        protpdbdimred3.setObjLabel('pdb dimred')
        self.launchProtocol(protpdbdimred3)

        protclassifyhierarchical3= self.newProtocol(FlexProtSubtomoClassify,
                                                   numOfClasses=3)
        protclassifyhierarchical3.ProtSynthesize.set(protSynthesize3)
        protclassifyhierarchical3.setObjLabel('hierarchical')
        self.launchProtocol(protclassifyhierarchical3)
        protclassifyKmeans3 = self.newProtocol(FlexProtSubtomoClassify,
                                                     numOfClasses=3,
                                                     classifyTechnique=1,
                                                     reducedDim=3)
        protclassifyKmeans3.ProtSynthesize.set(protSynthesize3)
        protclassifyKmeans3.setObjLabel('Kmeans')
        self.launchProtocol(protclassifyKmeans3)
        # ------------------------------------------------------------------------------------
        # Synthesize subtomograms with random relationship
        protSynthesize4 = self.newProtocol(FlexProtSynthesizeSubtomo,
                                         modeList='7-8',
                                         modeRelationChoice=MODE_RELATION_RANDOM)
        protSynthesize4.inputModes.set(protNMA.outputModes)
        protSynthesize4.setObjLabel('synthesized random')
        self.launchProtocol(protSynthesize4)

        protpdbdimred4 = self.newProtocol(FlexProtDimredPdb,
                                         reducedDim=3)
        protpdbdimred4.pdbs.set(protSynthesize4)
        protpdbdimred4.setObjLabel('pdb dimred')
        self.launchProtocol(protpdbdimred4)

        protclassifyhierarchical4= self.newProtocol(FlexProtSubtomoClassify,
                                                   numOfClasses=3)
        protclassifyhierarchical4.ProtSynthesize.set(protSynthesize4)
        protclassifyhierarchical4.setObjLabel('hierarchical')
        self.launchProtocol(protclassifyhierarchical4)
        protclassifyKmeans4 = self.newProtocol(FlexProtSubtomoClassify,
                                                     numOfClasses=3,
                                                     classifyTechnique=1,
                                                     reducedDim=3)
        protclassifyKmeans4.ProtSynthesize.set(protSynthesize4)
        protclassifyKmeans4.setObjLabel('Kmeans')
        self.launchProtocol(protclassifyKmeans4)

        