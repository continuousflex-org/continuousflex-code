# **************************************************************************
# *
# * Authors:
# * J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
# * Slavica Jonic (slavica.jonic@upmc.fr)
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


from pyworkflow.protocol.params import PointerParam, FileParam
from pwem.protocols import BatchProtocol
from pwem.objects import SetOfParticles, Volume

from xmipp3.convert import writeSetOfParticles
from pwem.utils import runProgram
import pwem.emlib.metadata as md


class FlexBatchProtNMACluster(BatchProtocol):
    """ Protocol executed when a cluster is created
    from NMA images and theirs deformations.
    """
    _label = 'nma cluster'
    
    def _defineParams(self, form):
        form.addHidden('inputNmaDimred', PointerParam, pointerClass='EMObject')
        form.addHidden('sqliteFile', FileParam)
        
    #--------------------------- INSERT steps functions --------------------------------------------
        
    def _insertAllSteps(self):
        imagesMd = self._getExtraPath('images.xmd')
        outputVol = self._getExtraPath('reconstruction.vol')
        
        self._insertFunctionStep('convertInputStep', imagesMd)
        params = '-i %(imagesMd)s -o %(outputVol)s --fast' % locals()
        self._insertFunctionStep('reconstructStep', params)
        self._insertFunctionStep('createOutputStep', outputVol)
        
    #--------------------------- STEPS functions --------------------------------------------   
        
    def convertInputStep(self, imagesMd):
        # It is unusual to create the output in the convertInputStep,
        # but just to avoid reading twice the sqlite with particles
        inputSet = self.inputNmaDimred.get().getInputParticles()
        partSet = self._createSetOfParticles()
        partSet.copyInfo(inputSet)
        
        tmpSet = SetOfParticles(filename=self.sqliteFile.get())        
        partSet.appendFromImages(tmpSet)
        # Register outputs
        partSet.setAlignmentProj()
        self._defineOutputs(outputParticles=partSet)
        self._defineTransformRelation(inputSet, partSet)
        
        writeSetOfParticles(partSet, imagesMd)

        # Add the NMA displacement to clusters XMD files
        md_file_nma = md.MetaData(self.inputNmaDimred.get().getParticlesMD())
        md_file_org = md.MetaData(imagesMd)
        for objID in md_file_org:
            # if image name is the same, we add the nma displacement from nma to org
            id_org = md_file_org.getValue(md.MDL_ITEM_ID, objID)
            for j in md_file_nma:
                id_nma = md_file_nma.getValue(md.MDL_ITEM_ID, j)
                print(id_nma)
                if id_org == id_nma:
                    displacements = md_file_nma.getValue(md.MDL_NMA, j)
                    md_file_org.setValue(md.MDL_NMA, displacements, objID)
                    break
        md_file_org.write(imagesMd)


    def reconstructStep(self, params):
        runProgram('xmipp_reconstruct_fourier_accel', params)
    
    def createOutputStep(self, outputVol):
        vol = Volume()
        vol.setFileName(outputVol)
        vol.setSamplingRate(self.outputParticles.getSamplingRate())
        self._defineOutputs(outputVol=vol)
        
    #--------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        return summary
    
    def _validate(self):
        errors = []
        return errors
    
    def _citations(self):
        return []
    
    def _methods(self):
        return []
    
