# **************************************************************************
# *
# * Authors:
# * Ilyes Hamitouche
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


from pyworkflow.object import String
from pyworkflow.protocol.params import (PointerParam, StringParam, EnumParam,
                                        IntParam, LEVEL_ADVANCED)
import pyworkflow.protocol.params as params
from pwem.protocols import ProtAnalysis3D
from pwem.utils import runProgram
import pwem.emlib.metadata as md
import numpy as np

OPTION_SHFITS = 0
OPTION_ANGLES = 1
OPTION_SHIFTS_ANGLES = 2
OPTION_NMA = 3
OPTION_ALL = 4

DEVICE_CUDA = 0
DEVICE_CPU = 1


class FlexProtDeepHEMNMATrain(ProtAnalysis3D):
    """ DeepHEMNMA protocol, a neural network that learns the rigid-body parameters and the normal mode
        amplitudes estimated by HEMNMA protocol.
    """
    _label = 'deep hemnma train'
    
    def __init__(self, **kwargs):
        ProtAnalysis3D.__init__(self, **kwargs)
        self.mappingFile = String()
    
    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('analyze_option', params.EnumParam, label='set the training mode',
                      display=params.EnumParam.DISPLAY_COMBO,
                      choices=['train on shifts',
                               'tain on angles',
                               'tain on shifts and angles',
                               'train on normal mode amplitudes'], default = OPTION_NMA,
                      help='TODO')
        group = form.addGroup('Train on conformational variability', condition='analyze_option == %d or analyze_option == %d'% (OPTION_NMA, OPTION_ALL))
        group.addParam('inputNMA', PointerParam, pointerClass='FlexProtAlignmentNMA',
                      label="Previous HEMNMA run",
                      help='Select a previous run of the NMA image alignment.', allowsNull=True)
        group = form.addGroup('Train on rigid-body variability ', condition='analyze_option == %d or analyze_option == %d or analyze_option == %d' %(OPTION_SHFITS, OPTION_ANGLES, OPTION_SHIFTS_ANGLES))
        group.addParam('inputParticles', PointerParam, pointerClass='SetOfParticles',
                      label="Preious run of rigid-body alignment",
                      help='Select a previous run of rigid-body alignment.', allowsNull=True)
        form.addParam('device_option', params.EnumParam, label='set the device for training',
                      display=params.EnumParam.DISPLAY_COMBO,
                      choices=['train on GPUs',
                               'tain on CPUs'], default = DEVICE_CUDA,
                      help='TODO')
        form.addParam('learning_rate', params.FloatParam, label = 'Learning rate', default = 0.0001)
        form.addParam('epochs', params.IntParam, expertLevel=params.LEVEL_ADVANCED,label = 'Number of epochs', default = 400)
        form.addParam('batch_size', params.IntParam ,expertLevel=params.LEVEL_ADVANCED, label = 'Batch size', default = 2)
        form.addParallelSection(threads=0, mpi=0)    
    
    
    #--------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
        print(self.analyze_option.get())
        print(self.inputParticles.get())
        print(self.device_option.get())
        print(self.learning_rate.get())
        # inputSet = self.getInputParticles()
        # rows = inputSet.getSize()
        # reducedDim = self.reducedDim.get()
        # method = self.dimredMethod.get()
        # extraParams = self.extraParams.get('')
        #
        # deformationsFile = self.getDeformationFile()
        #
        # self._insertFunctionStep('convertInputStep',
        #                          deformationsFile, inputSet.getObjId())
        # self._insertFunctionStep('performDimredStep',
        #                          deformationsFile, method, extraParams,
        #                          rows, reducedDim)
        # self._insertFunctionStep('createOutputStep')
        
        
    #--------------------------- STEPS functions --------------------------------------------
    def copy_parameters(self, md_file):

        self.imgsFn = self._getExtraPath('images.xmd')
        md = md.MetaData(self.imgsFn)
        rot = []
        tilt = []
        psi = []
        nma = []
        shift_x = []
        shift_y = []
        imgPath = []
        for objId in md:
            imgPath.append(self._getExtraPath('images.xmd')+mdImgs.getValue(md.MDL_IMAGE, objId))
            rot.append(mdImgs.getValue(md.MDL_ANGLE_ROT, objId))
            tilt.append(mdImgs.getValue(md.MDL_ANGLE_TILT, objId))
            psi.append(mdImgs.getValue(md.MDL_ANGLE_PSI, objId))
            shift_x.append(mdImgs.getValue(md.MDL_SHIFT_X, objId))
            shift_y.append(mdImgs.getValue(md.MDL_SHIFT_Y, objId))
            nma.append(mdImgs.getValue(md.MDL_NMA, objId))
        images_Path = np.array(img_Path)
        euler_angles = np.column_stack((rot, tilt, psi), dtype='float32')
        shifts = np.column_stack((shift_x, shift_y), dtype='float32')
        amplitudes = np.array(nma, dtype='float32')
        return images_path, euler_angles, shifts, amplitudes

    def performDeepHEMNMAStep(self, params):
        import continuousflex
        script_path = continuousflex.__path__[0] + '/protocols/utilities/deep_hemnma.py'
        command = "python " + script_path + str(params)
        check_call(command, shell=True, stdout=sys.stdout, stderr=sys.stderr,
                   env=None, cwd=None)

    def create_single_particle_path(self):
        self.writeModesMetaData()
        # Write a metadata with the normal modes information
        # to launch the nma alignment programs
        writeSetOfParticles(self.inputParticles.get(), self.imgsFn)


    def createOutputStep(self):
        pass

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
    
    #--------------------------- UTILS functions --------------------------------------------

    def getInputParticles(self):
        """ Get the output particles of the input NMA protocol. """
        return self.inputNMA.get().outputParticles

    def getParticlesMD(self):
        "Get the metadata files that contain the NMA displacement"
        return self.inputNMA.get()._getExtraPath('images.xmd')

    def getInputPdb(self):
        return self.inputNMA.get().getInputPdb()
    
    def getOutputMatrixFile(self):
        return self._getExtraPath('output_matrix.txt')
    
    def getDeformationFile(self):
        return self._getExtraPath('deformations.txt')
    
    def getProjectorFile(self):
        return self.mappingFile.get()

