# **************************************************************************
# * Authors:    Mohamad Harastani            (mohamad.harastani@upmc.fr)
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

from pwem.protocols import ProtAnalysis3D
import xmipp3.convert
import pwem.emlib.metadata as md
import pyworkflow.protocol.params as params
from pyworkflow.utils.path import makePath, copyFile
from os.path import basename, isfile
from .utilities.spider_files3 import save_volume, open_volume
from pyworkflow.utils import replaceBaseExt
import numpy as np
from pwem.utils import runProgram
from pwem.emlib.image import ImageHandler
from skimage.exposure import match_histograms

class FlexProtHistogramMatch(ProtAnalysis3D):
    """ Protocol for volume histogram matching. """
    _label = 'histogram matching'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputVolumes', params.PointerParam,
                      pointerClass='SetOfVolumes,Volume',
                      label="Input volume(s)", important=True,
                      help='Select a volume of a set of volumes')
        form.addParam('reference', params.PointerParam,
                      pointerClass='Volume',
                      label="Reference volume", important=True,
                      help='Select a reference volume')


    # --------------------------- INSERT steps functions --------------------------------------------

    def _insertAllSteps(self):
        # Define some outputs filenames
        self.imgsFn = self._getExtraPath('volumes.xmd')
        makePath(self._getExtraPath() + '/histogram_matched')
        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('doHistogramMatchingStep')
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------
    def convertInputStep(self):
        # Write a metadata with the volumes
        try:
            xmipp3.convert.writeSetOfVolumes(self.inputVolumes.get(), self._getExtraPath('input.xmd'))
        except:
            mdF = md.MetaData()
            mdF.setValue(md.MDL_IMAGE, self.inputVolumes.get().getFileName(), mdF.addObject())
            mdF.write(self.imgsFn)
            pass

    def doHistogramMatchingStep(self):
        # looping on all images and performing mwr
        reference = self.reference.get().getFileName()
        # xmipp convert to spider format just in case:
        params = '-i ' + reference + ' -o ' + self._getExtraPath('reference.spi')
        runProgram('xmipp_image_convert', params)
        ref = ImageHandler().read(self._getExtraPath('reference.spi')).getData()
        ref = np.squeeze(ref)
        mdImgs = md.MetaData(self._getExtraPath('input.xmd'))
        for objId in mdImgs:
            imgPath = mdImgs.getValue(md.MDL_IMAGE, objId)
            index, fname = xmipp3.convert.xmippToLocation(imgPath)
            new_imgPath = self._getExtraPath() + '/histogram_matched/'
            if index:  # case of stack
                new_imgPath += str(index).zfill(6) + '.spi'
            else:
                new_imgPath += basename(replaceBaseExt(basename(imgPath), 'spi'))
            # Get a copy of the volume converted to spider format
            temp_path = self._getTmpPath('temp.spi')
            # params = '-i ' + imgPath + ' -o ' + new_imgPath + ' --type vol'
            params = '-i ' + imgPath + ' -o ' + temp_path + ' --type vol'
            runProgram('xmipp_image_convert', params)
            # perform the mwr:
            # in case the file exists (continuing or injecting)
            if (isfile(new_imgPath)):
                continue
            else:
                v = ImageHandler().read(temp_path).getData()
                v = np.squeeze(v)
                map = match_histograms(v, ref)
                save_volume(np.float32(map), new_imgPath)
            # update the name in the metadata file
            mdImgs.setValue(md.MDL_IMAGE, new_imgPath, objId)
        mdImgs.write(self.imgsFn)

    def createOutputStep(self):
        partSet = self._createSetOfVolumes('histogram_matched')
        xmipp3.convert.readSetOfVolumes(self._getExtraPath('volumes.xmd'), partSet)
        partSet.setSamplingRate(self.inputVolumes.get().getSamplingRate())
        self._defineOutputs(HistogramMatched=partSet)


    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        return summary

    def _citations(self):
        return ['']

    def _methods(self):
        pass

    # --------------------------- UTILS functions --------------------------------------------
    def _printWarnings(self, *lines):
        """ Print some warning lines to 'warnings.xmd',
        the function should be called inside the working dir."""
        fWarn = open("warnings.xmd", 'w')
        for l in lines:
            print >> fWarn, l
        fWarn.close()