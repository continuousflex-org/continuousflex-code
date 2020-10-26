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
from os.path import basename
from sh_alignment.tompy.transform import fft, ifft, fftshift, ifftshift
from .utilities.spider_files3 import save_volume, open_volume
from pyworkflow.utils import replaceBaseExt
import numpy as np
import farneback3d
from .utilities.spider_files3 import *
import time
import PIL
import os
from .utilities.OF_plots import plot_quiver_3d

REFERENCE_EXT = 0
REFERENCE_STA = 1


class FlexProtHeteroFlow(ProtAnalysis3D):
    """ Protocol for subtomogram missingwedge filling. """
    _label = 'heteroflow protocol'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputVolumes', params.PointerParam,
                      pointerClass='SetOfVolumes,Volume',
                      label="Input volume(s)", important=True,
                      help='Select volumes')
        form.addParam('StartingReference', params.EnumParam,
                      choices=['from input file', 'from STA run'],
                      default=REFERENCE_EXT,
                      label='Starting reference', display=params.EnumParam.DISPLAY_COMBO,
                      help='either an external volume file or an output volume from STA protocol')
        form.addParam('ReferenceVolume', params.FileParam,
                      pointerClass='params.FileParam', allowsNull=True,
                      condition='StartingReference==%d' % REFERENCE_EXT,
                      label="Reference volume",
                      help='Choose a reference, typically from a STA previous run')
        form.addParam('STAVolume', params.PointerParam,
                      pointerClass='Volume', allowsNull=True,
                      condition='StartingReference==%d' % REFERENCE_STA,
                      label="Subtomogram average",
                      help='Choose a reference, typically from a STA previous run')
        form.addSection(label='3D OpticalFLow parameters')
        form.addParam('pyr_scale', params.FloatParam, default=0.5,
                      label='pyr_scale',
                      help='Multiscaling relationship')
        form.addParam('levels', params.IntParam, default=4,
                      label='levels',
                      help='Number of pyramid levels')
        form.addParam('winsize', params.IntParam, default=10,
                      label='winsize',
                      help='window size')
        form.addParam('iterations', params.IntParam, default=10,
                      label='iterations',
                      help='iterations')
        form.addParam('poly_n', params.IntParam, default=5,
                      label='poly_n',
                      help='Polynomial order for the relationship between the neighborhood pixels')
        form.addParam('poly_sigma', params.FloatParam, default=1.2,
                      label='poly_sigma',
                      help='polynomial constant')
        form.addParam('flags', params.IntParam, default=0,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='flags',
                      help='flag to pass for the optical flow')
        form.addParam('factor1', params.IntParam, default=100,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='factor1',
                      help='this factor will be multiplied by the gray levels of each subtomogram')
        form.addParam('factor2', params.IntParam, default=100,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='factor2',
                      help='this factor will be multiplied by the gray levels of the reference')

    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        # Define some outputs filenames
        self.imgsFn = self._getExtraPath('volumes.xmd')
        makePath(self._getExtraPath() + '/optical_flows')

        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('doAlignmentStep')
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------
    def convertInputStep(self):
        # Write a metadata with the volumes
        xmipp3.convert.writeSetOfVolumes(self.inputVolumes.get(), self._getExtraPath('volumes.xmd'))

    def doAlignmentStep(self):
        tempdir = self._getTmpPath()
        imgFn = self.imgsFn
        StartingReference = self.StartingReference.get()
        ReferenceVolume = self.ReferenceVolume.get()

        if StartingReference == REFERENCE_STA:
            STAVolume = self.STAVolume.get().getFileName()
        else:
            STAVolume = ReferenceVolume
        # in case the reference is in MRC format:
        path_vol0 = self._getExtraPath('reference.spi')
        params = '-i ' + STAVolume + ' -o ' + path_vol0 + ' --type vol'
        self.runJob('xmipp_image_convert', params)

        pyr_scale = self.pyr_scale.get()
        levels = self.levels.get()
        iterations = self.iterations.get()
        winsize = self.winsize.get()
        poly_n = self.poly_n.get()
        poly_sigma = self.poly_sigma.get()
        flags = self.flags.get()
        factor1 = self.factor1.get()
        factor2 = self.factor2.get()

        mdImgs = md.MetaData(imgFn)
        of_root = self._getExtraPath() + '/optical_flows/'

        N = 0
        for objId in mdImgs:
            N += 1
            imgPath = mdImgs.getValue(md.MDL_IMAGE, objId)
            # getting a copy converted to spider format to solve the problem with stacks or mrc files
            tmp = self._getTmpPath('tmp.spi')
            self.runJob('xmipp_image_convert','-i ' + imgPath + ' -o ' + tmp + ' --type vol')

            print('processing optical flow for volume ', objId)
            path_flowx = of_root + str(objId).zfill(6) + '_opflowx.spi'
            path_flowy = of_root + str(objId).zfill(6) + '_opflowy.spi'
            path_flowz = of_root + str(objId).zfill(6) + '_opflowz.spi'
            path_vol_i = tmp
            volumes_op_flowi = self.opflow_vols(path_vol0, path_vol_i, pyr_scale, levels, winsize, iterations, poly_n,
                                                poly_sigma, factor1, factor2, path_flowx, path_flowy, path_flowz)

        metric_mat = np.zeros([N, N])

        for i in range(1, N + 1):
            print('finding the values for row ', i)
            flowi = self.read_optical_flow_by_number(i)
            for j in range(1, N + 1):
                flowj = self.read_optical_flow_by_number(j)
                metric_mat[i - 1, j - 1] = self.metric_opflow_vols(flowi, flowj)
        correlation_matrix = self._getExtraPath('data.csv')
        np.savetxt(correlation_matrix, metric_mat, delimiter=',')


    def createOutputStep(self):
        pass

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        return summary

    def _citations(self):
        return []

    def _methods(self):
        pass

    # --------------------------- UTILS functions --------------------------------------------
    def opflow_vols(self, path_vol0, path_vol1, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, factor1=100,
                    factor2=100, path_volx='x_OF_3D.vol', path_voly='y_OF_3D.vol', path_volz='z_OF_3D.vol'):
        # Convention here is in reverse order
        vol0 = open_volume(path_vol1)
        vol1 = open_volume(path_vol0)
        # ranges are between 0 and 3.09, the values should be changed with some factor, otherwise the output is zero
        vol0 = vol0 * factor1
        vol1 = vol1 * factor2
        optflow = farneback3d.Farneback(
            pyr_scale=pyr_scale,  # Scaling between multi-scale pyramid levels
            levels=levels,  # Number of multi-scale levels
            winsize=winsize,  # Window size for Gaussian filtering of polynomial coefficients
            num_iterations=iterations,  # Iterations on each multi-scale level
            poly_n=poly_n,  # Size of window for weighted least-square estimation of polynomial coefficients
            poly_sigma=poly_sigma,  # Sigma for Gaussian weighting of least-square estimation of polynomial coefficients
        )
        t0 = time.time()
        # perform OF:
        flow = optflow.calc_flow(vol0, vol1)
        t_end = time.time()
        print("spent on calculating 3D optical flow", np.floor((t_end - t0) / 60), "minutes and",
              np.round(t_end - t0 - np.floor((t_end - t0) / 60) * 60), "seconds")

        # Extracting the flows in x, y and z dimensions:
        Flowx = flow[0, :, :, :]
        Flowy = flow[1, :, :, :]
        Flowz = flow[2, :, :, :]

        # # See if flow has some values in:
        # print("flow_X maximum:", np.amax(Flowx), "minumum", np.amax(Flowx))
        # print("flow_Y maximum:", np.amax(Flowy), "minumum", np.amax(Flowy))
        # print("flow_Z maximum:", np.amax(Flowz), "minumum", np.amax(Flowz))

        save_volume(Flowx, path_volx)
        save_volume(Flowy, path_voly)
        save_volume(Flowz, path_volz)

        return flow

    def read_optical_flow(self, path_flowx, path_flowy, path_flowz):
        x = open_volume(path_flowx)
        y = open_volume(path_flowy)
        z = open_volume(path_flowz)
        l = np.shape(x)
        # print(l)
        flow = np.zeros([3, l[0], l[1], l[2]])
        flow[0, :, :, :] = x
        flow[1, :, :, :] = y
        flow[2, :, :, :] = z
        return flow

    def read_optical_flow_by_number(self, num):
        op_path = self._getExtraPath() + '/optical_flows/'
        path_flowx = op_path + str(num).zfill(6) + '_opflowx.spi'
        path_flowy = op_path + str(num).zfill(6) + '_opflowy.spi'
        path_flowz = op_path + str(num).zfill(6) + '_opflowz.spi'
        flow = self.read_optical_flow(path_flowx, path_flowy, path_flowz)
        return flow

    def metric_opflow_vols(self, flow1, flow2):
        # print(np.shape(flow1))
        reshaped_flow1 = np.reshape(flow1, [3, np.shape(flow1)[1] * np.shape(flow1)[2] * np.shape(flow1)[3]])
        reshaped_flow2 = np.reshape(flow2, [3, np.shape(flow2)[1] * np.shape(flow2)[2] * np.shape(flow2)[3]])
        metric = np.sum(np.multiply(reshaped_flow1, reshaped_flow2))
        return metric

    def _printWarnings(self, *lines):
        """ Print some warning lines to 'warnings.xmd',
        the function should be called inside the working dir."""
        fWarn = open("warnings.xmd", 'w')
        for l in lines:
            print >> fWarn, l
        fWarn.close()
