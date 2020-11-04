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
from os.path import basename, join, exists, isfile

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
                      choices=['From an external volume file', 'Select a volume'],
                      default=REFERENCE_EXT,
                      label='Reference volume', display=params.EnumParam.DISPLAY_COMBO,
                      help='Either an external volume file or a subtomogram average')
        form.addParam('ReferenceVolume', params.FileParam,
                      pointerClass='params.FileParam', allowsNull=True,
                      condition='StartingReference==%d' % REFERENCE_EXT,
                      label="Reference volume",
                      help='Choose a reference, typically from a STA previous run')
        form.addParam('STAVolume', params.PointerParam,
                      pointerClass='Volume', allowsNull=True,
                      condition='StartingReference==%d' % REFERENCE_STA,
                      label="Selected volume",
                      help='Choose a reference, typically from a STA previous run')
        form.addParam('WarpAndEstimate', params.BooleanParam,
                      expertLevel=params.LEVEL_ADVANCED,
                      default=True,
                      label='Save a warped version of the reference for each input volume?',
                      help='This will create a set of volumes, representing the fitted version of the input volumes, '
                           'using the calculated optical flows, and calculate the cross correlation, mean square '
                           'distance and the mean absolute distance between the input volumes and estimated volumes')
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
        if (self.WarpAndEstimate.get()):
            self._insertFunctionStep('warpByFlow')
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
            self.runJob('xmipp_image_convert', '-i ' + imgPath + ' -o ' + tmp + ' --type vol')

            print('processing optical flow for volume ', objId)
            path_flowx = of_root + str(objId).zfill(6) + '_opflowx.spi'
            path_flowy = of_root + str(objId).zfill(6) + '_opflowy.spi'
            path_flowz = of_root + str(objId).zfill(6) + '_opflowz.spi'
            path_vol_i = tmp
            if (isfile(path_flowx)):
                continue
            else:
                volumes_op_flowi = self.opflow_vols(path_vol0, path_vol_i, pyr_scale, levels, winsize, iterations,
                                                    poly_n,
                                                    poly_sigma, factor1, factor2, path_flowx, path_flowy, path_flowz)

        metric_mat = np.zeros([N, N])

        for i in range(1, N + 1):
            print('finding the correlation matrix row ', i)
            flowi = self.read_optical_flow_by_number(i)
            for j in range(1, N + 1):
                print('        column', j)
                flowj = self.read_optical_flow_by_number(j)
                metric_mat[i - 1, j - 1] = self.metric_opflow_vols(flowi, flowj)
        correlation_matrix = self._getExtraPath('data.csv')
        np.savetxt(correlation_matrix, metric_mat, delimiter=',')

    def warpByFlow(self):
        makePath(self._getExtraPath() + '/estimated_volumes')
        estVol_root = self._getExtraPath() + '/estimated_volumes/'
        reference_fn = self._getExtraPath('reference.spi')
        reference = open_volume(reference_fn)
        stat_mat_fn = self._getExtraPath('cc_msd_mad.txt')
        # recount the number of volumes:
        imgFn = self.imgsFn
        mdImgs = md.MetaData(imgFn)

        N = 0
        for objId in mdImgs:
            N += 1

        for i in range(1, N + 1):
            print('Warping a copy of the reference volume by the optical flow ', i)
            flow_i = self.read_optical_flow_by_number(i)
            warped_i = farneback3d.warp_by_flow(reference, flow_i)
            warped_path_i = estVol_root + str(i).zfill(6) + '.spi'
            save_volume(warped_i, warped_path_i)

        # Find a matrix of metrics (normalized cross correlation, mean square distance, mean absolute distance)
        stat_mat = np.zeros([N, 3])
        mdImgs = md.MetaData(imgFn)
        i = 0
        for objId in mdImgs:
            print('Finding the cross corrlation, mean square distance and mean absolute distance for volume ', i + 1)
            imgPath = mdImgs.getValue(md.MDL_IMAGE, objId)
            # getting a copy converted to spider format to solve the problem with stacks or mrc files
            tmp = self._getTmpPath('tmp.spi')
            self.runJob('xmipp_image_convert', '-i ' + imgPath + ' -o ' + tmp + ' --type vol')
            vol_i = open_volume(tmp)
            warped_path_i = estVol_root + str(i + 1).zfill(6) + '.spi'
            warped_i = open_volume(warped_path_i)
            stat_mat[i, 0] = self.ncc(warped_i,vol_i)
            stat_mat[i, 1] = self.vmsq(warped_i,vol_i)
            stat_mat[i, 2] = self.vmab(warped_i,vol_i)
            i += 1
        np.savetxt(stat_mat_fn, stat_mat)
        pass

    def createOutputStep(self):
        if (self.WarpAndEstimate.get()):
            # first making a metadata for the wrapped volumes:
            out_mdfn = self._getExtraPath('volumes_out.xmd')
            pattern = '"' + self._getExtraPath() + '/estimated_volumes/*.spi"'
            command = '-p ' + pattern + ' -o ' + out_mdfn
            self.runJob('xmipp_metadata_selfile_create',command)
            # now creating the output set of volumes:
            partSet = self._createSetOfVolumes('Warped')
            xmipp3.convert.readSetOfVolumes(out_mdfn, partSet)
            partSet.setSamplingRate(self.inputVolumes.get().getSamplingRate())
            self._defineOutputs(WarpedRefByFlows=partSet)
            pass
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

    def normalize(self, v):
        """Normalize the data.
        @param v: input volume.
        @return: Normalized volume.
        """
        m = np.mean(v)
        v = v - m
        s = np.std(v)
        v = v / s
        return v

    def ncc(self, v1, v2):
        """Compute the Normalized Cross Correlation between the two volumes.
        @param v1: volume 1.
        @param v2: volume 2.
        @return: NCC.
        """
        vv1 = self.normalize(v1)
        vv2 = self.normalize(v2)
        score = np.sum(vv1 * vv2) / vv1.size
        return score

    def vmsq(self, v1, v2):
        """Compute the normalized mean square distance between the two volumes
        :param v1: volume1
        :param v2: volume2
        :return: score for mean square diff
        """
        from sklearn.metrics import mean_squared_error
        score = mean_squared_error(np.ndarray.flatten(v1), np.ndarray.flatten(v2)) / mean_squared_error(
            np.ndarray.flatten(v1),
            np.zeros([np.size(np.ndarray.flatten(v1))]))
        return score

    def vmab(self, v1, v2):
        """Compute the normalized mean absolute distance between the two volumes
        :param v1: volume1
        :param v2: volume2
        :return: score for mean square diff
        """
        from sklearn.metrics import mean_absolute_error
        score = mean_absolute_error(np.ndarray.flatten(v1), np.ndarray.flatten(v2)) / mean_absolute_error(
            np.ndarray.flatten(v1),
            np.zeros([np.size(np.ndarray.flatten(v1))]))
        return score
