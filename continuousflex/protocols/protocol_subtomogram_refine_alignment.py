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
from pwem.utils import runProgram
from pwem import Domain
from xmippLib import Euler_matrix2angles, Euler_angles2matrix
from pwem.objects import Volume



REFERENCE_EXT = 0
REFERENCE_STA = 1


class FlexProtRefineSubtomoAlign(ProtAnalysis3D):
    """ Protocol for subtomogram refine alignment. """
    _label = 'refine subtomogram alignment'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Settings')
        group = form.addGroup('Choose what processes you want to perform:')
        group.addParam('FillWedge', params.BooleanParam, default=True,
                       label='Correct the missing wedge?',
                       help='This step will correct the missing wedge by filling it with the average'
                            ' of each iteration.')
        group.addParam('tiltLow', params.IntParam, default=-60,
                      condition='FillWedge==%d' % True,
                      label='Lower tilt value',
                      help='The lower tilt angle used in obtaining the tilt series')
        group.addParam('tiltHigh', params.IntParam, default=60,
                      condition='FillWedge==%d' % True,
                      label='Upper tilt value',
                      help='The upper tilt angle used in obtaining the tilt series')
        group.addParam('Alignment_refine', params.BooleanParam, default=True,
                       label='Refine the rigid-body alignment?',
                       help='This step will refine the rigid-body alignment given a previous run of subtomogram averaging'
                            ' in the workspace')
        group.addParam('NumOfIters', params.IntParam, default=3,
                       condition='Alignment_refine',
                       label='Refinment iterations', help='How many times you want to iterate to perform'
                                                         ' subtomogram alignment refinement.')
        group.addParam('ApplyAlignment', params.EnumParam,
                       label='Apply volume/subtomogam alignment?',
                       choices=['Yes'],
                       default=REFERENCE_EXT,
                       display=params.EnumParam.DISPLAY_HLIST,
                       help='This protocol by default applies previous StA alignment on the subtomograms.')

        form.addSection(label='Input')
        form.addParam('inputVolumes', params.PointerParam,
                      pointerClass='SetOfVolumes',
                      label="Input volumes/subtomograms", important=True,
                      help='Select volumes')
        group = form.addGroup('Reference volume: last iteration average of StA',
                              condition='Alignment_refine or FillWedge')
        group.addParam('StartingReference', params.EnumParam,
                      choices=['Browse for an external volume file', 'Select a volume from the project workspace'],
                      default=REFERENCE_STA,
                      label='Average volume', display=params.EnumParam.DISPLAY_COMBO)
        group.addParam('ReferenceVolume', params.FileParam,
                      pointerClass='params.FileParam', allowsNull=True,
                      condition='StartingReference==%d' % REFERENCE_EXT,
                      label="Volume path",
                      help='Choose a reference, typically from a StA previous run')
        group.addParam('STAVolume', params.PointerParam,
                      pointerClass='Volume', allowsNull=True,
                      condition='StartingReference==%d' % REFERENCE_STA,
                      label="Selected volume",
                      help='Choose a reference, typically from a StA previous run')
        group = form.addGroup('Alignment parameters: last iteration table of StA (Scipion/Xmipp metadata)')
        group.addParam('AlignmentParameters', params.EnumParam,
                      choices=['Browse for a file', 'Select a subtomogram averaging protocol '
                                                    'from the project workspace'],
                      default=REFERENCE_STA,
                      label='Alignment parameters', display=params.EnumParam.DISPLAY_COMBO,
                      help='either an external metadata file containing alignment parameters or StA run')
        group.addParam('MetaDataFile', params.FileParam,
                      pointerClass='params.FileParam', allowsNull=True,
                      condition='AlignmentParameters==%d' % REFERENCE_EXT,
                      label="File for rigid-body alignment parameters (Xmipp/Scipion MetaData)",
                      help='Alignment parameters, typically from a StA previous run')
        group.addParam('MetaDataSTA', params.PointerParam,
                      pointerClass='FlexProtSubtomogramAveraging', allowsNull=True,
                      condition='AlignmentParameters==%d' % REFERENCE_STA,
                      label="Subtomogram averaging (StA)",
                      help='A StA previous run')

        form.addSection(label='combined rigid-body & elastic alignment')
        group = form.addGroup('Optical flow parameters', condition='Alignment_refine')
        group.addParam('N_GPU', params.IntParam, default=3, important=True, allowsNull=True,
                              label = 'Parallel processes on GPU',
                              help='This parameter indicates the number of volumes that will be processed in parallel'
                                   ' (independently). The more powerful your GPU, the higher the number you can choose.')
        group.addParam('pyr_scale', params.FloatParam, default=0.5,
                      label='pyr_scale',
                      help='parameter specifying the image scale to build pyramids for each image (scale < 1).'
                           ' A classic pyramid is of generally 0.5 scale, every new layer added, it is'
                           ' halved to the previous one.')
        group.addParam('levels', params.IntParam, default=4,
                      label='levels',
                      help='evels=1 says, there are no extra layers (only the initial image).'
                           ' It is the number of pyramid layers including the first image.')
        group.addParam('winsize', params.IntParam, default=10,
                      label='winsize',
                      help='It is the average window size, larger the size, the more robust the algorithm is to noise,'
                           ' and provide smaller conformation detection, though gives blurred motion fields.'
                           ' You may try smaller window size for larger conformations but the method will be'
                           ' more sensitive to noise.')
        group.addParam('iterations', params.IntParam, default=10,
                      label='iterations',
                      help='Number of iterations to be performed at each pyramid level.')
        group.addParam('poly_n', params.IntParam, default=5,
                      label='poly_n',
                      help='It is typically 5 or 7, it is the size of the pixel neighbourhood which is used'
                           ' to find polynomial expansion between the pixels.')
        group.addParam('poly_sigma', params.FloatParam, default=1.2,
                      label='poly_sigma',
                      help='standard deviation of the gaussian that is for derivatives to be smooth as the basis of'
                           ' the polynomial expansion. It can be 1.2 for poly= 5 and 1.5 for poly= 7.')
        group.addHidden('flags', params.IntParam, default=0,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='flags',
                      help='flag to pass for the optical flow')
        group.addHidden('factor1', params.IntParam, default=100,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='factor1',
                      help='this factor will be multiplied by the gray levels of each subtomogram')
        group.addHidden('factor2', params.IntParam, default=100,
                      expertLevel=params.LEVEL_ADVANCED,
                      label='factor2',
                      help='this factor will be multiplied by the gray levels of the reference')
        group = form.addGroup('rigid-body alignment (refinement)', condition='Alignment_refine',)
        group.addParam('frm_freq', params.FloatParam, default=0.25,
                      # expertLevel=params.LEVEL_ADVANCED,
                      label='Maximum cross correlation frequency',
                      help='The normalized frequency should be between 0 and 0.5 '
                           'The more it is, the bigger the search frequency is, the more time it demands, '
                           'keeping it as default is recommended.')
        group.addParam('frm_maxshift', params.IntParam, default=4,
                      # expertlevel=params.LEVEL_ADVANCED,
                      label='Maximum shift for rigid body refinement (in pixels)',
                      help='The maximum shift is a number between 1 and half the size of your volume. '
                           'It represents the maximum distance searched in x,y and z directions.')

        form.addParallelSection(threads=0, mpi=5)


    # --------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        # Define some outputs filenames
        self.imgsFn = self._getExtraPath('volumes.xmd')
        # Make a new metadata fila that contains all the rigid-body information with updated names:
        self._insertFunctionStep('convertInputStep')
        self._insertFunctionStep('prepareMetaData')

        N = 0
        if (self.Alignment_refine.get()):
            N = self.NumOfIters.get()
            for i in range(1, N+1):
                makePath(self._getExtraPath() + '/optical_flows_' + str(i))
                if(self.FillWedge.get()):
                    self._insertFunctionStep('fillMissingWedge', i)
                self._insertFunctionStep('applyAlignment',i)
                self._insertFunctionStep('calculateOpticalFlows',i)
                self._insertFunctionStep('warpByFlow',i)
                self._insertFunctionStep('refineAlignment',i)
                self._insertFunctionStep('combineRefinedAlignment',i)
                self._insertFunctionStep('calculateNewAverage',i)

        if (self.FillWedge.get()):
            self._insertFunctionStep('fillMissingWedge', N+1)
        self._insertFunctionStep('applyAlignment', N+1)
        if N:
            self._insertFunctionStep('createOutputStep', N)
        else:
            self._insertFunctionStep('createOutputStep')
    # --------------------------- STEPS functions --------------------------------------------
    def convertInputStep(self):
        # Write a metadata with the volumes
        xmipp3.convert.writeSetOfVolumes(self.inputVolumes.get(), self._getExtraPath('input.xmd'))


    def prepareMetaData(self):
        tempdir = self._getTmpPath()
        imgFn = self.imgsFn
        AlignmentParameters = self.AlignmentParameters.get()
        MetaDataFile = self.MetaDataFile.get()
        if AlignmentParameters == REFERENCE_STA:
            MetaDataSTA = self.MetaDataSTA.get()._getExtraPath('final_md.xmd')
            MetaDataFile = MetaDataSTA
        copyFile(MetaDataFile,imgFn)
        mdImgs = md.MetaData(imgFn)
        # in case of metadata from an external file, it has to be updated with the proper filenames from 'input.xmd'
        inputSet = self.inputVolumes.get()
        for objId in mdImgs:
            imgPath = mdImgs.getValue(md.MDL_IMAGE, objId)
            index, fn = xmipp3.convert.xmippToLocation(imgPath)
            if (index):  # case the input is a stack
                # Conside the index is the id in the input set
                particle = inputSet[index]
            else:  # input is not a stack
                # convert the inputSet to metadata:
                mdtemp = md.MetaData(self._getExtraPath('input.xmd'))
                # Loop and find the index based on the basename:
                bn_retrieved = basename(imgPath)
                for searched_index in mdtemp:
                    imgPath_temp = mdtemp.getValue(md.MDL_IMAGE, searched_index)
                    bn_searched = basename(imgPath_temp)
                    if bn_searched == bn_retrieved:
                        index = searched_index
                        particle = inputSet[index]
                        break
            mdImgs.setValue(md.MDL_IMAGE, xmipp3.convert.getImageLocation(particle), objId)
            mdImgs.setValue(md.MDL_ITEM_ID, int(particle.getObjId()), objId)
        # Sorting here To avoid future problems
        mdImgs.sort()
        mdImgs.write(self.imgsFn)


    def fillMissingWedge(self, num):
        tempdir = self._getTmpPath()

        if num == 1:
            imgFn = self.imgsFn
            StartingReference = self.StartingReference.get()
            ReferenceVolume = self.ReferenceVolume.get()
            if StartingReference == REFERENCE_STA:
                STAVolume = self.STAVolume.get().getFileName()
            else:
                STAVolume = ReferenceVolume
            # in case the reference is in MRC format:
            path_vol0 = self._getExtraPath('reference' + str(num) + '.spi')
            params = '-i ' + STAVolume + ' -o ' + path_vol0 + ' --type vol'
            runProgram('xmipp_image_convert', params)

        else:
            imgFn = self._getExtraPath('combined_'+str(num-1)+'.xmd')
            STAVolume = self._getExtraPath('reference' + str(num) + '.spi')

        tiltLow = self.tiltLow.get()
        tiltHigh = self.tiltHigh.get()

        # creating a missing-wedge mask:
        start_ang = tiltLow
        end_ang = tiltHigh
        size = self.inputVolumes.get().getDim()
        MW_mask = np.ones(size)
        x, z = np.mgrid[0.:size[0], 0.:size[2]]
        x -= size[0] / 2
        ind = np.where(x)
        z -= size[2] / 2

        angles = np.zeros(z.shape)
        angles[ind] = np.arctan(z[ind] / x[ind]) * 180 / np.pi

        angles = np.reshape(angles, (size[0], 1, size[2]))
        angles = np.repeat(angles, size[1], axis=1)

        MW_mask[angles > -start_ang] = 0
        MW_mask[angles < -end_ang] = 0

        MW_mask[size[0] // 2, :, :] = 0
        MW_mask[size[0] // 2, :, size[2] // 2] = 1
        fnmask = self._getExtraPath('Mask.spi')
        save_volume(np.float32(MW_mask), fnmask)
        runProgram('xmipp_transform_geometry', '-i ' + fnmask + ' --rotate_volume euler 0 90 0')
        mdImgs = md.MetaData(imgFn)
        new_imgPath = self._getExtraPath() + '/mw_filled_' + str(num) + '/'
        makePath(new_imgPath)
        # Missing wedge filling now
        for objId in mdImgs:
            imgPath = mdImgs.getValue(md.MDL_IMAGE, objId)
            index, fname = xmipp3.convert.xmippToLocation(imgPath)
            new_imgPath = self._getExtraPath() + '/mw_filled_' + str(num) + '/'
            if index:  # case of stack
                new_imgPath += str(index).zfill(6) + '.spi'
            else:
                new_imgPath += basename(replaceBaseExt(basename(imgPath), 'spi'))
            # Get a copy of the volume converted to spider format
            params = '-i ' + imgPath + ' -o ' + new_imgPath + ' --type vol'
            runProgram('xmipp_image_convert', params)
            # print('xmipp_image_convert',params)
            # update the name in the metadata file
            mdImgs.setValue(md.MDL_IMAGE, new_imgPath, objId)
            # Align the reference with the subtomogram:
            rot = str(mdImgs.getValue(md.MDL_ANGLE_ROT, objId))
            tilt = str(mdImgs.getValue(md.MDL_ANGLE_TILT, objId))
            psi = str(mdImgs.getValue(md.MDL_ANGLE_PSI, objId))
            shiftx = str(mdImgs.getValue(md.MDL_SHIFT_X, objId))
            shifty = str(mdImgs.getValue(md.MDL_SHIFT_Y, objId))
            shiftz = str(mdImgs.getValue(md.MDL_SHIFT_Z, objId))
            # print(imgPath,rot,tilt,psi,shiftx,shifty,shiftz)
            params = '-i ' + STAVolume + ' -o ' + tempdir + '/temp.vol '
            params += '--rotate_volume euler ' + rot + ' ' + tilt + ' ' + psi + ' '
            params += '--shift ' + shiftx + ' ' + shifty + ' ' + shiftz + ' '
            if self.getAngleY()==90:
            # if self.angleY.get():
                params += '--inverse'
            # print('xmipp_transform_geometry',params)
            runProgram('xmipp_transform_geometry', params)
            if self.getAngleY() == 90:
            # if self.angleY.get():
                params = '-i ' + tempdir + '/temp.vol -o ' + tempdir + '/temp.vol '
                params += '--rotate_volume euler 0 -90 0 '
                # print('xmipp_transform_geometry',params)
                runProgram('xmipp_transform_geometry', params)
            # Now the STA is aligned, add the missing wedge region to the subtomogram:
            v = open_volume(new_imgPath)
            I = fft(v)
            I = fftshift(I)
            v_ave = open_volume(tempdir + '/temp.vol')
            Iave = fft(v_ave)
            Iave = fftshift((Iave))
            Mask = open_volume(fnmask)
            Mask = np.ones(np.shape(Mask)) - Mask
            Iave = Iave * Mask
            #
            I = I + Iave
            #
            I = ifftshift(I)
            v_result = np.float32(ifft(I))
            #
            save_volume(v_result, new_imgPath)

            # for debugging, save everything that was aligned in the first iteration
            if objId == 1:
                v_ave = open_volume(tempdir + '/temp.vol')
                save_volume(v_ave, self._getExtraPath('aligned_average_with_first_volume.spi'))

        mdImgs.write(self._getExtraPath('MWFilled_' + str(num) + '.xmd'))


    def applyAlignment(self,num):
        makePath(self._getExtraPath()+'/aligned_'+str(num))
        tempdir = self._getTmpPath()

        if (self.FillWedge.get()):
            mdImgs = md.MetaData(self._getExtraPath('MWFilled_' + str(num) + '.xmd'))
        else:
            if num == 1:
                mdImgs = md.MetaData(self.imgsFn)
            else:
                mdImgs = md.MetaData(self._getExtraPath('combined_' + str(num - 1) + '.xmd'))


        for objId in mdImgs:
            imgPath = mdImgs.getValue(md.MDL_IMAGE, objId)
            new_imgPath = self._getExtraPath() + '/aligned_'+str(num) + '/' + basename(imgPath)
            mdImgs.setValue(md.MDL_IMAGE, new_imgPath, objId)
            rot = str(mdImgs.getValue(md.MDL_ANGLE_ROT, objId))
            tilt = str(mdImgs.getValue(md.MDL_ANGLE_TILT, objId))
            psi = str(mdImgs.getValue(md.MDL_ANGLE_PSI, objId))
            shiftx = str(mdImgs.getValue(md.MDL_SHIFT_X, objId))
            shifty = str(mdImgs.getValue(md.MDL_SHIFT_Y, objId))
            shiftz = str(mdImgs.getValue(md.MDL_SHIFT_Z, objId))
            # rotate 90 around y, align, then rotate -90 to get to neutral
            params = '-i ' + imgPath + ' -o ' + tempdir + '/temp.vol '
            if self.getAngleY() == 90:
            # if(self.angleY):
                params += '--rotate_volume euler 0 90 0 '
            else: # only to convert
                params += '--rotate_volume euler 0 0 0 '
            runProgram('xmipp_transform_geometry', params)
            params = '-i ' + tempdir + '/temp.vol -o ' + new_imgPath + ' '
            params += '--rotate_volume euler ' + rot + ' ' + tilt + ' ' + psi + ' '
            params += '--shift ' + shiftx + ' ' + shifty + ' ' + shiftz + ' '
            if self.getAngleY() == 0:
            # if (not(self.angleY)):
                params += ' --inverse '

            # print('xmipp_transform_geometry',params)
            runProgram('xmipp_transform_geometry', params)
        self.fnaligned = self._getExtraPath('volumes_aligned_'+str(num)+'.xmd')
        mdImgs.write(self.fnaligned)


    def calculateOpticalFlows(self, num):
        tempdir = self._getTmpPath()
        imgFn = self._getExtraPath('volumes_aligned_'+str(num)+'.xmd')

        if num == 1:
            StartingReference = self.StartingReference.get()
            ReferenceVolume = self.ReferenceVolume.get()
            if StartingReference == REFERENCE_STA:
                STAVolume = self.STAVolume.get().getFileName()
            else:
                STAVolume = ReferenceVolume

            # in case the reference is in MRC format:
            path_vol0 = self._getExtraPath('reference' + str(num) + '.spi')
            params = '-i ' + STAVolume + ' -o ' + path_vol0 + ' --type vol'
            runProgram('xmipp_image_convert', params)

        else:
            path_vol0 = self._getExtraPath('reference' + str(num) + '.spi')


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
        of_root = self._getExtraPath() + '/optical_flows_' + str(num) + '/'

        N = 0
        for objId in mdImgs:
            N += 1
            imgPath = mdImgs.getValue(md.MDL_IMAGE, objId)
            # getting a copy converted to spider format to solve the problem with stacks or mrc files
            tmp = self._getTmpPath('tmp.spi')
            runProgram('xmipp_image_convert', '-i ' + imgPath + ' -o ' + tmp + ' --type vol')

            print('processing optical flow for volume ', objId)
            path_flowx = of_root + str(objId).zfill(6) + '_opflowx.spi'
            path_flowy = of_root + str(objId).zfill(6) + '_opflowy.spi'
            path_flowz = of_root + str(objId).zfill(6) + '_opflowz.spi'
            path_vol_i = tmp
            if (isfile(path_flowx)):
                continue
            else:
                volumes_op_flowi = self.opflow_vols(path_vol_i, path_vol0, pyr_scale, levels, winsize, iterations,
                                                    poly_n,
                                                    poly_sigma, factor1, factor2, path_flowx, path_flowy, path_flowz)


    def warpByFlow(self, num):
        makePath(self._getExtraPath() + '/estimated_volumes_' + str(num))
        estVol_root = self._getExtraPath() + '/estimated_volumes_' + str(num) + '/'
        reference = open_volume(self._getExtraPath('reference' + str(num) + '.spi'))
        # recount the number of volumes:
        imgFn = self.imgsFn
        mdImgs = md.MetaData(imgFn)

        N = 0
        for objId in mdImgs:
            N += 1

        mdWarped = md.MetaData()
        for i in range(1, N + 1):
            print('Warping a copy of the reference volume by the optical flow ', i)
            flow_i = self.read_optical_flow_by_number(i, op_path=self._getExtraPath() + '/optical_flows_' + str(num) + '/')
            warped_i = farneback3d.warp_by_flow(reference, np.float32(flow_i))
            warped_path_i = estVol_root + str(i).zfill(6) + '.spi'
            save_volume(warped_i, warped_path_i)
            mdWarped.setValue(md.MDL_IMAGE, warped_path_i, mdWarped.addObject())
            mdWarped.setValue(md.MDL_ITEM_ID, i, i)
        warpedVolFn = self._getExtraPath('warped_volumes_' + str(num) + '.xmd')
        mdWarped.write(warpedVolFn)


    def refineAlignment(self, num):
        tempdir = self._getTmpPath()
        imgFn = self._getExtraPath('warped_volumes_' + str(num) + '.xmd')

        frm_freq = self.frm_freq.get()
        frm_maxshift = self.frm_maxshift.get()

        result = self._getExtraPath('refinement_'+str(num)+'.xmd')
        reference = self._getExtraPath('reference' + str(num) + '.spi')

        print('tempdir is ', tempdir)
        print('imgFn is ', imgFn)
        print('frm_freq is ', frm_freq)
        print('frm_maxshift is ', frm_maxshift)
        print('result is ', result)
        print('reference is ', reference)

        args = "-i %(imgFn)s -o %(result)s --odir %(tempdir)s --resume --ref %(reference)s" \
               " --frm_parameters %(frm_freq)f %(frm_maxshift)d "

        self.runJob("xmipp_volumeset_align", args % locals(),
                    env=Domain.importFromPlugin('xmipp3').Plugin.getEnviron())

        mdImgs = md.MetaData(result)
        inputSet = md.MetaData(imgFn)
        # setting item_id (lost due to mpi usually)
        for objId in mdImgs:
            imgPath = mdImgs.getValue(md.MDL_IMAGE, objId)
            # Conside the index is the id in the input set
            for objId2 in inputSet:
                NewImgPath = inputSet.getValue(md.MDL_IMAGE, objId2)
                if (NewImgPath == imgPath):
                    target_ID = inputSet.getValue(md.MDL_ITEM_ID, objId2)
                    break
            mdImgs.setValue(md.MDL_ITEM_ID, target_ID, objId)
        mdImgs.sort()
        mdImgs.write(result)


    def combineRefinedAlignment(self, num):
        # 1- read both metadata (before and after refinment)
        if num == 1:
            MD_original = md.MetaData(self.imgsFn)
        else:
            MD_original = md.MetaData(self._getExtraPath('combined_'+str(num-1)+'.xmd'))

        MD_refined = md.MetaData(self._getExtraPath('refinement_'+str(num)+'.xmd'))
        # This metadata will be populated and saved
        MD_combined = md.MetaData()
        # Iteratively:
        # 2- find the transformation matrices
        for objId in MD_original:
            rot_o = MD_original.getValue(md.MDL_ANGLE_ROT, objId)
            tilt_o = MD_original.getValue(md.MDL_ANGLE_TILT, objId)
            psi_o = MD_original.getValue(md.MDL_ANGLE_PSI, objId)
            shiftx_o = MD_original.getValue(md.MDL_SHIFT_X, objId)
            shifty_o = MD_original.getValue(md.MDL_SHIFT_Y, objId)
            shiftz_o = MD_original.getValue(md.MDL_SHIFT_Z, objId)

            rot_r = MD_refined.getValue(md.MDL_ANGLE_ROT, objId)
            tilt_r = MD_refined.getValue(md.MDL_ANGLE_TILT, objId)
            psi_r = MD_refined.getValue(md.MDL_ANGLE_PSI, objId)
            shiftx_r = MD_refined.getValue(md.MDL_SHIFT_X, objId)
            shifty_r = MD_refined.getValue(md.MDL_SHIFT_Y, objId)
            shiftz_r = MD_refined.getValue(md.MDL_SHIFT_Z, objId)

            T1 = Euler_angles2matrix(rot_o, tilt_o, psi_o)
            T_o = np.zeros([4,4])
            T_o[:3,:3] = T1
            T_o[0,3] = shiftx_o
            T_o[1,3] = shifty_o
            T_o[2,3] = shiftz_o
            T_o[3,3] = 1

            T2 = Euler_angles2matrix(rot_r, tilt_r, psi_r)
            T_r = np.zeros([4,4])
            T_r[:3, :3] = T2
            T_r[0,3] = shiftx_r
            T_r[1,3] = shifty_r
            T_r[2,3] = shiftz_r
            T_r[3,3] = 1

            # 3- multiply the matrices
            if self.getAngleY() == 90:
            # if(self.angleY.get()):
                # In this case the refinement matrix should be inverted
                T_r_inv = np.linalg.inv(T_r)
                T_shift = np.matmul(T_r_inv,T_o)
                # T_shift = np.matmul(T_o, T_r_inv)
                T2_inv = np.linalg.inv(T2)
                # This is taken separately to avoid numerical errors
                T_ang= np.matmul(T2_inv,T1)
                # T_ang = np.matmul(T1,T2_inv)
            else:
                # In this case the refinement matrix should be used as it is
                T_shift = np.matmul(T_o, T_r)
                # T_shift = np.matmul(T_r,T_o)
                # This is taken separately to avoid numerical errors
                T_ang = np.matmul(T1, T2)
                # T_ang = np.matmul(T2, T1)

            # 4- Find the angles and shifts of the overall matrix
            rot_i, tilt_i, psi_i = Euler_matrix2angles(T_ang)
            # print(T1-T_ang)
            x_i = T_shift[0, 3]
            y_i = T_shift[1, 3]
            z_i = T_shift[2, 3]

            # Populate the metadata
            name_i = MD_original.getValue(md.MDL_IMAGE, objId)
            MD_combined.setValue(md.MDL_IMAGE, name_i, MD_combined.addObject())
            MD_combined.setValue(md.MDL_ANGLE_ROT, rot_i, objId)
            MD_combined.setValue(md.MDL_ANGLE_TILT, tilt_i, objId)
            MD_combined.setValue(md.MDL_ANGLE_PSI, psi_i, objId)
            MD_combined.setValue(md.MDL_SHIFT_X, float(x_i), objId)
            MD_combined.setValue(md.MDL_SHIFT_Y, float(y_i), objId)
            MD_combined.setValue(md.MDL_SHIFT_Z, float(z_i), objId)
            if self.getAngleY() == 90:
            # if(self.angleY.get()):
                MD_combined.setValue(md.MDL_ANGLE_Y, 90.0, objId)
            else:
                MD_combined.setValue(md.MDL_ANGLE_Y, 0.0, objId)
            MD_combined.setValue(md.MDL_ITEM_ID, objId, objId)
        # Save the metadata
        MD_combined.write(self._getExtraPath('combined_'+str(num)+'.xmd'))


    def calculateNewAverage(self, num):
        flag = self.getAngleY() == 90
        # flag = self.angleY.get()

        volumesMd = self._getExtraPath('combined_'+str(num)+'.xmd')
        mdVols = md.MetaData(volumesMd)

        counter = 0
        first = True
        for objId in mdVols:
            counter = counter + 1
            imgPath = mdVols.getValue(md.MDL_IMAGE, objId)

            rot = mdVols.getValue(md.MDL_ANGLE_ROT, objId)
            tilt = mdVols.getValue(md.MDL_ANGLE_TILT, objId)
            psi = mdVols.getValue(md.MDL_ANGLE_PSI, objId)
            x_shift = mdVols.getValue(md.MDL_SHIFT_X, objId)
            y_shift = mdVols.getValue(md.MDL_SHIFT_Y, objId)
            z_shift = mdVols.getValue(md.MDL_SHIFT_Z, objId)

            # The flip one is used here to determine if we need to use the option --inverse
            # with xmipp_transform_geometry
            flip = mdVols.getValue(md.MDL_ANGLE_Y, objId)
            # The new reference is for the next iteration
            outputVol = self._getExtraPath('reference' + str(num+1) + '.spi')
            tempVol = self._getExtraPath('temp.vol')
            extra = self._getExtraPath()

            if flag == 0 :
                if first:
                    print("THERE IS NO COMPENSATION FOR THE MISSING WEDGE")
                    first = False

                params = '-i %(imgPath)s -o %(tempVol)s --inverse --rotate_volume euler %(rot)s %(tilt)s %(psi)s' \
                         ' --shift %(x_shift)s %(y_shift)s %(z_shift)s -v 0' % locals()

            else:
                if first:
                    print("THERE IS A COMPENSATION FOR THE MISSING WEDGE")
                    first = False
                # First got to rotate each volume 90 degrees about the y axis, align it, then rotate back and sum it
                params = '-i %(imgPath)s -o %(tempVol)s --rotate_volume euler 0 90 0' % locals()
                runProgram('xmipp_transform_geometry', params)
                params = '-i %(tempVol)s -o %(tempVol)s --rotate_volume euler %(rot)s %(tilt)s %(psi)s' \
                         ' --shift %(x_shift)s %(y_shift)s %(z_shift)s ' % locals()

            runProgram('xmipp_transform_geometry', params)

            if counter == 1 :
                os.system("mv %(tempVol)s %(outputVol)s" % locals())

            else:
                params = '-i %(tempVol)s --plus %(outputVol)s -o %(outputVol)s ' % locals()
                runProgram('xmipp_image_operate', params)

        params = '-i %(outputVol)s --divide %(counter)s -o %(outputVol)s ' % locals()
        runProgram('xmipp_image_operate', params)
        os.system("rm -f %(tempVol)s" % locals())


    def createOutputStep(self, num =1):
        # now creating the output set of aligned volumes:
        out_mdfn = self._getExtraPath('volumes_aligned_'+str(num)+'.xmd')
        partSet = self._createSetOfVolumes('aligned')
        xmipp3.convert.readSetOfVolumes(out_mdfn, partSet)
        partSet.setSamplingRate(self.inputVolumes.get().getSamplingRate())
        # now creating the output of the final average:
        if (self.Alignment_refine.get()):
            outvolume = Volume()
            outvolume.setSamplingRate((self.inputVolumes.get().getSamplingRate()))
            outvolume.setFileName(self._getExtraPath('reference' + str(num+1) + '.spi'))
            self._defineOutputs(OutputVolumes=partSet, RefinedAverage=outvolume)
        else:
            self._defineOutputs(OutputVolumes=partSet)

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
        # TODO: find a way to automate this normalization
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

    def read_optical_flow_by_number(self, num, op_path = None):
        if op_path is None:
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

    def getAngleY(self):
        AlignmentParameters = self.AlignmentParameters.get()
        MetaDataFile = self.MetaDataFile.get()
        if AlignmentParameters == REFERENCE_STA:
            MetaDataFile = self.MetaDataSTA.get()._getExtraPath('final_md.xmd')
        try:
            mdFile = md.MetaData(MetaDataFile)
            angleY = mdFile.getValue(md.MDL_ANGLE_Y, 1)
        except:
            angleY = 0

        return int(angleY)
