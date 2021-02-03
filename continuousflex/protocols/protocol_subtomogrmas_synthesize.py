# **************************************************************************
# *
# * Authors:  Mohamad Harastani          (mohamad.harastani@upmc.fr)
# * TODO: Add Remi
# *
# * IMPMC, UPMC Sorbonne University
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


from os.path import basename

from pwem.convert.atom_struct import cifToPdb
from pyworkflow.utils import replaceBaseExt, replaceExt

from pyworkflow.utils import isPower2, getListFromRangeString
from pyworkflow.utils.path import copyFile, cleanPath
import pyworkflow.protocol.params as params
from pwem.protocols import ProtAnalysis3D

from pyworkflow.protocol.params import NumericRangeParam
import pwem as em
import pwem.emlib.metadata as md
from xmipp3.base import XmippMdRow
from xmipp3.convert import (writeSetOfParticles, xmippToLocation,
                            getImageLocation, createItemMatrix,
                            setXmippAttributes)
from .convert import modeToRow
from pwem.objects import AtomStruct, Volume
import xmipp3
import os
import numpy as np

NMA_ALIGNMENT_WAV = 0
NMA_ALIGNMENT_PROJ = 1

MODE_RELATION_LINEAR = 0
MODE_RELATION_3CLUSTERS = 1
MODE_RELATION_5CLUSTERS = 2
MODE_RELATION_MESH = 3
MODE_RELATION_RANDOM = 4


MISSINGWEDGE_YES = 0
MISSINGWEDGE_NO = 1

ROTATION_SHIFT_YES = 0
ROTATION_SHIFT_NO = 1

RECONSTRUCTION_FOURIER = 0
RECONSTRUCTION_WBP = 1

NOISE_CTF_YES = 0
NOISE_CTF_NO = 1

FULL_TOMOGRAM_YES= 0
FULL_TOMOGRAM_NO = 1

class FlexProtSynthesizeSubtomo(ProtAnalysis3D):
    """ Protocol for flexible angular alignment. """
    _label = 'synthesize subtomograms'

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('inputModes', params.PointerParam, pointerClass='SetOfNormalModes',
                      label="Normal modes",
                      help='Set of modes computed by normal mode analysis.')
        form.addParam('modeList', NumericRangeParam,
                      label="Modes selection",
                      help='Select the normal modes that will be used for image analysis. \n'
                           'If you leave this field empty, all computed modes will be selected for image analysis.\n'
                           'You have several ways to specify the modes.\n'
                           '   Examples:\n'
                           ' "7,8-10" -> [7,8,9,10]\n'
                           ' "8, 10, 12" -> [8,10,12]\n'
                           ' "8 9, 10-12" -> [8,9,10,11,12])\n')
        form.addParam('modeRelationChoice', params.EnumParam, default=MODE_RELATION_LINEAR,
                      choices=['Linear relationship', 'Clusters (3 clusters)', 'Clusters (5 clusters)', 'Mesh', 'Random'],
                      label='Relationship between the modes',
                      help='TODO')
        form.addParam('numberOfVolumes', params.IntParam, default=1,
                      label='Number of volumes',
                      help='later')
        form.addParam('samplingRate', params.FloatParam, default=1.0,
                      label='Sampling Rate',
                      help='later')
        form.addParam('volumeSize', params.IntParam, default=64,
                      label='Volume Size',
                      help='later')

        form.addSection(label='Missing wedge parameters')
        form.addParam('missingWedgeChoice', params.EnumParam, default=MISSINGWEDGE_YES,
                      choices=['Simulate missing wedge artefacts', 'No missing wedge'],
                      label='Missing Wedge choice',
                      help='TODO')
        form.addParam('tiltStep', params.IntParam, default=1,
                      label='tilt step angle',
                      help='later')
        form.addParam('tiltLow', params.IntParam, default=-60,
                      condition='missingWedgeChoice==%d' % MISSINGWEDGE_YES,
                      label='Lower tilt value',
                      help='The lower tilt angle used in obtaining the tilt series')
        form.addParam('tiltHigh', params.IntParam, default=60,
                      condition='missingWedgeChoice==%d' % MISSINGWEDGE_YES,
                      label='Upper tilt value',
                      help='The upper tilt angle used in obtaining the tilt series')

        form.addSection(label='Noise and CTF')
        form.addParam('noiseCTFChoice', params.EnumParam, default=ROTATION_SHIFT_YES,
                      choices=['Yes', 'No'],
                      label='Apply Noise and CTF',
                      help='TODO')
        form.addParam('targetSNR', params.FloatParam, default=0.1,
                      condition='noiseCTFChoice==%d' % NOISE_CTF_YES,
                      label='Tagret SNR',
                      help='TODO')
        form.addParam('ctfVoltage', params.FloatParam, default=200.0,
                      condition='noiseCTFChoice==%d' % NOISE_CTF_YES,
                      label='CTF Voltage',
                      help='TODO')
        form.addParam('ctfSphericalAberration', params.FloatParam, default=2,
                      condition='noiseCTFChoice==%d' % NOISE_CTF_YES,
                      label='CTF Spherical Aberration',
                      help='TODO')
        form.addParam('ctfMagnification', params.FloatParam, default=50000.0,
                      condition='noiseCTFChoice==%d' % NOISE_CTF_YES,
                      label='CTF Magnification',
                      help='TODO')
        form.addParam('ctfDefocusU', params.FloatParam, default=-10000.0,
                      condition='noiseCTFChoice==%d' % NOISE_CTF_YES,
                      label='CTF DefocusU',
                      help='TODO')
        form.addParam('ctfDefocusV', params.FloatParam, default=-10000.0,
                      condition='noiseCTFChoice==%d' % NOISE_CTF_YES,
                      label='CTF DefocusV',
                      help='TODO')
        form.addParam('ctfQ0', params.FloatParam, default=-0.112762,
                      condition='noiseCTFChoice==%d' % NOISE_CTF_YES,
                      label='CTF Q0',
                      help='TODO')

        form.addSection('Reconstruction')
        form.addParam('reconstructionChoice', params.EnumParam, default=ROTATION_SHIFT_YES,
                      choices=['Fourier interpolation', ' Weighted BackProjection'],
                      label='Reconstruction method',
                      help='TODO')

        form.addSection('Rigid body alignment')
        form.addParam('rotationShiftChoice', params.EnumParam, default=ROTATION_SHIFT_YES,
                      choices=['Yes', 'No'],
                      label='Simulate Rotations and Shifts',
                      help='TODO')
        form.addParam('maxShift', params.FloatParam, default=5.0,
                      label='Maximum Shift',
                      help='TODO')

        form.addSection('Generate full tomogram')
        form.addParam('fullTomogramChoice', params.EnumParam, default=FULL_TOMOGRAM_YES,
                      choices=['Yes', 'No'],
                      label='Generate full tomogram',
                      help='TODO')
        form.addParam('numberOfTomograms', params.IntParam, default=1,
                      condition='fullTomogramChoice==%d' % FULL_TOMOGRAM_YES,
                      label='Number of tomograms',
                      help='TODO')
        form.addParam('tomoSizeX', params.IntParam, default=512,
                      condition='fullTomogramChoice==%d' % FULL_TOMOGRAM_YES,
                      label='Tomogram Size X',
                      help='TODO')
        form.addParam('tomoSizeY', params.IntParam, default=512,
                      condition='fullTomogramChoice==%d' % FULL_TOMOGRAM_YES,
                      label='Tomogram Size Y',
                      help='TODO')
        form.addParam('tomoSizeZ', params.IntParam, default=128,
                      condition='fullTomogramChoice==%d' % FULL_TOMOGRAM_YES,
                      label='Tomogram Size Z',
                      help='TODO')
        form.addParam('boxSize', params.IntParam, default=64,
                      condition='fullTomogramChoice==%d' % FULL_TOMOGRAM_YES,
                      label='Box Size',
                      help='TODO')

        # form.addParallelSection(threads=0, mpi=8)

        # --------------------------- INSERT steps functions --------------------------------------------

    def getInputPdb(self):
        """ Return the Pdb object associated with the normal modes. """
        return self.inputModes.get().getPdb()

    def _insertAllSteps(self):
        self._insertFunctionStep("generate_deformations")
        self._insertFunctionStep("generate_volume_from_pdb")

        if self.rotationShiftChoice == ROTATION_SHIFT_YES:
            self._insertFunctionStep("generate_rotation_and_shift")

        if self.fullTomogramChoice == FULL_TOMOGRAM_YES:
            self._insertFunctionStep("create_phantom")
            self._insertFunctionStep("map_volumes_to_tomogram")

        self._insertFunctionStep("project_volumes")

        if self.noiseCTFChoice == NOISE_CTF_YES:
            self._insertFunctionStep("apply_noise_and_ctf")

        if self.fullTomogramChoice == FULL_TOMOGRAM_YES:
            pass
        else:
            self._insertFunctionStep("reconstruct")

        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions --------------------------------------------
    def convertInputStep(self, atomsFn):
        pass
        # Write the modes metadata taking into account the selection
        # self.writeModesMetaData()
        # Write a metadata with the normal modes information
        # to launch the nma alignment programs
        # writeSetOfParticles(self.inputParticles.get(), self.imgsFn)

    # This is now done differently (see _insertAllSteps) and this line must be removed now
    # Copy the atoms file to current working dir
    # copyFile(atomsFn, self.atomsFn)

    def generate_deformations(self):
        # use the input relationship between the modes to generate normal mode amplitudes metadata

        fnPDB = self.inputModes.get().getPdb().getFileName()
        fnModeList = replaceExt(self.inputModes.get().getFileName(),'xmd')

        numberOfModes = self.inputModes.get().getSize()
        modeSelection = np.array(getListFromRangeString(self.modeList.get()))
        deformationFile = self._getExtraPath('GroundTruth.xmd')
        subtomogramMD = md.MetaData()
        # these vairables for the generation in 3 clusters
        cluster1 = cluster2 = cluster3 = False
        # these variables for the generaton like a mesh (has to be 121 volumes)
        XX, YY = np.meshgrid(np.linspace(start=-150, stop=150, num=11), np.linspace(start=150, stop=-150, num=11))
        mode7_samples = XX.reshape(-1)
        mode8_samples = YY.reshape(-1)
        # iterate over the number of outputs desired
        for i in range(self.numberOfVolumes.get()):
            deformations = np.zeros(numberOfModes)

            if self.modeRelationChoice == MODE_RELATION_LINEAR:
                amplitude = 200 # TODO : choice of amplitude
                deformations[modeSelection - 1] = np.ones(len(modeSelection))*np.random.uniform(-amplitude, amplitude)
            elif self.modeRelationChoice == MODE_RELATION_3CLUSTERS:
                # TODO: add the option to the centers and the diameter of each cluster (now the diameter is zero)
                center1 = (-150, 0)
                center2 = (+150, 0)
                center3 = (0, +150)
                if not(cluster1 or cluster2 or cluster3):
                    cluster1 = True
                if cluster3:
                    deformations[modeSelection - 1] = center3
                    cluster3 = False
                if cluster2:
                    deformations[modeSelection - 1] = center2
                    cluster2 = False
                    cluster3 = True
                if cluster1:
                    deformations[modeSelection - 1] = center1
                    cluster1 = False
                    cluster2 = True
            elif self.modeRelationChoice == MODE_RELATION_5CLUSTERS:
                pass
            elif self.modeRelationChoice == MODE_RELATION_RANDOM:
                amplitude=200
                deformations[modeSelection-1] = np.random.uniform(-amplitude, amplitude, len(modeSelection))
            elif self.modeRelationChoice == MODE_RELATION_MESH:
                new_point=(mode7_samples[i],mode8_samples[i])
                deformations[modeSelection - 1] = new_point

            # we won't keep the first 6 modes
            deformations = deformations[6:]

            params = " --pdb " + fnPDB
            params+= " --nma " + fnModeList
            params+= " -o " + self._getExtraPath(str(i+1).zfill(5)+'_deformed.pdb')
            params+= " --deformations " + ' '.join(str(i) for i in deformations)
            self.runJob('xmipp_pdb_nma_deform', params)

            subtomogramMD.setValue(md.MDL_IMAGE, self._getExtraPath(str(i+1)+'.spi'), subtomogramMD.addObject())
            subtomogramMD.setValue(md.MDL_NMA, list(deformations), i+1)

        subtomogramMD.write(deformationFile)


    def generate_volume_from_pdb(self):
        for i in range(self.numberOfVolumes.get()):
            params = " -i " + self._getExtraPath(str(i + 1).zfill(5) + '_deformed.pdb')
            params += " --sampling " + str(self.samplingRate.get())
            params += " --size " + str(self.volumeSize.get())
            params += " -v 0 --centerPDB "
            self.runJob('xmipp_volume_from_pdb', params)

    def generate_rotation_and_shift(self):
        subtomogramMD = md.MetaData(self._getExtraPath('GroundTruth.xmd'))
        for i in range(self.numberOfVolumes.get()):
            rot1 = 360 * np.random.uniform()
            tilt1 = 180 * np.random.uniform()
            psi1 = 360 * np.random.uniform()
            shift_x1 = self.maxShift.get() * np.random.uniform(-1,1)
            shift_y1 = self.maxShift.get() * np.random.uniform(-1,1)
            shift_z1 = self.maxShift.get() * np.random.uniform(-1,1)

            params = " -i " + self._getExtraPath(str(i + 1).zfill(5) + '_deformed.vol')
            params += " -o " + self._getExtraPath(str(i + 1).zfill(5) + '_deformed.vol')
            params += " --rotate_volume euler " + str(rot1) + ' ' + str(tilt1) + ' ' + str(psi1)
            params += " --shift " + str(shift_x1) + ' ' + str(shift_y1) + ' ' + str(shift_z1)
            params += " --dont_wrap "
            self.runJob('xmipp_transform_geometry', params)

            subtomogramMD.setValue(md.MDL_SHIFT_X, shift_x1, i + 1)
            subtomogramMD.setValue(md.MDL_SHIFT_Y, shift_y1, i + 1)
            subtomogramMD.setValue(md.MDL_SHIFT_Z, shift_z1, i + 1)
            subtomogramMD.setValue(md.MDL_ANGLE_ROT, rot1, i + 1)
            subtomogramMD.setValue(md.MDL_ANGLE_TILT, tilt1, i + 1)
            subtomogramMD.setValue(md.MDL_ANGLE_PSI, psi1, i + 1)
        subtomogramMD.write(self._getExtraPath('GroundTruth.xmd'))

    def create_phantom(self):
        tomoSizeX = self.tomoSizeX.get()
        tomoSizeY = self.tomoSizeY.get()
        tomoSizeZ = self.tomoSizeZ.get()
        with open(self._getExtraPath('tomogram.param'), 'a') as file:
            file.write(
                "\n".join([
                        "# XMIPP_STAR_1 *",
                        "data_block1",
                        " _dimensions3D  '%(tomoSizeX)s %(tomoSizeY)s %(tomoSizeZ)s'" % locals(),
                        " _phantomBGDensity  0",
                        " _scale  1",
                        "data_block2",
                        "loop_",
                        " _featureType",
                        " _featureOperation",
                        " _featureDensity",
                        " _featureCenter",
                        " _featureSpecificVector",
                        " cub + 0.0 '0 0 0' '0 0 0 0 0 0'"]))

        # generate a phantom for each tomograms
        for i in range(self.numberOfTomograms.get()):
            params = " -i " + self._getExtraPath('tomogram.param')
            params += " -o " + self._getExtraPath(str(i+1).zfill(5) +'_tomogram.vol')
            self.runJob('xmipp_phantom_create', params)

    def map_volumes_to_tomogram(self):
        tomoSizeX = self.tomoSizeX.get()
        tomoSizeY = self.tomoSizeY.get()
        tomoSizeZ = self.tomoSizeZ.get()
        boxSize = self.boxSize.get()

        # create a 2D grid of the tomogram at the size of a box
        boxGrid = np.mgrid[boxSize // 2: tomoSizeX - boxSize // 2 + 1 :  boxSize,
                           boxSize // 2: tomoSizeY - boxSize // 2 + 1:  boxSize]

        #get the grid positions as an array
        numberOfBoxes = boxGrid.shape[1]*boxGrid.shape[2]
        boxPositions = boxGrid.reshape(2, numberOfBoxes).T

        # The number of particles per tomogram is the integer division of the number of volumes
        #  and the number of tomograms
        numberOfTomograms = self.numberOfTomograms.get()
        particlesPerTomogram = self.numberOfVolumes.get()//numberOfTomograms

        for t in range(numberOfTomograms):

            # Shuffle the positions order to fill the tomogram boxes in random order
            np.random.shuffle(boxPositions)

            # if the number of particles per tomograms is > of the number of boxes, some particles are ignored
            for i in range(np.min([particlesPerTomogram, numberOfBoxes])):

                # Create a metadata file per volumes mapped per tomograms
                tomogramMapMD = md.MetaData()
                tomogramMapMD.setValue(md.MDL_IMAGE, self._getExtraPath(str(i+1).zfill(5) + '.vol'), tomogramMapMD.addObject())
                tomogramMapMD.setValue(md.MDL_XCOOR, int(boxPositions[i,0]), 1)
                tomogramMapMD.setValue(md.MDL_YCOOR, int(boxPositions[i,1]), 1)
                tomogramMapMD.setValue(md.MDL_ZCOOR, int(np.random.randint(boxSize // 2, tomoSizeZ - boxSize // 2)), 1)
                tomogramMapMD.write( self._getExtraPath(str(t+1).zfill(5) +"_"+str(i+1).zfill(5) +'_tomogram_map.xmd'))

                # Map each volumes to the right tomogram
                params = " -i " + self._getExtraPath(str(t+1).zfill(5) +'_tomogram.vol')
                params += " -o " + self._getExtraPath(str(t+1).zfill(5) +'_tomogram.vol')
                params += " --geom " + self._getExtraPath(str(t+1).zfill(5) +"_"+str(i+1).zfill(5) +'_tomogram_map.xmd')
                params += " --ref " + self._getExtraPath(str(i + t*particlesPerTomogram +1).zfill(5) + '_deformed.vol')
                params += " --method copy "
                self.runJob('xmipp_tomo_map_back', params)

    def project_volumes(self):

        if self.fullTomogramChoice == FULL_TOMOGRAM_YES:
            # If tomograms are selected, the volumes projected will be tomograms
            sizeX = self.tomoSizeX.get()
            sizeY = self.tomoSizeY.get()
            numberOfVolumes = self.numberOfTomograms.get()
            volumeName = "_tomogram.vol"

        else:
            # else, the deformed volumes are projected
            sizeX = self.volumeSize.get()
            sizeY = self.volumeSize.get()
            numberOfVolumes = self.numberOfVolumes.get()
            volumeName = "_deformed.vol"

        tiltStep = self.tiltStep.get()
        if self.missingWedgeChoice == MISSINGWEDGE_YES:
            tiltLow, tiltHigh = self.tiltLow.get(), self.tiltHigh.get()
        else:
            tiltLow, tiltHigh = -90, 90

        with open(self._getExtraPath('projection.param'), 'a') as file:
            file.write(
                "\n".join([
                    "# XMIPP_STAR_1 *",
                    "# Projection Parameters",
                    "data_noname",
                    "# X and Y projection dimensions [Xdim Ydim]",
                    "_projDimensions '%(sizeX)s %(sizeY)s'" % locals(),
                    "# Angle Set Source -----------------------------------------------------------",
                    "# tilt axis, direction defined by rot and tilt angles in degrees",
                    "_angleRot 90",
                    "_angleTilt 90",
                    "# tilt axis offset in pixels",
                    "_shiftX 0",
                    "_shiftY 0",
                    "_shiftZ 0",
                    "# Tilting description [tilt0 tiltF tiltStep] in degrees",
                    "_projTiltRange '%(tiltLow)s %(tiltHigh)s %(tiltStep)s'" % locals(),
                    "# Noise description ----------------------------------------------------------",
                    "#     applied to angles [noise (bias)]",
                    "_noiseAngles '0 0'",
                    "#     applied to pixels [noise (bias)]",
                    "_noisePixelLevel '0 0'",
                    "#     applied to particle center coordenates [noise (bias)]",
                    "_noiseParticleCoord '0 0'"]))
        for i in range(numberOfVolumes):
            params = " -i " +  self._getExtraPath(str(i + 1).zfill(5) + volumeName)
            params += " --oroot " + self._getExtraPath(str(i + 1).zfill(5) + '_projected')
            params += " --params " + self._getExtraPath('projection.param')
            self.runJob('xmipp_tomo_project', params)

    def apply_noise_and_ctf(self):

        if self.fullTomogramChoice == FULL_TOMOGRAM_YES:
            numberOfVolumes = self.numberOfTomograms.get()

        else:
            numberOfVolumes = self.numberOfVolumes.get()

        with open(self._getExtraPath('ctf.param'), 'a') as file:
            file.write(
                "\n".join([
                    "# XMIPP_STAR_1 *",
                    "data_noname",
                    "_ctfVoltage " + str(self.ctfVoltage.get()),
                    "_ctfSphericalAberration " + str(self.ctfSphericalAberration.get()),
                    "_ctfSamplingRate " + str(self.samplingRate.get()),
                    "_magnification " + str(self.ctfMagnification.get()),
                    "_ctfDefocusU " + str(self.ctfDefocusU.get()),
                    "_ctfDefocusV " + str(self.ctfDefocusV.get()),
                    "_ctfQ0 " + str(self.ctfQ0.get())]))

        for i in range(numberOfVolumes):
            params = " -i " + self._getExtraPath(str(i + 1).zfill(5) + '_projected.sel')
            params += " --ctf " + self._getExtraPath('ctf.param')
            paramsNoiseCTF = params+ " --after_ctf_noise --targetSNR " + str(self.targetSNR.get())
            self.runJob('xmipp_phantom_simulate_microscope', paramsNoiseCTF)

            # the metadata for the i_th stack is self._getExtraPath(str(i + 1).zfill(5) + '_projected.sel')
            MD_i = md.MetaData(self._getExtraPath(str(i + 1).zfill(5) + '_projected.sel'))
            for objId in MD_i:
                img_name = MD_i.getValue(md.MDL_IMAGE, objId)
                params_j = " -i " + img_name + " -o " + img_name
                params_j += " --ctf " + self._getExtraPath('ctf.param')
                self.runJob('xmipp_ctf_phase_flip', params_j)

    def reconstruct(self):
        for i in range(self.numberOfVolumes.get()):
            params = " -i " + self._getExtraPath(str(i + 1).zfill(5) + '_projected.sel')
            params += " -o " + self._getExtraPath(str(i + 1).zfill(5) + '_reconstructed.vol')

            if self.reconstructionChoice == RECONSTRUCTION_FOURIER:
                self.runJob('xmipp_reconstruct_fourier', params)
            elif self.reconstructionChoice == RECONSTRUCTION_WBP:
                self.runJob('xmipp_reconstruct_wbp', params)


    def writeModesMetaData(self):
        """ Iterate over the input SetOfNormalModes and write
        the proper Xmipp metadata.
        Take into account a possible selection of modes (This option is 
        just a shortcut for testing. The recommended
        way is just create a subset from the GUI and use that as input)
        """
        # modeSelection = []
        # if self.modeList.empty():
        #     modeSelection = []
        # else:
        #     modeSelection = getListFromRangeString(self.modeList.get())
        #
        # mdModes = md.MetaData()
        #
        # inputModes = self.inputModes.get()
        # for mode in inputModes:
        #     # If there is a mode selection, only
        #     # take into account those selected
        #     if not modeSelection or mode.getObjId() in modeSelection:
        #         row = XmippMdRow()
        #         modeToRow(mode, row)
        #         row.writeToMd(mdModes, mdModes.addObject())
        # mdModes.write(self.modesFn)
        pass

    def copyDeformationsStep(self, deformationMd):
        pass
        # copyFile(deformationMd, self.imgsFn)
        # # We need to update the image name with the good ones
        # # and the same with the ids.
        # inputSet = self.inputParticles.get()
        # mdImgs = md.MetaData(self.imgsFn)
        # for objId in mdImgs:
        #     imgPath = mdImgs.getValue(md.MDL_IMAGE, objId)
        #     index, fn = xmippToLocation(imgPath)
        #     # Conside the index is the id in the input set
        #     particle = inputSet[index]
        #     mdImgs.setValue(md.MDL_IMAGE, getImageLocation(particle), objId)
        #     mdImgs.setValue(md.MDL_ITEM_ID, int(particle.getObjId()), objId)
        # mdImgs.write(self.imgsFn)

    def performNmaStep(self, atomsFn, modesFn):
        pass
        # sampling = self.inputParticles.get().getSamplingRate()
        # discreteAngularSampling = self.discreteAngularSampling.get()
        # trustRegionScale = self.trustRegionScale.get()
        # odir = self._getTmpPath()
        # imgFn = self.imgsFn
        #
        # args = "-i %(imgFn)s --pdb %(atomsFn)s --modes %(modesFn)s --sampling_rate %(sampling)f "
        # args += "--discrAngStep %(discreteAngularSampling)f --odir %(odir)s --centerPDB "
        # args += "--trustradius_scale %(trustRegionScale)d --resume "
        #
        # if self.getInputPdb().getPseudoAtoms():
        #     args += "--fixed_Gaussian "
        #
        # if self.alignmentMethod == NMA_ALIGNMENT_PROJ:
        #     args += "--projMatch "
        #
        # self.runJob("xmipp_nma_alignment", args % locals())
        #
        # cleanPath(self._getPath('nmaTodo.xmd'))
        #
        # inputSet = self.inputParticles.get()
        # mdImgs = md.MetaData(self.imgsFn)
        # for objId in mdImgs:
        #     imgPath = mdImgs.getValue(md.MDL_IMAGE, objId)
        #     index, fn = xmippToLocation(imgPath)
        #     # Conside the index is the id in the input set
        #     particle = inputSet[index]
        #     mdImgs.setValue(md.MDL_IMAGE, getImageLocation(particle), objId)
        #     mdImgs.setValue(md.MDL_ITEM_ID, int(particle.getObjId()), objId)
        # mdImgs.write(self.imgsFn)

    def createOutputStep(self):
        # first making a metadata for only the subtomograms:
        out_mdfn = self._getExtraPath('subtomograms.xmd')
        pattern = '"' + self._getExtraPath() + '/*_reconstructed.vol"'
        command = '-p ' + pattern + ' -o ' + out_mdfn
        self.runJob('xmipp_metadata_selfile_create', command)
        # now creating the output set of volumes as output:
        partSet = self._createSetOfVolumes('subtomograms')
        xmipp3.convert.readSetOfVolumes(out_mdfn, partSet)
        partSet.setSamplingRate(self.samplingRate.get())
        self._defineOutputs(outputVolumes=partSet)
        # volume = Volume(self._getExtraPath(str(1).zfill(5) +'_reconstructed.vol'))
        # self._defineOutputs(outputVolume=volume)

    # --------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        return summary

    def _validate(self):
        errors = []
        # xdim = self.inputParticles.get().getDim()[0]
        # if not isPower2(xdim):
        #     errors.append("Image dimension (%s) is not a power of two, consider resize them" % xdim)
        return errors

    def _citations(self):
        return ['Jonic2005', 'Sorzano2004b', 'Jin2014']

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

    def _getLocalModesFn(self):
        modesFn = self.inputModes.get().getFileName()
        return self._getBasePath(modesFn)

    # def _updateParticle(self, item, row):
    #     setXmippAttributes(item, row, md.MDL_ANGLE_ROT, md.MDL_ANGLE_TILT, md.MDL_ANGLE_PSI, md.MDL_SHIFT_X,
    #                        md.MDL_SHIFT_Y, md.MDL_FLIP, md.MDL_NMA, md.MDL_COST)
    #     createItemMatrix(item, row, align=em.ALIGN_PROJ)
