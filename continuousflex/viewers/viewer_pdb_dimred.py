# **************************************************************************
# * Authors:  Mohamad Harastani          (mohamad.harastani@upmc.fr)
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
# **************************************************************************


from os.path import basename
import numpy as np
from pwem.emlib import MetaData, MDL_ORDER
from pyworkflow.protocol.params import StringParam, LabelParam, EnumParam, FloatParam, PointerParam
from pyworkflow.viewer import (ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO)
from pyworkflow.utils import replaceBaseExt, replaceExt
from pwem.viewers import ChimeraView
from pyworkflow.viewer import Viewer


from pwem.objects.data import SetOfParticles,SetOfVolumes, Class2D, ClassVol
from continuousflex.viewers.nma_plotter import FlexNmaPlotter
from continuousflex.protocols import FlexProtDimredPdb
import xmipp3
from xmipp3.convert import writeSetOfVolumes, writeSetOfParticles, readSetOfVolumes, readSetOfParticles
import pwem.emlib.metadata as md
from pwem.viewers import ObjectView
import matplotlib.pyplot as plt
from pwem.emlib.image import ImageHandler

from joblib import load
from continuousflex.viewers.tk_dimred import ClusteringWindowDimred, TrajectoriesWindowDimred
from continuousflex.protocols.data import Point, Data, PathData
from pwem.viewers import VmdView
from pyworkflow.utils.path import cleanPath, makePath
from continuousflex.protocols.utilities.genesis_utilities import save_dcd
from continuousflex.protocols.utilities.pdb_handler import ContinuousFlexPDBHandler
from pyworkflow.gui.browser import FileBrowserWindow
from continuousflex.protocols.protocol_pdb_dimred import REDUCE_METHOD_PCA, REDUCE_METHOD_UMAP


import os

X_LIMITS_NONE = 0
X_LIMITS = 1
Y_LIMITS_NONE = 0
Y_LIMITS = 1
Z_LIMITS_NONE = 0
Z_LIMITS = 1

NUM_POINTS_TRAJECTORY=10


class FlexProtPdbDimredViewer(ProtocolViewer):
    """ Visualization of dimensionality reduction on PDBs
    """
    _label = 'viewer PDBs dimred'
    _targets = [FlexProtDimredPdb]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def __init__(self, **kwargs):
        ProtocolViewer.__init__(self, **kwargs)
        self._data = None

    def _defineParams(self, form):
        form.addSection(label='Visualization')
        form.addParam('displayTrajectories', LabelParam,
                      label='Open trajectories tool ?',
                      help='Open a GUI to visualize the PCA space'
                           ' to draw and adjust trajectories.')
        form.addParam('displayClustering', LabelParam,
                      label='Open clustering tool?',
                      help='Open a GUI to visualize the images as points '
                           'and select some of them to create clusters, and compute the 3D reconstructions from the '
                           'clusters.')

        form.addParam('inputSet', PointerParam, pointerClass ='SetOfParticles,SetOfVolumes',
                      label='Em data for cluster animation',  allowsNull=True,
                      help="")

        # form.addParam("dataSet", StringParam, default= "", label="Data set label")
        form.addParam('displayPcaSingularValues', LabelParam,
                      label="Display singular values",
                      help="The values should help you see how many dimensions are in the data ",
                      condition=self.protocol.method.get()==REDUCE_METHOD_PCA)


        group = form.addGroup("Window parameters")
        group.addParam('s', FloatParam, default=5, allowsNull=True,
                       label='Radius')
        group.addParam('alpha', FloatParam, default=0.5, allowsNull=True,
                       label='Transparancy')
        group.addParam('xlimits_mode', EnumParam,
                      choices=['Automatic (Recommended)', 'Set manually x-axis limits'],
                      default=X_LIMITS_NONE,
                      label='x-axis limits', display=EnumParam.DISPLAY_COMBO,
                      help='This allows you to use a specific range of x-axis limits')
        group.addParam('xlim_low', FloatParam, default=None,
                      condition='xlimits_mode==%d' % X_LIMITS,
                      label='Lower x-axis limit')
        group.addParam('xlim_high', FloatParam, default=None,
                      condition='xlimits_mode==%d' % X_LIMITS,
                      label='Upper x-axis limit')
        group.addParam('ylimits_mode', EnumParam,
                      choices=['Automatic (Recommended)', 'Set manually y-axis limits'],
                      default=Y_LIMITS_NONE,
                      label='y-axis limits', display=EnumParam.DISPLAY_COMBO,
                      help='This allows you to use a specific range of y-axis limits')
        group.addParam('ylim_low', FloatParam, default=None,
                      condition='ylimits_mode==%d' % Y_LIMITS,
                      label='Lower y-axis limit')
        group.addParam('ylim_high', FloatParam, default=None,
                      condition='ylimits_mode==%d' % Y_LIMITS,
                      label='Upper y-axis limit')
        group.addParam('zlimits_mode', EnumParam,
                      choices=['Automatic (Recommended)', 'Set manually z-axis limits'],
                      default=Z_LIMITS_NONE,
                      label='z-axis limits', display=EnumParam.DISPLAY_COMBO,
                      help='This allows you to use a specific range of z-axis limits')
        group.addParam('zlim_low', FloatParam, default=None,
                      condition='zlimits_mode==%d' % Z_LIMITS,
                      label='Lower z-axis limit')
        group.addParam('zlim_high', FloatParam, default=None,
                      condition='zlimits_mode==%d' % Z_LIMITS,
                      label='Upper z-axis limit')


    def _getVisualizeDict(self):
        return {
                'displayTrajectories': self._displayTrajectories,
                'displayClustering': self._displayClustering,
                'displayPcaSingularValues': self.viewPcaSinglularValues,
                }


    def _displayTrajectories(self, paramName):
        self.trajectoriesWindow = self.tkWindow(TrajectoriesWindowDimred,
                                                title='Trajectories Tool',
                                                dim=self.protocol.reducedDim.get(),
                                                data=self.getData(),
                                                callback=self._generateAnimation,
                                                loadCallback=self._loadAnimation,
                                                saveClusterCallback=self.saveClusterCallback,
                                                numberOfPoints=NUM_POINTS_TRAJECTORY,
                                                limits_mode=0,
                                                LimitL=None,
                                                LimitH=None,
                                                xlim_low=self.xlim_low.get(),
                                                xlim_high=self.xlim_high.get(),
                                                ylim_low=self.ylim_low.get(),
                                                ylim_high=self.ylim_high.get(),
                                                zlim_low=self.zlim_low.get(),
                                                zlim_high=self.zlim_high.get(),
                                                s=self.s,
                                                alpha=self.alpha)
        return [self.trajectoriesWindow]

    def _displayClustering(self, paramName):
        index = 1
        while(os.path.exists(self.protocol._getExtraPath("%s_cluster.xmd"%index))):
            cleanPath(self.protocol._getExtraPath("%s_cluster.xmd"%index))
            index+=1
        self.clusterWindow = self.tkWindow(ClusteringWindowDimred,
                                           title='Clustering Tool',
                                           dim=self.protocol.reducedDim.get(),
                                           data=self.getData(),
                                           callback=self._createCluster,
                                           limits_mode=0,
                                           LimitL=0.0,
                                           LimitH=1.0,
                                           xlim_low=self.xlim_low.get(),
                                           xlim_high=self.xlim_high.get(),
                                           ylim_low=self.ylim_low.get(),
                                           ylim_high=self.ylim_high.get(),
                                           zlim_low=self.zlim_low.get(),
                                           zlim_high=self.zlim_high.get(),
                                           s=self.s,
                                           alpha=self.alpha)
        return [self.clusterWindow]


    def viewPcaSinglularValues(self, paramName):
        pca = load(self.protocol._getExtraPath('pca_pickled.joblib'))
        fig = plt.figure('PCA singlular values')
        plt.stem(pca.singular_values_)
        plt.xticks(np.arange(0, len(pca.singular_values_), 1))
        plt.show()
        pass

    def getData(self):
        if self._data is None:
            self._data = self.loadData()
        return self._data

    def loadData(self):
        data = Data()
        pdb_matrix = np.loadtxt(self.protocol.getOutputMatrixFile())

        # dataSet = self.dataSet.get().split(";")
        # n_data = len(dataSet)
        # if n_data >1:
        #     weights = []
        #     for i in range(n_data):
        #         if dataSet[i] != '':
        #             for j in range(int(dataSet[i])):
        #                 weights.append(i/n_data)
        #
        # else:
        #
        weights = [0.0 for i in range(pdb_matrix.shape[0])]

        for i in range(pdb_matrix.shape[0]):
            data.addPoint(Point(pointId=i+1, data=pdb_matrix[i, :],weight=weights[i]))
        return data

    def _generateAnimation(self):
        prot = self.protocol

        # Get animation root
        animation = self.trajectoriesWindow.getAnimationName()
        animationPath = prot._getExtraPath('animation_%s' % animation)
        cleanPath(animationPath)
        makePath(animationPath)
        animationRoot = os.path.join(animationPath, 'animation_%s' % animation)

        # get trajectory coordinates
        trajectoryPoints = np.array([p.getData() for p in self.trajectoriesWindow.pathData])
        np.savetxt(animationRoot + 'trajectory.txt', trajectoryPoints)
        pca = load(prot._getExtraPath('pca_pickled.joblib'))
        deformations = pca.inverse_transform(trajectoryPoints)

        # Generate DCD trajectory
        initPDB = ContinuousFlexPDBHandler(prot.getPDBRef())
        initdcdcp = initPDB.copy()
        coords_list = []
        for i in range(NUM_POINTS_TRAJECTORY):
            coords_list.append(deformations[i].reshape((initdcdcp.n_atoms, 3)))
        save_dcd(mol=initdcdcp, coords_list=coords_list, prefix=animationRoot)
        initdcdcp.coords = coords_list[0]
        initdcdcp.write_pdb(animationRoot+".pdb")

        # Generate the vmd script
        vmdFn = animationRoot + '.vmd'
        vmdFile = open(vmdFn, 'w')
        vmdFile.write("""
        mol load pdb %s.pdb dcd %s.dcd
        animate style Rock
        display projection Orthographic
        mol modcolor 0 0 Index
        mol modstyle 0 0 Tube 1.000000 8.000000
        animate speed 1.0
        animate forward
        """ % (animationRoot,animationRoot))
        vmdFile.close()

        VmdView(' -e ' + vmdFn).show()

    def _createCluster(self):
        """ Create the cluster with the selected particles
        from the cluster. This method will be called when
        the button 'Create Cluster' is pressed.
        """

        # define metadata
        cluster = md.MetaData()
        for point in self.getData():
            if point.getState() == Point.SELECTED:
                cluster.setValue(md.MDL_ITEM_ID, int(point.getId()), cluster.addObject())
                point._weight = 0.5

        # get name
        cluster_name = self.clusterWindow.getClusterName()
        if cluster_name == "":
            index = 1
            while(os.path.exists(self.protocol._getExtraPath("%s_cluster.xmd"%index))):
                index+=1
            cluster_name = self.protocol._getExtraPath("%s_cluster.xmd"%index)

        # write metadata
        print("Write cluster to %s "%cluster_name)
        cluster.write(cluster_name)


    def saveClusterCallback(self):
        # get cluster name
        clusterName = "cluster_" + self.trajectoriesWindow.getClusterName()

        # get input metadata
        inputSet = self.inputSet.get()

        classID=[]
        for p in self.trajectoriesWindow.data:
            classID.append(p._weight)

        if isinstance(inputSet, SetOfParticles):
            classSet = self.protocol._createSetOfClasses2D(inputSet, clusterName)
        else:
            classSet = self.protocol._createSetOfClasses3D(inputSet,clusterName)

        classSet.classifyItems(
            updateItemCallback=updateItemCallback,
            updateClassCallback=None,
            itemDataIterator=iter(itemDataIterator(classID)),
            classifyDisabled=False,
            iterParams=None,
            doClone=True)

        # Run reconstruction
        self.protocol._defineOutputs(**{clusterName : classSet})
        from continuousflex.protocols.protocol_batch_cluster import FlexBatchProtClusterSet
        project = self.protocol.getProject()
        newProt = project.newProtocol(FlexBatchProtClusterSet)
        newProt.setObjLabel(clusterName)
        newProt.inputSet.set(getattr(self.protocol, clusterName))
        project.launchProtocol(newProt)
        project.getRunsGraph()

    def _loadAnimation(self):
        browser = FileBrowserWindow("Select the animation folder (animation_NAME)",
                                    self.getWindow(), self.protocol._getExtraPath(),
                                    onSelect=self._loadAnimationData)
        browser.show()

    def _loadAnimationData(self, obj):
        prot = self.protocol
        animationName = obj.getFileName()  # assumes that obj.getFileName is the folder of animation
        animationPath = prot._getExtraPath(animationName)
        animationRoot = os.path.join(animationPath, animationName)

        animationSuffixes = ['.vmd', '.pdb','.dcd', 'trajectory.txt']
        for s in animationSuffixes:
            f = animationRoot + s
            if not os.path.exists(f):
                self.errorMessage('Animation file "%s" not found. ' % f)
                return

        # Load animation trajectory points
        trajectoryPoints = np.loadtxt(animationRoot + 'trajectory.txt')
        data = PathData(dim=trajectoryPoints.shape[1])

        for i, row in enumerate(trajectoryPoints):
            data.addPoint(Point(pointId=i + 1, data=list(row), weight=1))

        self.trajectoriesWindow.setPathData(data)
        self.trajectoriesWindow.setAnimationName(animationName)
        self.trajectoriesWindow._onUpdateClick()

        def _showVmd():
            vmdFn = animationRoot + '.vmd'
            VmdView(' -e %s' % vmdFn).show()

        self.getTkRoot().after(500, _showVmd)

class VolumeTrajectoryViewer(ProtocolViewer):
    """ Visualization of a SetOfVolumes as a trajectory with ChimeraX
    """
    _label = 'Volume trajectory viewer'
    _targets = [SetOfVolumes]

    def _defineParams(self, form):
        form.addSection(label='Visualization')
        form.addParam('displayTrajectories', LabelParam,
                      label='ChimeraX',
                      help='Open the trajectory in ChimeraX.')
    def _getVisualizeDict(self):
        return {
                'displayTrajectories': self._visualize,
                }

    def _visualize(self, obj, **kwargs):
        """visualisation for volumes set"""
        for i in self.protocol:
            i.setSamplingRate(self.protocol.getSamplingRate())
            vol = ImageHandler().read(i)
            vol.write(self._getPath("VolumeTrajectoryViewer%i.vol"%i.getObjId()))
        # Show Chimera
        tmpChimeraFile = self._getPath("chimera.cxc")
        print(tmpChimeraFile)
        with open(tmpChimeraFile, "w") as f:
            f.write("open %s vseries true \n" % os.path.abspath(self._getPath("VolumeTrajectoryViewer*.vol")))
            # f.write("volume #1 style surface level 0.5")
            f.write("vseries play #1 loop true maxFrameRate 7 direction oscillate \n")

        cv = ChimeraView(tmpChimeraFile)
        return [cv]


def updateItemCallback(item, row):
    item.setClassId(row)

class itemDataIterator:
    def __init__(self, classID):
        self.classID = classID
    def __iter__(self):
        self.n = 0
        return self
    def __next__(self):
        index = self.classID[self.n]
        self.n += 1
        return index
