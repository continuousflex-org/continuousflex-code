# **************************************************************************
# * Authors:     J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
# *              Slavica Jonic  (slavica.jonic@upmc.fr)
# *              Mohamad Harastani (mohamad.harastani@upmc.fr)
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
from continuousflex.protocols.data import PathData

"""
This module implement the wrappers around Xmipp CL2D protocol
visualization program.
"""

from os.path import basename, join, exists
import numpy as np

from pwem.convert.atom_struct import cifToPdb
from pyworkflow.utils import replaceBaseExt

from pyworkflow.utils.path import cleanPath, makePath, cleanPattern
from pyworkflow.viewer import (ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO)
from pyworkflow.protocol.params import StringParam, LabelParam
from pwem.objects import SetOfParticles
from pwem.viewers import VmdView
from pyworkflow.gui.browser import FileBrowserWindow

from continuousflex.protocols.protocol_nma_dimred import FlexProtDimredNMA

from continuousflex.protocols.data import Point, Data

from continuousflex.viewers.nma_plotter import FlexNmaPlotter

from continuousflex.viewers.nma_gui import ClusteringWindow, TrajectoriesWindow
from pwem.utils import runProgram
from pyworkflow.protocol import params

FIGURE_LIMIT_NONE = 0
FIGURE_LIMITS = 1

X_LIMITS_NONE = 0
X_LIMITS = 1
Y_LIMITS_NONE = 0
Y_LIMITS = 1
Z_LIMITS_NONE = 0
Z_LIMITS = 1

POINT_LIMITS_NONE = 0
POINT_LIMITS = 1

class FlexDimredNMAViewer(ProtocolViewer):
    """ Visualization of results from the NMA protocol
    """
    _label = 'viewer nma dimred'
    _targets = [FlexProtDimredNMA]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def __init__(self, **kwargs):
        ProtocolViewer.__init__(self, **kwargs)
        self._data = None

    def getData(self):
        if self._data is None:
            self._data = self.loadData()
        return self._data

    def _defineParams(self, form):
        form.addSection(label='Visualization')
        form.addParam('displayRawDeformation', StringParam, default='1 2',
                      label='Display normal-mode amplitudes in the low-dimensional space',
                      help='Type 1 to see the histogram of normal-mode amplitudes in the low-dimensional space, '
                           'using axis 1; \n '
                           'Type 2 to see the histogram of normal-mode amplitudes in the low-dimensional space, '
                           'using axis 2; etc. \n '
                           'Type 1 2 to see normal-mode amplitudes in the low-dimensional space, using axes 1 and 2; \n'
                           'Type 1 2 3 to see normal-mode amplitudes in the low-dimensional space, using axes 1, 2, '
                           'and 3; etc. '
                      )

        form.addParam('displayClustering', LabelParam,
                      label='Open clustering tool?',
                      help='Open a GUI to visualize the images as points '
                           'and select some of them to create clusters, and compute the 3D reconstructions from the '
                           'clusters.')

        form.addParam('displayTrajectories', LabelParam,
                      label='Open trajectories tool?',
                      help='Open a GUI to visualize the images as points, '
                           'draw and adjust trajectories, and animate them.')

        form.addParam('limits_modes', params.EnumParam,
                      choices=['Automatic (Recommended)', 'Set manually Use upper and lower values'],
                      default=FIGURE_LIMIT_NONE,
                      label='Error limits', display=params.EnumParam.DISPLAY_COMBO,
                      help='If you want to use a range of Error in the color bar choose to set it manually.')
        form.addParam('LimitLow', params.FloatParam, default=None,
                      condition='limits_modes==%d' % FIGURE_LIMITS,
                      label='Lower Error value',
                      help='The lower Error used in the graph')
        form.addParam('LimitHigh', params.FloatParam, default=None,
                      condition='limits_modes==%d' % FIGURE_LIMITS,
                      label='Upper Error value',
                      help='The upper Error used in the graph')
        form.addParam('xlimits_mode', params.EnumParam,
                      choices=['Automatic (Recommended)', 'Set manually x-axis limits'],
                      default=X_LIMITS_NONE,
                      label='x-axis limits', display=params.EnumParam.DISPLAY_COMBO,
                      help='This allows you to use a specific range of x-axis limits')
        form.addParam('xlim_low', params.FloatParam, default=None,
                      condition='xlimits_mode==%d' % X_LIMITS,
                      label='Lower x-axis limit')
        form.addParam('xlim_high', params.FloatParam, default=None,
                      condition='xlimits_mode==%d' % X_LIMITS,
                      label='Upper x-axis limit')
        form.addParam('ylimits_mode', params.EnumParam,
                      choices=['Automatic (Recommended)', 'Set manually y-axis limits'],
                      default=Y_LIMITS_NONE,
                      label='y-axis limits', display=params.EnumParam.DISPLAY_COMBO,
                      help='This allows you to use a specific range of y-axis limits')
        form.addParam('ylim_low', params.FloatParam, default=None,
                      condition='ylimits_mode==%d' % Y_LIMITS,
                      label='Lower y-axis limit')
        form.addParam('ylim_high', params.FloatParam, default=None,
                      condition='ylimits_mode==%d' % Y_LIMITS,
                      label='Upper y-axis limit')
        form.addParam('zlimits_mode', params.EnumParam,
                      choices=['Automatic (Recommended)', 'Set manually z-axis limits'],
                      default=Z_LIMITS_NONE,
                      label='z-axis limits', display=params.EnumParam.DISPLAY_COMBO,
                      help='This allows you to use a specific range of z-axis limits')
        form.addParam('zlim_low', params.FloatParam, default=None,
                      condition='zlimits_mode==%d' % Z_LIMITS,
                      label='Lower z-axis limit')
        form.addParam('zlim_high', params.FloatParam, default=None,
                      condition='zlimits_mode==%d' % Z_LIMITS,
                      label='Upper z-axis limit')
        # Scatter points size and transparancy
        form.addParam('points_shades', params.EnumParam,
                      choices=['Automatic (Recommended)', 'Set manually point radius and transparancy'],
                      default=POINT_LIMITS_NONE,
                      label='Scatter points radius and transparancy', display=params.EnumParam.DISPLAY_COMBO,
                      help='This allows you to use change the points radius and transparancy in the scatter plot'
                           '. By trying different values, it may help you discover the densest regions in the space.')
        line = form.addLine('Radius and transparancy',
                            condition='points_shades==%d' % POINT_LIMITS,
                            help='Values for points rarius have can be any positive real number.'
                                 ' Values for transparancy are between 0 and 1.')
        line.addParam('s', params.FloatParam, default=None, allowsNull=True,
                      label='Radius')
        line.addParam('alpha', params.FloatParam, default=None, allowsNull=True,
                        label='Transparancy')

    def _getVisualizeDict(self):
        return {'displayRawDeformation': self._viewRawDeformation,
                'displayClustering': self._displayClustering,
                'displayTrajectories': self._displayTrajectories,
                }

    def _viewRawDeformation(self, paramName):
        components = self.displayRawDeformation.get()
        return self._doViewRawDeformation(components)

    def _doViewRawDeformation(self, components):
        components = list(map(int, components.split()))
        dim = len(components)
        views = []

        if dim > 0:
            modeList = [m - 1 for m in components]
            modeNameList = ['Axis %d' % m for m in components]
            missingList = []

            if missingList:
                return [self.errorMessage("Invalid mode(s) *%s*\n." % (', '.join(missingList)),
                                          title="Invalid input")]

            # Actually plot
            if self.limits_modes == FIGURE_LIMIT_NONE:
                plotter = FlexNmaPlotter(data=self.getData(),
                                            xlim_low=self.xlim_low, xlim_high=self.xlim_high,
                                            ylim_low=self.ylim_low, ylim_high=self.ylim_high,
                                            zlim_low=self.zlim_low, zlim_high=self.zlim_high,
                                            s=self.s, alpha=self.alpha)
            else:
                plotter = FlexNmaPlotter(data=self.getData(),
                                            LimitL=self.LimitLow, LimitH=self.LimitHigh,
                                            xlim_low=self.xlim_low, xlim_high=self.xlim_high,
                                            ylim_low=self.ylim_low, ylim_high=self.ylim_high,
                                            zlim_low=self.zlim_low, zlim_high=self.zlim_high,
                                            s=self.s, alpha=self.alpha)
            baseList = [basename(n) for n in modeNameList]

            self.getData().XIND = modeList[0]
            if dim == 1:
                plotter.plotArray1D("Histogram of normal-mode amplitudes in low-dimensional space: %s" % baseList[0],
                                    "Amplitude", "Number of images")
            else:
                self.getData().YIND = modeList[1]
                if dim == 2:
                    plotter.plotArray2D("Normal-mode amplitudes in low-dimensional space: %s vs %s" % tuple(baseList),
                                        *baseList)
                elif dim == 3:
                    self.getData().ZIND = modeList[2]
                    plotter.plotArray3D("Normal-mode amplitudes in low-dimensional space: %s %s %s" % tuple(baseList),
                                        *baseList)
            views.append(plotter)

        return views

    def _displayClustering(self, paramName):
        self.clusterWindow = self.tkWindow(ClusteringWindow,
                                           title='Clustering Tool',
                                           dim=self.protocol.reducedDim.get(),
                                           data=self.getData(),
                                           callback=self._createCluster,
                                           limits_mode=self.limits_modes,
                                           LimitL=self.LimitLow,
                                           LimitH=self.LimitHigh,
                                           xlim_low=self.xlim_low,
                                           xlim_high=self.xlim_high,
                                           ylim_low=self.ylim_low,
                                           ylim_high=self.ylim_high,
                                           zlim_low=self.zlim_low,
                                           zlim_high=self.zlim_high,
                                           s=self.s,
                                           alpha=self.alpha)
        return [self.clusterWindow]

    def _displayTrajectories(self, paramName):
        self.trajectoriesWindow = self.tkWindow(TrajectoriesWindow,
                                                title='Trajectories Tool',
                                                dim=self.protocol.reducedDim.get(),
                                                data=self.getData(),
                                                callback=self._generateAnimation,
                                                loadCallback=self._loadAnimation,
                                                numberOfPoints=10,
                                                limits_mode=self.limits_modes,
                                                LimitL=self.LimitLow,
                                                LimitH=self.LimitHigh,
                                                xlim_low=self.xlim_low,
                                                xlim_high=self.xlim_high,
                                                ylim_low=self.ylim_low,
                                                ylim_high=self.ylim_high,
                                                zlim_low=self.zlim_low,
                                                zlim_high=self.zlim_high,
                                                s=self.s,
                                                alpha=self.alpha)
        return [self.trajectoriesWindow]

    def _createCluster(self):
        """ Create the cluster with the selected particles
        from the cluster. This method will be called when
        the button 'Create Cluster' is pressed.
        """
        # Write the particles
        prot = self.protocol
        project = prot.getProject()
        inputSet = prot.getInputParticles()
        makePath(prot._getTmpPath())
        fnSqlite = prot._getTmpPath('cluster_particles.sqlite')
        cleanPath(fnSqlite)
        partSet = SetOfParticles(filename=fnSqlite)
        partSet.copyInfo(inputSet)
        for point in self.getData():
            if point.getState() == Point.SELECTED:
                particle = inputSet[point.getId()]
                partSet.append(particle)
        partSet.write()
        partSet.close()

        from continuousflex.protocols.protocol_batch_cluster import FlexBatchProtNMACluster
        # from xmipp3.protocols.nma.protocol_batch_cluster import BatchProtNMACluster
        newProt = project.newProtocol(FlexBatchProtNMACluster)
        clusterName = self.clusterWindow.getClusterName()
        if clusterName:
            newProt.setObjLabel(clusterName)
        newProt.inputNmaDimred.set(prot)
        newProt.sqliteFile.set(fnSqlite)

        project.launchProtocol(newProt)
        project.getRunsGraph()

    def _loadAnimationData(self, obj):
        prot = self.protocol
        animationName = obj.getFileName()  # assumes that obj.getFileName is the folder of animation
        animationPath = prot._getExtraPath(animationName)

        animationFiles = [animationName + '.vmd', animationName + '.pdb', 'trajectory.txt']
        for s in animationFiles:
            f = join(animationPath, s)
            if not exists(f):
                self.errorMessage('Animation file "%s" not found. ' % f)
                return

        # Load animation trajectory points
        trajectoryPoints = np.loadtxt(join(animationPath, 'trajectory.txt'))
        data = PathData(dim=trajectoryPoints.shape[1])

        for i, row in enumerate(trajectoryPoints):
            data.addPoint(Point(pointId=i + 1, data=list(row), weight=1))

        self.trajectoriesWindow.setPathData(data)
        self.trajectoriesWindow.setAnimationName(animationName)
        self.trajectoriesWindow._onUpdateClick()

        def _showVmd():
            vmdFn = join(animationPath, animationName + '.vmd')
            VmdView(' -e %s' % vmdFn).show()

        self.getTkRoot().after(500, _showVmd)

    def _loadAnimation(self):
        prot = self.protocol
        browser = FileBrowserWindow("Select the animation folder (animation_NAME)",
                                    self.getWindow(), prot._getExtraPath(),
                                    onSelect=self._loadAnimationData)
        browser.show()

    def _generateAnimation(self):
        prot = self.protocol
        projectorFile = prot.getProjectorFile()

        animation = self.trajectoriesWindow.getAnimationName()
        animationPath = prot._getExtraPath('animation_%s' % animation)

        cleanPath(animationPath)
        makePath(animationPath)
        animationRoot = join(animationPath, 'animation_%s' % animation)

        trajectoryPoints = np.array([p.getData() for p in self.trajectoriesWindow.pathData])
        np.savetxt(join(animationPath, 'trajectory.txt'), trajectoryPoints)

        if projectorFile:
            M = np.loadtxt(projectorFile)
            deformations = np.dot(trajectoryPoints, np.linalg.pinv(M))
        else:
            Y = np.loadtxt(prot.getOutputMatrixFile())
            X = np.loadtxt(prot.getDeformationFile())
            # Find closest points in deformations
            deformations = [X[np.argmin(np.sum((Y - p) ** 2, axis=1))] for p in trajectoryPoints]

        pdb = prot.getInputPdb()
        pdbFile = pdb.getFileName()

        structureEM = prot.getInputPdb().getPseudoAtoms()
        if not structureEM:
            localFn = replaceBaseExt(basename(pdbFile), 'pdb')
            cifToPdb(pdbFile, localFn)
            pdbFile = basename(localFn)

        modesFn = prot.inputNMA.get()._getExtraPath('modes.xmd')

        for i, d in enumerate(deformations):
            atomsFn = animationRoot + 'atomsDeformed_%02d.pdb' % (i + 1)
            cmd = '-o %s --pdb %s --nma %s --deformations %s' % (atomsFn, pdbFile, modesFn, str(d)[1:-1])
            runProgram('xmipp_pdb_nma_deform', cmd)

        # Join all deformations in a single pdb
        # iterating going up and down through all points
        # 1 2 3 ... n-2 n-1 n n-1 n-2 ... 3, 2
        n = len(deformations)
        r1 = list(range(1, n + 1))
        r2 = list(range(2, n))  # Skip 1 at the end
        r2.reverse()
        loop = r1 + r2

        trajFn = animationRoot + '.pdb'
        trajFile = open(trajFn, 'w')

        for i in loop:
            atomsFn = animationRoot + 'atomsDeformed_%02d.pdb' % i
            atomsFile = open(atomsFn)
            for line in atomsFile:
                trajFile.write(line)
            trajFile.write('TER\nENDMDL\n')
            atomsFile.close()

        trajFile.close()
        # Delete temporary atom files
        cleanPattern(animationRoot + 'atomsDeformed_??.pdb')

        # Generate the vmd script
        vmdFn = animationRoot + '.vmd'
        vmdFile = open(vmdFn, 'w')
        vmdFile.write("""
        mol new %s
        animate style Loop
        display projection Orthographic
        mol modcolor 0 0 Index
        mol modstyle 0 0 Beads 1.000000 8.000000
        animate speed 0.5
        animate forward
        """ % trajFn)
        vmdFile.close()

        VmdView(' -e ' + vmdFn).show()

    def loadData(self):
        """ Iterate over the images and the output matrix txt file
        and create a Data object with theirs Points.
        """
        matrix = np.loadtxt(self.protocol.getOutputMatrixFile())
        particles = self.protocol.getInputParticles()

        data = Data()
        for i, particle in enumerate(particles):
            data.addPoint(Point(pointId=particle.getObjId(),
                                data=matrix[i, :],
                                weight=particle._xmipp_cost.get()))

        return data
