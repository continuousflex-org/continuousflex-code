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
from pyworkflow.protocol.params import StringParam, LabelParam
from pyworkflow.viewer import (ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO)
from pyworkflow.utils import replaceBaseExt, replaceExt

from continuousflex.protocols.data import Point, Data
from continuousflex.viewers.nma_plotter import FlexNmaPlotter
from continuousflex.protocols import FlexProtDimredPdb
import xmipp3
import pwem.emlib.metadata as md
from pyworkflow.utils.process import runJob
from pwem.viewers import ObjectView
import matplotlib.pyplot as plt

class FlexProtPdbDimredViewer(ProtocolViewer):
    """ Visualization of dimensionality reduction on PDBs
    """
    _label = 'viewer PDBs dimred'
    _targets = [FlexProtDimredPdb]
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
                      label='Display the principle axes',
                      help='Type 1 to see the histogram of PCA axis 1; \n'
                           'type 2 to to see the histogram of PCA axis 2, etc.\n'
                           'Type 1 2 to see the 2D plot of amplitudes for PCA axes 1 2.\n'
                           'Type 1 2 3 to see the 3D plot of amplitudes for PCA axes 1 2 3; etc.'
                           )

    def _getVisualizeDict(self):
        return {'displayRawDeformation': self._viewRawDeformation}

    def _viewRawDeformation(self, paramName):
        components = self.displayRawDeformation.get()
        return self._doViewRawDeformation(components)
        
    def _doViewRawDeformation(self, components):
        components = list(map(int, components.split()))
        # print(components)
        dim = len(components)
        views = []
        print(self.protocol.getOutputMatrixFile())
        X = np.loadtxt(fname=self.protocol.getOutputMatrixFile())
        if dim == 1:
            plt.hist(X[:,components[0]-1])
        if dim == 2:
            plt.scatter(X[:,components[0]-1],X[:,components[1]-1])
        if dim == 3:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.scatter(X[:,components[0]-1],X[:,components[1]-1],X[:,components[2]-1])
        plt.show()



        
