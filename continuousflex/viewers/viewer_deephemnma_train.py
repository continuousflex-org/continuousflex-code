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
"""
This module implement the wrappers aroung Xmipp CL2D protocol
visualization program.
"""
from continuousflex.protocols.protocol_deep_hemnma_train import FlexProtDeepHEMNMATrain
from pwem.viewers import EmProtocolViewer
from pyworkflow.protocol.params import LabelParam, IntParam, EnumParam, StringParam
from pyworkflow.viewer import ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO
import numpy as np
from subprocess import check_call
import sys



class FlexDeepHEMNMAViewer(EmProtocolViewer):
    """ Visualization of results from the deepHEMNMA protocol
    """
    _label = 'viewer deepHEMNMA'
    _targets = [FlexProtDeepHEMNMATrain]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def __init__(self, **kwargs):
        ProtocolViewer.__init__(self, **kwargs)
        self._data = None

    def _defineParams(self, form):
        form.addSection(label='Visualization')
        form.addParam('displaycurves', LabelParam,
                      label="Display training curves",
                      help="Display the training and validation losses")


    def _getVisualizeDict(self):
        return {'displaycures': self._viewcurves}

    def _viewcurves(self):
        logdir = self.self.protocol._getExtraPath('scalars/')
        command = "tensorboard --logidr " + logdir
        check_call(command, shell=True, stdout=sys.stdout, stderr=sys.stderr, env=None, cwd=None)