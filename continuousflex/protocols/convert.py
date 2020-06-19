# **************************************************************************
# *
# * Authors:     
# * J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
# * Slavica Jonic (slavica.jonic@upmc.fr)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
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

import os
from collections import OrderedDict

from pwem.emlib import (MDL_NMA_MODEFILE, MDL_NMA_COLLECTIVITY, MDL_NMA_SCORE,
                        MDL_ORDER)
from pyworkflow.utils import Environ
from pwem.objects import NormalMode

from xmipp3.convert import rowToObject, objectToRow
from xmipp3.constants import NMA_HOME
            
MODE_DICT = OrderedDict([ 
       ("_modeFile", MDL_NMA_MODEFILE),
       ("_collectivity", MDL_NMA_COLLECTIVITY),
       ("_score", MDL_NMA_SCORE)
       ])


def rowToMode(row):
    """ Set properties of a NormalMode object from a Metadata row. """
    mode = NormalMode()
    rowToObject(row, mode, MODE_DICT)
    mode.setObjId(row.getValue(MDL_ORDER))
    return mode


def modeToRow(mode, row):
    """ Write the MetaData row from a given NormalMode object. """
    row.setValue(MDL_ORDER, int(mode.getObjId()))
    objectToRow(mode, row, MODE_DICT)
    
    
def getNMAEnviron():
    """ Create the needed environment for NMA programs. """
    from xmipp3 import Plugin
    environ = Plugin.getEnviron()
    environ.update({'PATH': Plugin.getVar(NMA_HOME)}, position=Environ.BEGIN)
    return environ
