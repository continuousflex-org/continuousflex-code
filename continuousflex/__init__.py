# **************************************************************************
# *
# * Authors:
# * Mohamad Harastani (mohamad.harastani@upmc.fr)
# * Slavica Jonic (slavica.jonic@upmc.fr)
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

import pyworkflow.em
from continuousflex.constants import *
import pyworkflow.utils as pwutils
getXmippPath = importFromPlugin("xmipp3.base", 'getXmippPath')

_logo = "logo.png"

class Plugin(pyworkflow.em.Plugin):
    _homeVar = CONTINUOUSFLEX_HOME
    _pathVars = [CONTINUOUSFLEX_HOME]
    _supportedVersions = [VV]

    @classmethod
    def _defineVariables(cls):
        cls._defineEmVar(CONTINUOUSFLEX_HOME, 'xmipp')
        cls._defineEmVar(NMA_HOME,'nma')

    #   @classmethod
    #   def getEnviron(cls):
    #       """ Setup the environment variables needed to launch the program. """
    #      environ = Environ(os.environ)
    #       environ.update({
    #            'PATH': Plugin.getHome(),
    #        }, position=Environ.BEGIN)
    #
    #       return environ

    @classmethod
    def getEnviron(cls, xmippFirst=True):
        """ Create the needed environment for Xmipp programs. """
        environ = pwutils.Environ(os.environ)
        pos = pwutils.Environ.BEGIN if xmippFirst else pwutils.Environ.END
        environ.update({
            'PATH': getXmippPath('bin'),
            'LD_LIBRARY_PATH': getXmippPath('lib'),
            'PYTHONPATH': getXmippPath('pylib')
        }, position=pos)

        # environ variables are strings not booleans
        if os.environ.get('CUDA', 'False') != 'False':
            environ.update({
                'PATH': os.environ.get('CUDA_BIN', ''),
                'LD_LIBRARY_PATH': os.environ.get('NVCC_LIBDIR', '')
            }, position=pos)

        return environ



    @classmethod
    def isVersionActive(cls):
        return cls.getActiveVersion().startswith(VV)

    @classmethod
    def defineBinaries(cls, env):

        env.addPackage('nma', version='2.0', deps=['arpack'],
                       url='https://github.com/slajo/NMA_basic_code/raw/master/nma.tgz',
                       createBuildDir=False,
                       buildDir='nma',
                       target="nma",
                       commands=[('cd ElNemo; make; mv nma_* ..',
                                  'nma_elnemo_pdbmat'),
                                 ('cd NMA_cart; LDFLAGS=-L%s make; mv nma_* ..'
                                  % env.getLibFolder(), 'nma_diag_arpack')],
                       neededProgs=['gfortran'], default=True)


pyworkflow.em.Domain.registerPlugin(__name__)



