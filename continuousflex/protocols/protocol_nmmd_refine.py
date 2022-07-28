# **************************************************************************
# * Authors: Rémi Vuillemot             (remi.vuillemot@upmc.fr)
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
# **************************************************************************

from continuousflex.protocols.protocol_genesis import *
import pyworkflow.protocol.params as params

class ProtNMMDRefine(ProtGenesis):
    """ Protocol to perform NMMD refinement using GENESIS """
    _label = 'NMMD refine'

    def __init__(self, **kwargs):
        ProtGenesis.__init__(self, **kwargs)

    # --------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Refinement')

        form.addParam('numberOfIter', params.IntParam, label="Number of iterations", default=3,
                      help="TODO", important=True)

        ProtGenesis._defineParams(self, form)


    def _insertAllSteps(self):

        # Convert input PDB
        self._insertFunctionStep("convertInputPDBStep")

        # Convert normal modes
        if (self.simulationType.get() == SIMULATION_NMMD or self.simulationType.get() == SIMULATION_RENMMD):
            self._insertFunctionStep("convertNormalModeFileStep")

        # Convert input EM data
        if self.EMfitChoice.get() != EMFIT_NONE:
            self._insertFunctionStep("convertInputEMStep")

        for iter_global in range(self.numberOfIter.get()):

            # Create INP files
            self._insertFunctionStep("createINPs")

            # RUN simulation
            if not self.disableParallelSim.get() and  \
                self.getNumberOfSimulation() >1  and  existsCommand("parallel") :
                self._insertFunctionStep("runSimulationParallel")
            else:
                if not self.disableParallelSim.get() and  \
                    self.getNumberOfSimulation() >1  and  not existsCommand("parallel"):
                    self.warning("Warning : Can not use parallel computation for GENESIS,"
                                        " please install \"GNU parallel\". Running in linear mode.")
                for i in range(self.getNumberOfSimulation()):
                    self._insertFunctionStep("runSimulation", i)

        # Create output data
        self._insertFunctionStep("createOutputStep")


