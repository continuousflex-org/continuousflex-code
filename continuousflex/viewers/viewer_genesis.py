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


from pyworkflow.viewer import (ProtocolViewer, DESKTOP_TKINTER, WEB_DJANGO)
import pyworkflow.protocol.params as params
from continuousflex.protocols.protocol_genesis import ProtGenesis
from .plotter import FlexPlotter
from pyworkflow.utils import getListFromRangeString
import numpy as np
import os

from continuousflex.protocols.utilities.genesis_utilities import PDBMol, matchPDBatoms

class GenesisViewer(ProtocolViewer):
    """ Visualization of results from the GENESIS protocol
    """
    _label = 'viewer genesis'
    _targets = [ProtGenesis]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def _defineParams(self, form):
        form.addSection(label='Visualization')
        form.addParam('fitRange', params.NumericRangeParam,
                      label="List of fitting to display",
                      default='1',
                      help=' Examples:\n'
                           ' "1,3-5" -> [1,3,4,5]\n'
                           ' "1, 2, 4" -> [1,2,4]\n')
        form.addParam('displayEnergy', params.LabelParam,
                      label='[EMFIT] Display Energy',
                      help='TODO')

        form.addParam('displayCC', params.LabelParam,
                      label='[EMFIT] Display correlation coefficient',
                      help='TODO')

        form.addParam('displayRMSD', params.LabelParam,
                      label='[EMFIT] Display RMSD',
                      help='TODO')

        form.addParam('targetPDB', params.FileParam,
                      pointerClass='AtomStruct', label="[EMFIT] target PDB",
                      help='Select the target PDB.')

    def _getVisualizeDict(self):
        return {
            'displayEnergy': self._plotEnergy,
            'displayCC': self._plotCC,
            'displayRMSD': self._plotRMSD,
                }


    def _plotEnergy(self, paramName):
        self._plotEnergyTotal()
        self._plotEnergyDetail()

    def _plotEnergyTotal(self):
        plotter = FlexPlotter()
        ax = plotter.createSubPlot("Energy", "Time (ps)", "CC")
        ene_default = ["TOTAL_ENE", "POTENTIAL_ENE", "KINETIC_ENE"]

        fitlist = self.getFitlist()
        ene = {}
        time_step = float( self.protocol.time_step.get())
        for i in fitlist:
            log_file = readLogFile(self.protocol._getExtraPath("%s_output.log" % (str(i).zfill(5))))
            for e in ene_default:
                if e in log_file:
                    if e in ene :
                        ene[e].append(log_file[e])
                    else:
                        ene[e] = [log_file[e]]

        x = np.array(log_file["STEP"])*time_step

        for e in ene:
            ax.errorbar(x = x, y=np.mean(ene[e], axis=0), yerr=np.std(ene[e], axis=0), label=e,
                        capthick=1.7, capsize=5,elinewidth=1.7, errorevery=len(log_file["STEP"]) //10)
        plotter.legend()
        plotter.show()

    def _plotEnergyDetail(self):
        plotter = FlexPlotter()
        ax = plotter.createSubPlot("Energy", "Time (ps)", "CC")
        ene_default = ["BOND", "ANGLE", "UREY-BRADLEY", "DIHEDRAL", "IMPROPER", "CMAP", "VDWAALS", "ELECT", "NATIVE_CONTACT",
               "NON-NATIVE_CONT"]

        fitlist = self.getFitlist()
        ene = {}
        time_step = float( self.protocol.time_step.get())
        for i in fitlist:
            log_file = readLogFile(self.protocol._getExtraPath("%s_output.log" % (str(i).zfill(5))))
            for e in ene_default:
                if e in log_file:
                    if e in ene :
                        ene[e].append(log_file[e])
                    else:
                        ene[e] = [log_file[e]]

        x = np.array(log_file["STEP"])*time_step

        for e in ene:
            ax.errorbar(x = x, y=np.mean(ene[e], axis=0), yerr=np.std(ene[e], axis=0), label=e,
                        capthick=1.7, capsize=5,elinewidth=1.7, errorevery=len(log_file["STEP"]) //10)
        plotter.legend()
        plotter.show()

    def _plotCC(self, paramName):
        plotter = FlexPlotter()
        ax = plotter.createSubPlot("Correlation coefficient", "Time (ps)", "CC")

        # Get CC list
        fitlist = self.getFitlist()
        time_step = float( self.protocol.time_step.get())
        cc = []
        for i in fitlist:
            outputPrefix = self.protocol._getExtraPath("%s_output" % (str(i).zfill(5)))
            log_file = readLogFile(outputPrefix + ".log")
            cc.append(log_file['RESTR_CVS001'])

        # Plot CC
        x = np.array(log_file["STEP"])*time_step
        for i in range(len(cc)):
            ax.plot(x, cc[-1], color="tab:blue", alpha=0.3)
        ax.errorbar(x = x, y=np.mean(cc, axis=0), yerr=np.std(cc, axis=0),
                    capthick=1.7, capsize=5,elinewidth=1.7, color="tab:blue", errorevery=len(log_file["STEP"]) //10)

        plotter.show()

    def _plotRMSD(self, paramName):
        plotter = FlexPlotter()
        ax = plotter.createSubPlot("RMSD ($\AA$)", "Time (ps)", "RMSD ($\AA$)")

        fitlist = self.getFitlist()
        time_step = float( self.protocol.time_step.get())
        rmsd = []
        for i in fitlist:
            outputPrefix = self.protocol._getExtraPath("%s_output" % (str(i).zfill(5)))
            log_file = readLogFile(outputPrefix + ".log")
            rmsd.append(rmsdFromDCD(outputPrefix=outputPrefix, inputPDB=self.protocol.getInputPDBprefix(i)+".pdb",
                                    targetPDB=self.targetPDB.get().getFileName()))

        x = np.array(log_file["STEP"])*time_step
        for i in range(len(rmsd)):
            ax.plot(x, rmsd[i], color="tab:blue", alpha=0.3)
        ax.errorbar(x = x, y=np.mean(rmsd, axis=0), yerr=np.std(rmsd, axis=0),
                    capthick=1.7, capsize=5,elinewidth=1.7, color="tab:blue", errorevery=len(log_file["STEP"]) //10)

        plotter.show()


    def getFitlist(self):
        return np.array(getListFromRangeString(self.fitRange.get()))


def readLogFile(log_file):
    with open(log_file,"r") as file:
        header = None
        dic = {}
        for line in file:
            if line.startswith("INFO:"):
                if header is None:
                    header = line.split()
                    for i in range(1,len(header)):
                        dic[header[i]] = []
                else:
                    splitline = line.split()
                    if len(splitline) == len(header):
                        for i in range(1,len(header)):
                            dic[header[i]].append(float(splitline[i]))
    return dic

def rmsdFromDCD(outputPrefix, inputPDB, targetPDB):

    # EXTRACT PDBs from dcd file
    with open("%s_dcd2pdb.tcl" % outputPrefix, "w") as f:
        s = ""
        s += "mol load pdb %s dcd %s.dcd\n" % (inputPDB, outputPrefix)
        s += "set nf [molinfo top get numframes]\n"
        s += "for {set i 0 } {$i < $nf} {incr i} {\n"
        s += "[atomselect top all frame $i] writepdb %stmp$i.pdb\n" % outputPrefix
        s += "}\n"
        s += "exit\n"
        f.write(s)
    os.system("vmd -dispdev text -e %s_dcd2pdb.tcl > /dev/null" % outputPrefix)

    # DEF RMSD
    def RMSD(c1, c2):
        return np.sqrt(np.mean(np.square(np.linalg.norm(c1 - c2, axis=1))))

    # COMPUTE RMSD
    rmsd = []
    inputPDBmol = PDBMol(inputPDB)
    targetPDBmol = PDBMol(targetPDB)

    idx = matchPDBatoms([inputPDBmol, targetPDBmol], ca_only=True)
    rmsd.append(RMSD(inputPDBmol.coords[idx[:, 0]], targetPDBmol.coords[idx[:, 1]]))
    i=0
    while(os.path.exists("%stmp%i.pdb"%(outputPrefix,i+1))):
        f = "%stmp%i.pdb"%(outputPrefix,i+1)
        mol = PDBMol(f)
        rmsd.append(RMSD(mol.coords[idx[:, 0]], targetPDBmol.coords[idx[:, 1]]))
        i+=1

    # CLEAN TMP FILES AND SAVE
    os.system("rm -f %stmp*" % (outputPrefix))
    return rmsd