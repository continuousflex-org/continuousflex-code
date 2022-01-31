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
from continuousflex.protocols.utilities.genesis_utilities import *

from .plotter import FlexPlotter
from pyworkflow.utils import getListFromRangeString
import numpy as np
import os
import glob
import pwem.emlib.metadata as md

import pickle

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
                      label='Display Energy',
                      help='TODO')

        form.addParam('displayCC', params.LabelParam,
                      label='Display correlation coefficient',
                      help='TODO')

        form.addParam('displayRMSDts', params.LabelParam,
                      label='Display RMSD time series',
                      help='TODO')

        form.addParam('displayRMSD', params.LabelParam,
                      label='Display RMSD',
                      help='TODO')

        form.addParam('targetPDB', params.PathParam, default=None,
                        label="List of Target PDBs",
                        help='Use the file pattern as file location with /*.pdb')
        form.addParam('referencePDB', params.PathParam, default="",
                        label="Reference PDB (optional)",
                        help='TODO')

        form.addParam('alignTarget', params.BooleanParam, default=False,
                        label="Align Target PDB",
                        help='TODO')

        form.addParam('displayAngularDistance', params.LabelParam,
                      label='Display Angular distance',
                      help='TODO')

        form.addParam('displayAngularDistanceTs', params.LabelParam,
                      label='Display Angular distance Time series',
                      help='TODO')

        form.addParam('rigidBodyParams', params.FileParam, default=None,
                        label="Target Rigid Body Parameters",
                        help='TODO')


        form.addParam('displayPCA', params.LabelParam,
                      label='Display PCA',
                      help='TODO')

        form.addParam('displayTraj', params.LabelParam,
                      label='Display Trajecory',
                      help='TODO')

    def _getVisualizeDict(self):
        return {
            'displayEnergy': self._plotEnergy,
            'displayCC': self._plotCC,
            'displayRMSDts': self._plotRMSDts,
            'displayRMSD': self._plotRMSD,
            'displayAngularDistance': self._plotAngularDistance,
            'displayAngularDistanceTs': self._plotAngularDistanceTs,
            'displayPCA': self._plotPCA,
            'displayTraj': self._plotTraj,
                }

    def _plotTraj(self, paramName):
        fitlist = self.getFitlist()
        traj_viewer(pdb_file=self.protocol.getInputPDBprefix(fitlist[0] - 1)+".pdb",
                    dcd_file=self.protocol.getOutputPrefix(fitlist[0] - 1)+".dcd")

    def _plotEnergy(self, paramName):
        self._plotEnergyTotal()
        self._plotEnergyDetail()

    def _plotEnergyTotal(self):
        plotter = FlexPlotter()
        ax = plotter.createSubPlot("Energy", "Time (ps)", "Energy")
        ene_default = ["TOTAL_ENE", "POTENTIAL_ENE", "KINETIC_ENE"]

        fitlist = self.getFitlist()
        ene = {}
        for i in fitlist:
            outputPrefix = self.protocol.getOutputPrefixAll(i - 1)
            for j in outputPrefix:
                log_file = readLogFile(j + ".log")
                for e in ene_default:
                    if e in log_file:
                        if e in ene :
                            ene[e].append(log_file[e])
                        else:
                            ene[e] = [log_file[e]]

        x = np.arange(len(log_file["TOTAL_ENE"]))*\
            (int(self.protocol.eneout_period.get()) )*\
            float( self.protocol.time_step.get())
        for e in ene:
            ax.errorbar(x = x, y=np.mean(ene[e], axis=0), yerr=np.std(ene[e], axis=0), label=e,
                        capthick=1.7, capsize=5,elinewidth=1.7,
                        errorevery=np.max([len(log_file["STEP"]) //10,1]))
        plotter.legend()
        plotter.show()

    def _plotEnergyDetail(self):
        plotter = FlexPlotter()
        ax = plotter.createSubPlot("Energy", "Time (ps)", "Energy")
        ene_default = ["BOND", "ANGLE", "UREY-BRADLEY", "DIHEDRAL", "IMPROPER", "CMAP", "VDWAALS", "ELECT", "NATIVE_CONTACT",
               "NON-NATIVE_CONT", "RESTRAINT_TOTAL"]

        fitlist = self.getFitlist()
        ene = {}
        for i in fitlist:
            outputPrefix = self.protocol.getOutputPrefixAll(i - 1)
            for j in outputPrefix:
                log_file = readLogFile(j+".log")
                for e in ene_default:
                    if e in log_file:
                        if e in ene :
                            ene[e].append(log_file[e])
                        else:
                            ene[e] = [log_file[e]]

        x = np.arange(len(log_file["BOND"])) * \
            (int(self.protocol.eneout_period.get())) * \
            float(self.protocol.time_step.get())
        for e in ene:
            ax.errorbar(x = x, y=np.mean(ene[e], axis=0), yerr=np.std(ene[e], axis=0), label=e,
                        capthick=1.7, capsize=5,elinewidth=1.7,
                        errorevery=np.max([len(log_file["STEP"]) //10,1]))
        plotter.legend()
        plotter.show()

    def _plotCC(self, paramName):
        plotter = FlexPlotter()
        ax = plotter.createSubPlot("Correlation coefficient", "Time (ps)", "CC")

        # Get CC list
        fitlist = self.getFitlist()
        cc = []
        for i in fitlist:
            outputPrefix = self.protocol.getOutputPrefixAll(i-1)
            for j in outputPrefix:
                log_file = readLogFile(j + ".log")
                cc.append(log_file['RESTR_CVS001'])

        # Plot CC
        for i in range(len(cc)):
            x = np.arange(len(cc[i])) * \
                (int(self.protocol.eneout_period.get())) * \
                float(self.protocol.time_step.get())
            if len(cc) <= 50:
                ax.plot(x, cc[i], alpha=0.3, label="#%i"%i)

        try :
            cc_mean = np.mean(cc, axis=0)
            cc_std = np.std(cc, axis=0)
            ax.errorbar(x = x, y=cc_mean, yerr=cc_std,
                        capthick=1.7, capsize=5,elinewidth=1.7, color="black",
                        errorevery=np.max([len(log_file["STEP"]) //10,1]), label="Avergae")
        except TypeError:
            pass
        plotter.show()

    def _plotRMSDts(self, paramName):
        plotter = FlexPlotter()
        ax = plotter.createSubPlot("RMSD ($\AA$)", "Time (ps)", "RMSD ($\AA$)")

        # Get matching atoms
        if self.referencePDB.get() != "":
            ref_pdb = PDBMol(self.referencePDB.get())
        else:
            ref_pdb = PDBMol(self.protocol.getInputPDBprefix()+".pdb")
        target_pdb = PDBMol(self.getTargetPDB(1))
        idx = matchPDBatoms([ref_pdb, target_pdb], ca_only=True)

        # Get RMSD list
        fitlist = self.getFitlist()
        rmsd = []
        for i in fitlist:
            outputPrefix = self.protocol.getOutputPrefixAll(i-1)
            for j in outputPrefix:
                log_file = readLogFile(j + ".log")
                rmsd.append(rmsdFromDCD(outputPrefix=j, inputPDB=self.protocol.getInputPDBprefix(i-1)+".pdb",
                    targetPDB=self.getTargetPDB(i),idx=idx, align = self.alignTarget.get()))

        # Plot RMSD
        for i in range(len(rmsd)):
            x = np.arange(len(rmsd[i])) * \
                (int(self.protocol.crdout_period.get())) * \
                float(self.protocol.time_step.get())
            if len(rmsd) <=50:
                ax.plot(x, rmsd[i], alpha=0.3, label="#%i"%i)

        try :
            rmsd_mean = np.mean(rmsd, axis=0)
            rmsd_std = np.std(rmsd, axis=0)
            ax.errorbar(x = x, y=rmsd_mean, yerr=rmsd_std,
                        capthick=1.7, capsize=5,elinewidth=1.7, color="black",
                        errorevery=np.max([len(log_file["STEP"]) //10,1]), label="Average")
        except TypeError:
            pass
        plotter.legend()
        plotter.show()

    def _plotRMSD(self, paramName):
        plotter = FlexPlotter()
        ax = plotter.createSubPlot("RMSD ($\AA$)", "# Simulation", "RMSD ($\AA$)")

        # Get RMSD list
        fitlist = self.getFitlist()
        initial_mols = []
        final_mols = []
        target_mols = []
        for i in fitlist:
            inputPDB = self.protocol.getInputPDBprefix(i-1)+".pdb"
            targetPDB = self.getTargetPDB(i)
            outputPrefs = self.protocol.getOutputPrefixAll(i-1)
            for outputPrefix in outputPrefs:
                outputPDB = outputPrefix +".pdb"
                # if not os.path.exists(outputPDB):
                #     lastPDBFromDCD(inputPDB=self.protocol.getInputPDBprefix(i-1)+".pdb",
                #             inputDCD=outputPrefix+".dcd", outputPDB=outputPrefix+"tmp.pdb")
                #     outputPDB = outputPrefix+"tmp.pdb"

                initial_mols.append(PDBMol(inputPDB))
                final_mols.append(PDBMol(outputPDB))
                target_mols.append(PDBMol(targetPDB))

        if self.referencePDB.get() != "":
            ref_mol = PDBMol(self.referencePDB.get())
        else:
            ref_mol = initial_mols[0]
        idx = matchPDBatoms(mols=[ref_mol, target_mols[0]],ca_only=True)
        rmsdi=[]
        rmsdf=[]
        for i in range(len(fitlist)):
            for outputPrefix in outputPrefs:
                rmsdi.append(getRMSD(mol1=initial_mols[i],mol2=target_mols[i], idx=idx, align=self.alignTarget.get()))
                rmsdf.append(getRMSD(mol1=final_mols[i]  ,mol2=target_mols[i], idx=idx, align=self.alignTarget.get()))

        ax.plot(rmsdf, "o", color="tab:blue", label="RMSDf")
        ax.plot(rmsdi, "o", color="tab:green", label="RMSDi")

        plotter.legend()
        plotter.show()



    def getFitlist(self):
        return np.array(getListFromRangeString(self.fitRange.get()))

    def _plotAngularDistance(self, paramName):
        angular_dist = []
        shift_dist = []
        mdImgGT = md.MetaData(self.rigidBodyParams.get())
        fitlist = self.getFitlist()
        for i in fitlist:
            imgfn = self.protocol._getExtraPath("%s_current_angles.xmd" % (str(i).zfill(5)))
            if os.path.exists(imgfn):
                mdImgFn = md.MetaData(imgfn)

                angular_dist.append(getAngularDist(md1=mdImgGT, md2=mdImgFn, idx1=i,idx2=1))
                shift_dist.append(getShiftDist(md1=mdImgGT, md2=mdImgFn, idx1=i,idx2=1))

        plotter1 = FlexPlotter()
        ax1 = plotter1.createSubPlot("Angular Distance (°)", "# Image", "Angular Distance (°)")
        ax1.plot(angular_dist, "o")
        plotter1.show()

        print("Angular distance mean %f:"%np.mean(angular_dist))
        print("Angular distance std %f:"%np.std(angular_dist))

        plotter2 = FlexPlotter()
        ax2 = plotter2.createSubPlot("Shift Distance ($\AA$)", "# Image", "Shift Distance ($\AA$)")
        ax2.plot(shift_dist, "o")
        plotter2.show()

        print("Shift distance mean %f:"%np.mean(shift_dist))
        print("Shift distance std %f:"%np.std(shift_dist))

    def _plotAngularDistanceTs(self, paramName):
        mdImgGT = md.MetaData(self.rigidBodyParams.get())
        fitlist = self.getFitlist()
        niter= self.protocol.rb_n_iter.get()
        angular_dist = np.zeros((len(fitlist),niter))

        for i in range(len(fitlist)):
            for j in range(niter):
                imgfn = self.protocol._getExtraPath("%s_iter%i_angles.xmd" % (str(fitlist[i]).zfill(5), j))
                if os.path.exists(imgfn):
                    mdImgFn = md.MetaData(imgfn)
                    angular_dist[i,j] = getAngularDist(md1=mdImgGT, md2=mdImgFn, idx1=fitlist[i], idx2=1)

                else:
                    print("%s not found" %imgfn)

        plotter1 = FlexPlotter()
        ax1 = plotter1.createSubPlot("Angular Distance (°)", "Number of iterations", "Angular Distance (°)")
        for i in range(len(fitlist)):
            ax1.plot(angular_dist[i,:])
        plotter1.show()


    def _plotPCA(self, paramName):

        initPDB = PDBMol(self.protocol.getInputPDBprefix(0)+".pdb")

        # MAtch atoms with target
        if self.targetPDB.get() is not None:
            targetPDB = PDBMol(self.getTargetPDB(1))
            idx = matchPDBatoms([initPDB,targetPDB], ca_only=False)
        else:
            idx = np.array([np.arange(initPDB.n_atoms)]).T

        # Get Init PDB coords
        initPDBs = []
        for i in range(self.protocol.getNumberOfInputPDB()):
            mol = PDBMol(self.protocol.getInputPDBprefix(i)+".pdb")
            initPDBs.append(mol.coords[idx[:,0]].flatten())

        # Get fitted PDBs coords
        fitlist = self.getFitlist()
        fitPDBs = []
        for i in fitlist:
            outputPrefix = self.protocol.getOutputPrefixAll(i - 1)
            for j in outputPrefix:
                mol = PDBMol(j+".pdb")
                fitPDBs.append(mol.coords[idx[:,0]].flatten())

        data = fitPDBs + initPDBs
        length=[len(fitPDBs), len(initPDBs)]
        labels=["Fitted PDBs", "Init. PDBs"]

        # Get TargetPDBs coords
        if self.targetPDB.get() is not None:
            targetPDBs=[]
            for i in fitlist:
                targetPDBs.append(PDBMol(self.getTargetPDB(i)).coords[idx[:,1]].flatten())
            data = data+targetPDBs
            length.append(len(targetPDBs))
            labels.append("Target PDBs")

        # Display PCA
        initPDB.select_atoms(idx[:,0])
        fig, ax=compute_pca(data=data, length=length, labels=labels,
                    n_components=2, figsize=(5, 5), initdcd=initPDB)
        fig.show()

        np.save(file = self.protocol._getExtraPath("PCA_data.npy"), arr= data)
        np.save(file = self.protocol._getExtraPath("PCA_length.npy"), arr= length)
        np.save(file = self.protocol._getExtraPath("PCA_labels.npy"), arr= labels)

    def getTargetPDB(self, index):
        targetPDBlist = [f for f in glob.glob(self.targetPDB.get())]
        targetPDBlist.sort()
        if index-1 < len(targetPDBlist):
            return targetPDBlist[index-1]
        else:
            return targetPDBlist[0]

