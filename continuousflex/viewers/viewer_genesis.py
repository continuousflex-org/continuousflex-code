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
from continuousflex.protocols.protocol_genesis import *
from continuousflex.protocols.utilities.genesis_utilities import *

from .plotter import FlexPlotter
from pwem.viewers import VmdView, ChimeraView
from pyworkflow.utils import getListFromRangeString
import numpy as np
import os
import glob
import pwem.emlib.metadata as md
import re

from sklearn.decomposition import PCA
from matplotlib.pyplot import cm

class GenesisViewer(ProtocolViewer):
    """ Visualization of results from the GENESIS protocol
    """
    _label = 'viewer genesis'
    _targets = [ProtGenesis]
    _environments = [DESKTOP_TKINTER, WEB_DJANGO]

    def _defineParams(self, form):
        form.addSection(label='Visualization')

        if self.protocol.getNumberOfInputEM() >1:
            form.addParam('fitRange', params.NumericRangeParam,
                          label="EM data selection",
                          default="1-%i"%self.protocol.getNumberOfInputEM(),
                          important = True,
                          help=' Select the EM data to display. Examples:'
                               ' "1,3-5" -> [1,3,4,5]'
                               ' "1, 2, 4" -> [1,2,4]')
        if self.protocol.simulationType.get() == SIMULATION_REMD:
            form.addParam('replicaRange', params.NumericRangeParam,
                          label="Replica selection",
                          default="1-%i"%self.protocol.nreplica.get(),
                          help=' Select the replicas to display. Examples:'
                               ' "1,3-5" -> [1,3,4,5]'
                               ' "1, 2, 4" -> [1,2,4]')

        form.addParam('compareToPDB', params.BooleanParam, default=False,
                        label="Compare to external PDB",
                        help='TODO')
        form.addParam('targetPDB', params.PathParam, default=None,
                        label="Target PDB (s)", important=True,
                        help=' Target PDBs to compute RMSD against. Atom mathcing is performed between '
                             ' the output PDBs and the target PDBs. Use the file pattern as file location with /*.pdb',
                       condition= "compareToPDB")
        form.addParam('referencePDB', params.PathParam, default="",
                        label="Intial PDB",
                        help='Atom matching will ignore the output PDB and will use the initial PDB instead.',
                        expertLevel=params.LEVEL_ADVANCED,condition= "compareToPDB")

        form.addParam('alignTarget', params.BooleanParam, default=False,
                        label="Align Target PDB",
                        help='TODO',condition= "compareToPDB")

        group = form.addGroup('Chimera 3D view')
        group.addParam('displayChimera', params.LabelParam,
                      label='Display results in Chimera',
                      help='Show initial and final structures in Chimera')

        group = form.addGroup('VMD trajectory view')
        group.addParam('displayTrajVMD', params.LabelParam,
                      label='Display trajecory in VMD',
                      help='TODO')

        group = form.addGroup('Energy Analysis')
        group.addParam('displayEnergy', params.LabelParam,
                      label='Display Potential Energy',
                      help='Show time series of the potentials used in MD simulation/Minimization')

        group = form.addGroup('RMSD analysis')
        group.addParam('displayRMSDts', params.LabelParam,
                      label='Display RMSD time series',
                      help='TODO',condition= "compareToPDB")

        group.addParam('displayRMSD', params.LabelParam,
                      label='Display final RMSD',
                      help='TODO',condition= "compareToPDB")


        if self.protocol.EMfitChoice.get() != EMFIT_NONE:
            group = form.addGroup('Cryo EM fitting')
            group.addParam('displayCC', params.LabelParam,
                          label='Display Correlation Coefficient',
                          help='Show C.C. time series during the simulation')
            if self.protocol.EMfitChoice.get() == EMFIT_IMAGES and \
                    self.protocol.estimateAngleShift.get():
                group.addParam('rigidBodyParams', params.FileParam, default=None,
                              label="Target Rigid Body Parameters",
                              help='Target parameter to compare')
                group.addParam('displayAngularDistance', params.LabelParam,
                              label='Display final angular distance',
                              help='Show angular distance in degrees to the target rigid body params')
                group.addParam('displayAngularDistanceTs', params.LabelParam,
                              label='Display angular distance time series',
                              help='Show angular distance time series'
                                   'in degrees to the target rigid body params')
                group.addParam('symmetry', params.StringParam,
                              label='Symmetry group', default="C1",
                              help='Symmetry group for angular distance computation if any. Valid groups are : '
                                   'C1, Ci, Cs, Cn (from here on n must be an integer number with no more than 2 digits)' 
                                       ' Cnv, Cnh, Sn, Dn, Dnv, Dnh, T, Td, Th, O, Oh '
                                       ' I, I1, I2, I3, I4, I5, Ih, helical, dihedral, helicalDihedral ')

        group = form.addGroup('PCA analysis')
        group.addParam('displayPCA', params.LabelParam,
                      label='Display PCA',
                      help='TODO')

    def _getVisualizeDict(self):
        return {
            'displayChimera': self._plotChimera,
            'displayEnergy': self._plotEnergy,
            'displayCC': self._plotCC,
            'displayRMSDts': self._plotRMSDts,
            'displayRMSD': self._plotRMSD,
            'displayAngularDistance': self._plotAngularDistance,
            'displayAngularDistanceTs': self._plotAngularDistanceTs,
            'displayPCA': self._plotPCA,
            'displayTrajVMD': self._plotTrajVMD,
                }

    def _plotChimera(self, paramName):
        tmpChimeraFile = self.protocol._getExtraPath("chimera.cxc")
        index = self.getEMList()[0]

        with open(tmpChimeraFile, "w") as f:
            f.write("open %s.pdb \n"% os.path.abspath(self.protocol.getInputPDBprefix(index)))
            f.write("color #1 magenta \n" )
            count = 1

            outpdbfile = self.getOutputPrefixAll(index)[0] +".pdb"
            if os.path.exists(outpdbfile):
                if os.path.getsize(outpdbfile) != 0:
                    f.write("open %s \n"% os.path.abspath(outpdbfile))
                    count+=1
                    f.write("color #%s lime \n"%count)

            if self.compareToPDB.get():
                f.write("open %s \n" % os.path.abspath(self.getTargetPDB(index)))
                count+=1
                f.write("color #%s orange \n"%count)

            if self.protocol.EMfitChoice.get() == EMFIT_VOLUMES:
                f.write("open %s.mrc \n" % os.path.abspath(self.protocol.getInputEMprefix(index)))
                count+=1
                f.write("volume #%i transparency 0.5\n"%count)

            f.write("hide atoms \n")
            f.write("show cartoons \n")
            f.write("lighting soft \n")

        cv = ChimeraView(tmpChimeraFile)
        cv.show()

    def _plotTrajVMD(self, paramName):
        tmpVmdFile = self.protocol._getExtraPath("vmd.tcl")
        index = self.getEMList()[0]
        with open(tmpVmdFile, "w") as f:
            f.write("mol new %s.pdb waitfor all\n" % self.protocol.getInputPDBprefix(index))
            f.write("mol addfile %s.dcd waitfor all\n" % self.getOutputPrefixAll(index)[0])

            if self.protocol.forcefield.get() == FORCEFIELD_CAGO:
                f.write("mol modstyle 0 0 Tube \n")
            else:
                f.write("mol modstyle 0 0 NewCartoon \n")
            f.write("mol modcolor 0 0 Molecule\n")

            if self.protocol.EMfitChoice.get() == EMFIT_VOLUMES:
                f.write("mol addfile %s.mrc waitfor all\n" % self.protocol.getInputEMprefix(index))
                f.write("mol addrep 0 \n")
                f.write("mol modstyle 1 0 Isosurface 0.5 0 0 0 1 1  \n")
                f.write("mol modmaterial 1 0 Transparent \n")

            if self.compareToPDB.get():
                targetFile = self.getTargetPDB(index)
                f.write("set nf [molinfo top get numframes]\n")
                f.write("mol new %s waitfor all\n" %targetFile)
                f.write( "for {set i 1 } {$i < $nf} {incr i} {\n")
                f.write( "animate dup frame 0 1\n")
                f.write("}\n")
                if self.protocol.forcefield.get() == FORCEFIELD_CAGO:
                    f.write("mol modstyle 0 1 Tube \n")
                else:
                    f.write("mol modstyle 0 1 NewCartoon \n")
                f.write("mol modcolor 0 1 Molecule\n")
            f.write("animate style Loop\n")
            f.write("display projection Orthographic\n")

        vmdviewer = VmdView(" -e " + tmpVmdFile)
        vmdviewer.show()

    def _plotEnergy(self, paramName):
        self._plotEnergyTotal()
        self._plotEnergyDetail()

    def _plotEnergyTotal(self):
        ene_default = ["TOTAL_ENE", "POTENTIAL_ENE", "KINETIC_ENE"]

        ene = {}
        for i in self.getEMList():
            outputPrefix = self.getOutputPrefixAll(i)
            for j in outputPrefix:
                log_file = readLogFile(j + ".log")
                for e in ene_default:
                    if e in log_file:
                        if e in ene :
                            ene[e].append(log_file[e])
                        else:
                            ene[e] = [log_file[e]]
        enelist =[]
        labels=[]
        for e in ene :
            labels.append(e)
            enelist.append(ene[e])


        self.genesisPlotter(title="Energy (kcal/mol)", data=enelist, ndata=len(enelist),
                            nrep=len(enelist[0]), labels=labels)

    def _plotEnergyDetail(self):
        ene_default = ["BOND", "ANGLE", "UREY-BRADLEY", "DIHEDRAL", "IMPROPER", "CMAP", "VDWAALS", "ELECT", "NATIVE_CONTACT",
               "NON-NATIVE_CONT", "RESTRAINT_TOTAL"]

        ene = {}
        for i in self.getEMList():
            outputPrefix = self.getOutputPrefixAll(i)
            for j in outputPrefix:
                log_file = readLogFile(j+".log")
                for e in ene_default:
                    if e in log_file:
                        if e in ene :
                            ene[e].append(log_file[e])
                        else:
                            ene[e] = [log_file[e]]
        enelist =[]
        labels=[]
        for e in ene :
            labels.append(e)
            enelist.append(ene[e])

        self.genesisPlotter(title="Energy (kcal/mol)", data=enelist, ndata=len(enelist),
                            nrep=len(enelist[0]), labels=labels)

    def _plotCC(self, paramName):
        # Get CC list
        cc = []
        labels=[]
        emlist = self.getEMList()
        for i in emlist:
            outputPrefix = self.getOutputPrefixAll(i)
            cc_rep = []
            labels.append("CC %s"%str(i))
            for j in outputPrefix:
                log_file = readLogFile(j + ".log")
                if 'RESTR_CVS001' in log_file:
                    cc_rep.append(log_file['RESTR_CVS001'])
                else:
                    raise RuntimeError("CC not present in the log file")
            cc.append(cc_rep)

        self.genesisPlotter(title="CC", data=cc, ndata=len(emlist),
                            nrep=len(self.getOutputPrefixAll()), labels=labels)


    def genesisPlotter(self, title, data, ndata, nrep, labels):
        plotter = FlexPlotter()
        ax = plotter.createSubPlot(title, "", title)
        nmax= 10
        colors = [cm.get_cmap("tab10", 10)(i) for i in range(nmax) ]

        for i in range(ndata):
            if ndata <= nmax and nrep > 1:
                try:
                    meandata = np.mean(data[i], axis=0)
                    stddata = np.std(data[i], axis=0)
                    x = self.getTimePeriod(len(meandata))
                    ax.errorbar(x=x, y=meandata, yerr=stddata,
                                capthick=1.7, capsize=5, elinewidth=1.7, color=colors[i] if ndata!= 1 else "black",
                                errorevery=np.max([len(meandata) // 10, 1]), label="%s Average" % labels[i])
                except TypeError:
                    x = self.getTimePeriod(len(data[i][0]))
                    ax.plot(x, data[i][0], color=colors[i], label=labels[i])

            for j in range(nrep):
                x = self.getTimePeriod(len(data[i][j]))
                if 1 < nrep <= nmax:
                    if ndata == 1 :
                        ax.plot(x, data[i][j], color= colors[j], alpha=0.5, label="#%i"%j)
                    else:
                        ax.plot(x, data[i][j], color= colors[i], alpha=0.5)
                if nrep == 1 and ndata <= 10:
                    ax.plot(x, data[i][j], color= colors[i],label=labels[i])
        if ndata > nmax :
            try:
                meandata = np.mean(data, axis=(0,1))
                stddata = np.std(data, axis=(0,1))
                x = self.getTimePeriod(len(meandata))
                ax.errorbar(x=x, y=meandata, yerr=stddata,
                            capthick=1.7, capsize=5, elinewidth=1.7, color="black",
                            errorevery=np.max([len(meandata) // 10, 1]), label="Global Average")
            except TypeError:
                x = self.getTimePeriod(len(data[0][0]))
                ax.plot(x, data[0][0], color=colors[0], label=labels[0])
        xlabel = "Number of iterations" if self.protocol.simulationType.get() == SIMULATION_MIN else "Time (ps)"
        ax.set_xlabel(xlabel)
        plotter.legend()
        plotter.show()

    def _plotRMSDts(self, paramName):

        # Get matching atoms
        if self.referencePDB.get() != "":
            ref_pdb = PDBMol(self.referencePDB.get())
        else:
            ref_pdb = PDBMol(self.protocol.getInputPDBprefix()+".pdb")
        target_pdb = PDBMol(self.getTargetPDB())
        idx = matchPDBatoms([ref_pdb, target_pdb], ca_only=True)

        # Get RMSD list
        rmsd = []
        labels=[]
        emlist = self.getEMList()
        for i in emlist:
            outputPrefix = self.getOutputPrefixAll(i)
            labels.append("RMSD %s"%str(i))
            rmsd_rep=[]
            for j in outputPrefix:
                rmsd_rep.append(rmsdFromDCD(outputPrefix=j, inputPDB=self.protocol.getInputPDBprefix(i)+".pdb",
                    targetPDB=self.getTargetPDB(i),idx=idx, align = self.alignTarget.get()))
            rmsd.append(rmsd_rep)

        self.genesisPlotter(title="RMSD ($\AA$)", data=rmsd, ndata=len(emlist),
                            nrep=len(self.getOutputPrefixAll()), labels=labels)


    def _plotRMSD(self, paramName):
        plotter = FlexPlotter()
        ax = plotter.createSubPlot("RMSD ($\AA$)", "# Simulation", "RMSD ($\AA$)")

        # Get RMSD list
        initial_mols = []
        final_mols = []
        target_mols = []
        for i in self.getEMList():
            inputPDB = self.protocol.getInputPDBprefix(i)+".pdb"
            targetPDB = self.getTargetPDB(i)
            outputPrefs = self.getOutputPrefixAll(i)
            target_mols.append(PDBMol(targetPDB))
            initial_mols.append(PDBMol(inputPDB))
            for outputPrefix in outputPrefs:
                outputPDB = outputPrefix +".pdb"
                final_mols.append(PDBMol(outputPDB))

        if self.referencePDB.get() != "":
            ref_mol = PDBMol(self.referencePDB.get())
        else:
            ref_mol = initial_mols[0]
        idx = matchPDBatoms(mols=[ref_mol, target_mols[0]],ca_only=True)
        rmsdi=[]
        rmsdf=[]
        for i in range(len(self.getEMList())):
            for j in range(len(outputPrefs)):
                rmsdi.append(getRMSD(mol1=initial_mols[i],mol2=target_mols[i], idx=idx, align=self.alignTarget.get()))
                rmsdf.append(getRMSD(mol1=final_mols[i*len(outputPrefs) + j]  ,
                                     mol2=target_mols[i], idx=idx, align=self.alignTarget.get()))

        ax.plot(rmsdf, "o", color="tab:blue", label="Final RMSD", markeredgecolor='black')
        ax.plot(rmsdi, "o", color="tab:green", label="Initial RMSD", markeredgecolor='black')
        plotter.legend()
        plotter.show()

    def _plotAngularDistance(self, paramName):
        angular_dist = []
        shift_dist = []
        mdImgGT = md.MetaData(self.rigidBodyParams.get())
        tmpPrefix = self.protocol._getExtraPath("tmpAngles")
        for i in self.getEMList():
            imgfn = self.protocol._getExtraPath("%s_current_angles.xmd" % (str(i+1).zfill(5)))
            if os.path.exists(imgfn):
                angDist, shftDist = getAngularShiftDist(angle1MetaFile=imgfn,
                                    angle2MetaData=mdImgGT, angle2Idx=int(i+1),
                                    tmpPrefix=tmpPrefix, symmetry=self.symmetry.get())
                angular_dist.append(angDist)
                shift_dist.append(shftDist)

        plotter1 = FlexPlotter()
        ax1 = plotter1.createSubPlot("Angular Distance (°)", "# Image", "Angular Distance (°)")
        ax1.plot(angular_dist, "o")
        plotter1.show()

        print("Angular distance mean %f:"%np.mean(angular_dist))
        print("Angular distance std %f:"%np.std(angular_dist))

        plotter2 = FlexPlotter()
        ax2 = plotter2.createSubPlot("Shift Distance (pix)", "# Image", "Shift Distance (pix)")
        ax2.plot(shift_dist, "o")
        plotter2.show()

        print("Shift distance mean %f:"%np.mean(shift_dist))
        print("Shift distance std %f:"%np.std(shift_dist))

    def _plotAngularDistanceTs(self, paramName):
        mdImgGT = md.MetaData(self.rigidBodyParams.get())
        EMList = self.getEMList()
        niter= self.protocol.rb_n_iter.get()
        angular_dist = np.zeros((len(EMList),niter))
        tmpPrefix = self.protocol._getExtraPath("tmpAngles")


        for i in range(len(EMList)):
            for j in range(niter):
                imgfn = self.protocol._getExtraPath("%s_iter%i_angles.xmd" % (str(EMList[i]+1).zfill(5), j))
                if os.path.exists(imgfn):
                    angDist,_ = getAngularShiftDist(angle1MetaFile=imgfn,
                                    angle2MetaData=mdImgGT, angle2Idx=int(EMList[i]+1),
                                    tmpPrefix=tmpPrefix, symmetry=self.symmetry.get())
                    angular_dist[i, j] = angDist

                else:
                    print("%s not found" %imgfn)

        plotter1 = FlexPlotter()
        ax1 = plotter1.createSubPlot("Angular Distance (°)", "Number of iterations", "Angular Distance (°)")
        for i in range(len(EMList)):
            ax1.plot(angular_dist[i,:])
        plotter1.show()


    def _plotPCA(self, paramName):

        initPDB = PDBMol(self.protocol.getInputPDBprefix()+".pdb")

        # MAtch atoms with target
        if self.compareToPDB.get():
            targetPDB = PDBMol(self.getTargetPDB())
            if self.referencePDB.get() != "":
                refPDB = PDBMol(self.referencePDB.get())
            else:
                refPDB = initPDB
            matchingAtoms = matchPDBatoms([refPDB,targetPDB], ca_only=False)
        else:
            matchingAtoms = np.array([np.arange(initPDB.n_atoms)]).T

        # Get Init PDB coords
        initPDBs = []
        for i in range(self.protocol.getNumberOfInputPDB()):
            mol = PDBMol(self.protocol.getInputPDBprefix(i)+".pdb")
            initPDBs.append(mol.coords[matchingAtoms[:,0]].flatten())

        # Get fitted PDBs coords
        fitPDBs = []
        fitMols = []
        for i in self.getEMList():
            outputPrefix = self.getOutputPrefixAll(i)
            for j in outputPrefix:
                mol = PDBMol(j+".pdb")
                fitPDBs.append(mol.coords[matchingAtoms[:,0]].flatten())
            fitMols.append(mol)

        data = fitPDBs + initPDBs
        length=[len(fitPDBs), len(initPDBs)]
        labels=["Fitted PDBs", "Init. PDBs"]

        # Get TargetPDBs coords
        if self.compareToPDB.get():
            targetPDBs=[]
            for i in self.getEMList():
                targetMol = PDBMol(self.getTargetPDB(i))
                if self.alignTarget.get():
                    alignMol(fitMols[i], targetMol, idx=matchingAtoms)
                targetPDBs.append(targetMol.coords[matchingAtoms[:,1]].flatten())
            data = data+targetPDBs
            length.append(len(targetPDBs))
            labels.append("Target PDBs")

        # Compute PCA
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(np.array(data)).T

        # Plot PCA data
        idx_cumsum = np.concatenate((np.array([0]), np.cumsum(length))).astype(int)
        plotter = FlexPlotter()
        ax = plotter.createSubPlot("PCA", "PCA component 1", "PCA component 2")
        for i in range(len(length)):
            plotter.plot(pca_components[0, idx_cumsum[i]:idx_cumsum[i + 1]],
                         pca_components[1, idx_cumsum[i]:idx_cumsum[i + 1]],
                         "o", label=labels[i],
                         markeredgecolor='black')
        plotter.legend()
        plotter.show()
        fig = plotter.getFigure()

        # Prepare onclick event
        click_coord = []
        inv_pca = []
        n_inv_pca = 10
        initPDB.select_atoms(matchingAtoms[:,0])

        def onclick(event):
            if len(click_coord) < 2:
                click_coord.append((event.xdata, event.ydata))
                x = event.xdata
                y = event.ydata

            if len(click_coord) == 2:
                click_sel = np.array([np.linspace(click_coord[0][0], click_coord[1][0], n_inv_pca),
                                      np.linspace(click_coord[0][1], click_coord[1][1], n_inv_pca)
                                      ])
                ax.plot(click_sel[0], click_sel[1], "-o", color="black")
                inv_pca.insert(0, pca.inverse_transform(click_sel.T))
                click_coord.clear()
                fig.canvas.draw()

                initdcdcp = initPDB.copy()
                coords_list = []
                for i in range(n_inv_pca):
                    coords_list.append(inv_pca[0][i].reshape((initdcdcp.n_atoms, 3)))
                tmpPath = self.protocol._getTmpPath("traj")
                save_dcd(mol=initdcdcp, coords_list=coords_list, prefix=tmpPath)
                initdcdcp.coords = coords_list[0]
                initdcdcp.save(tmpPath+".pdb")
                vmdviewer = VmdView("%s.pdb %s.dcd"%(tmpPath, tmpPath))
                vmdviewer.show()

        fig.canvas.mpl_connect('button_press_event', onclick)

        np.save(file = self.protocol._getExtraPath("PCA_data.npy"), arr= data)
        np.save(file = self.protocol._getExtraPath("PCA_length.npy"), arr= length)
        np.save(file = self.protocol._getExtraPath("PCA_labels.npy"), arr= labels)


    def getEMList(self):
        if self.protocol.getNumberOfInputEM() > 1:
            return np.array(getListFromRangeString(self.fitRange.get())) -1
        else:
            return np.array([0])

    def getTargetPDB(self, index=0):
        targetPDBlist = [f for f in glob.glob(self.targetPDB.get())]
        targetPDBlist.sort()
        if index < len(targetPDBlist):
            return targetPDBlist[index]
        else:
            return targetPDBlist[0]

    def getOutputPrefixAll(self, index=0):
        outPrf = np.array(self.protocol.getOutputPrefixAll(index))
        if self.protocol.simulationType.get() == SIMULATION_REMD:
            return outPrf[np.array(getListFromRangeString(self.replicaRange.get())) - 1]
        else:
            return outPrf

    def getTimePeriod(self, length):
        if self.protocol.simulationType.get() == SIMULATION_MIN:
            timestep = 1.0
        else:
            timestep = float(self.protocol.time_step.get())
        eneperiod = int(self.protocol.eneout_period.get())

        return np.arange(length) * eneperiod * timestep

