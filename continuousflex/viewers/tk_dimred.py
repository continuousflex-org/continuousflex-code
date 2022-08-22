from continuousflex.viewers.nma_gui import TrajectoriesWindow, ClusteringWindow
import tkinter as tk
from pyworkflow.gui.widgets import Button, HotButton, ComboBox
from tkinter import Radiobutton

from pyworkflow.utils.properties import Icon
import numpy as np
import scipy as sp
from continuousflex.protocols.data import Point, Data, PathData

import pyworkflow.gui as gui

TOOL_TRAJECTORY = 1
TOOL_CLUSTERING = 2

class PCAWindowDimred(TrajectoriesWindow, ClusteringWindow):

    def __init__(self, **kwargs):
        TrajectoriesWindow.__init__(self, **kwargs)
        self.saveClusterCallback = kwargs.get('saveClusterCallback', None)
        self.numberOfPoints = kwargs.get('numberOfPoints', 10)
        print( kwargs.get('numberOfPoints'))
        self._alpha=self.alpha
        self._s=self.s
        self._clusterNumber = 0

    def _createContent(self, content):
        TrajectoriesWindow._createContent(self, content)
        self._createClusteringBox(content)
        self._exportBox(content)

    def _createFigureBox(self, content):
        frame = tk.LabelFrame(content, text='Figure')
        frame.columnconfigure(0, minsize=50)
        frame.columnconfigure(1, weight=1)  # , minsize=30)
        # Create the 'Axes' label
        self._addLabel(frame, 'Axes', 0, 0)

        # Create a listbox with x1, x2 ...
        listbox = tk.Listbox(frame, height=5,
                             selectmode=tk.MULTIPLE, bg='white')
        for x in range(1, self.dim + 1):
            listbox.insert(tk.END, 'x%d' % x)
        listbox.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        self.listbox = listbox

        # Selection controls
        self._addLabel(frame, 'Rejection', 1, 0)
        # Selection label
        self.selectionVar = tk.StringVar()
        self.clusterLabel = tk.Label(frame, textvariable=self.selectionVar)
        self.clusterLabel.grid(row=1, column=1, sticky='w', padx=5, pady=(10, 5))
        self._updateSelectionLabel()
        # --- Expression
        expressionFrame = tk.Frame(frame)
        expressionFrame.grid(row=2, column=1, sticky='w')
        tk.Label(expressionFrame, text='Expression').grid(row=0, column=0, sticky='ne')
        self.expressionVar = tk.StringVar()
        expressionEntry = tk.Entry(expressionFrame, textvariable=self.expressionVar,
                                   width=30, bg='white')
        expressionEntry.grid(row=0, column=1, sticky='nw')
        helpText = 'e.g. x1>0 and x1<100 or x3>20'
        tk.Label(expressionFrame, text=helpText).grid(row=1, column=1, sticky='nw')

        # Buttons
        buttonFrame = tk.Frame(frame)
        buttonFrame.grid(row=5, column=1, sticky='sew', pady=(10, 5))
        buttonFrame.columnconfigure(0, weight=1)
        resetBtn = Button(buttonFrame, text='Reset', command=self._onResetClick)
        resetBtn.grid(row=0, column=0, sticky='ne', padx=(5, 0))
        updateBtn = Button(buttonFrame, text='Update Plot', imagePath='fa-refresh.png',
                           command=self._onUpdateClick)
        updateBtn.grid(row=0, column=1, sticky='ne', padx=5)


        selFrame = tk.Frame(frame)
        selFrame.grid(row=6, column=1, sticky='w', pady=(10, 5), padx=5)
        tk.Label(selFrame, text="Interactive mode", font=self.fontBold).grid(row=0, column=0)

        self.selectTool = tk.IntVar()
        r1 = Radiobutton(selFrame, text="Trajectory", variable=self.selectTool, value=TOOL_TRAJECTORY, command=self._onUpdateClick)
        r1.grid(row=0, column=1, padx=5)
        r2 = Radiobutton(selFrame, text="Clustering", variable=self.selectTool, value=TOOL_CLUSTERING, command=self._onUpdateClick)
        r2.grid(row=0, column=2, padx=5)
        self.selectTool.set(TOOL_TRAJECTORY)

        frame.grid(row=0, column=0, sticky='new', padx=5, pady=(10, 5))

    def _onUpdateClick(self,e=None):
        if self.selectTool.get() == TOOL_TRAJECTORY :
            TrajectoriesWindow._onUpdateClick(self,e)
            self.createClusterBtn.config(state=tk.DISABLED)
            if (self.pathData.getSize() < self.numberOfPoints):
                self.updateClusterBtn.config(state=tk.NORMAL)
            self.eraseBtn.config(state=tk.DISABLED)
            self.trajSimBtn.config(state=tk.NORMAL)


        if self.selectTool.get() == TOOL_CLUSTERING:
            ClusteringWindow._onUpdateClick(self, e)
            self.createClusterBtn.config(state=tk.NORMAL)
            self.updateClusterBtn.config(state=tk.DISABLED)
            self.trajSimBtn.config(state=tk.DISABLED)
            self.eraseBtn.config(state=tk.NORMAL)

    def _exportBox(self,content):
        frame = tk.LabelFrame(content, text='Export')

        self._addLabel(frame, 'Name', 0, 0)
        self.clusterName = tk.StringVar()
        clusterEntry = tk.Entry(frame, textvariable=self.clusterName,
                                width=30, bg='white')
        clusterEntry.grid(row=0, column=1, pady=5)

        self.saveClusterBtn = Button(frame, text='Export', state=tk.DISABLED,
                              tooltip='export clusters to scipion', command=self._onSaveClusterClick)
        self.saveClusterBtn.grid(row=0, column=2, padx=5)


        self.loadBtn = Button(frame, text='Load', imagePath='fa-folder-open.png',
                              tooltip='Load a previous PCA clustering', command=self._onLoadClick)
        self.loadBtn.grid(row=0, column=3)


        frame.grid(row=3, column=0, sticky='new', padx=5, pady=(10, 5))



    def _createClusteringBox(self, content):
        frame = tk.LabelFrame(content, text='Clustering')
        frame.columnconfigure(0, minsize=50)
        frame.columnconfigure(1, weight=1)


        buttonsFrame = tk.Frame(frame)
        buttonsFrame.grid(row=1, column=0,
                          sticky='se', padx=5, pady=5)
        buttonsFrame.columnconfigure(0, weight=1)

        self.createClusterBtn = HotButton(buttonsFrame, text='New cluster', state=tk.DISABLED,
                                     tooltip='Create new cluster',
                                     imagePath='fa-plus-circle.png', command=self._onCreateCluster)
        self.createClusterBtn.grid(row=0, column=1, padx=5)
        self.eraseBtn = Button(buttonsFrame, text='Erase',  tooltip='Erase cluster', command=self._onErase)
        self.eraseBtn.grid(row=0, column=2, padx=5)


        frame.grid(row=2, column=0, sticky='new', padx=5, pady=(10, 5))

    def _createTrajectoriesBox(self, content):
        frame = tk.LabelFrame(content, text='Trajectories')
        # frame.columnconfigure(0, minsize=50)
        # frame.columnconfigure(1, weight=1)  # , minsize=30)

        buttonsFrame2 = tk.Frame(frame)
        buttonsFrame2.grid(row=0, column=0,
                          sticky='w', padx=5, pady=5)
        buttonsFrame2.columnconfigure(0, weight=1)
        self.trajSimBtn = Button(buttonsFrame2, text='Generate points', state=tk.NORMAL,
                                     tooltip='Generate trajectory points based on axis and trajectory type', command=self._onSimClick)
        self.trajSimBtn.grid(row=0, column=0, padx=5)
        self.trajAxisBtn = ComboBox(buttonsFrame2, choices=["axis %i"%(i+1) for i in range(self.dim)])
        self.trajAxisBtn.grid(row=0, column=1, padx=(5, 10))
        self.trajTypeBtn = ComboBox(buttonsFrame2, choices=["percentiles",
                                                            "linear betmeen min and max", "Linear betmeen -2*std and +2*std"
                                                            , "Gaussian betmeen min and max", "Gaussian betmeen -2*std and +2*std"])
        self.trajTypeBtn.grid(row=0, column=2, padx=(5, 5))

        buttonsFrame = tk.Frame(frame)
        buttonsFrame.grid(row=1, column=0,
                          sticky='w', padx=5, pady=5)
        buttonsFrame.columnconfigure(0, weight=1)
        self.generateBtn = HotButton(buttonsFrame, text='Show in VMD', state=tk.DISABLED,
                                     tooltip='Select trajectory points to generate the animations',
                                     imagePath='fa-plus-circle.png', command=self._onCreateClick)
        self.generateBtn.grid(row=0, column=0, padx=5)
        self.comboBtn = ComboBox(buttonsFrame, choices=["Inverse transformation", "cluster average", "cluster PCA"])
        self.comboBtn.grid(row=0, column=1, padx=(5, 10))


        buttonsFrame2 = tk.Frame(frame)
        buttonsFrame2.grid(row=2, column=0,
                          sticky='w', padx=5, pady=5)
        self.updateClusterBtn = HotButton(buttonsFrame2, text='Update cluster', state=tk.DISABLED,
                                     tooltip='Create new cluster',
                                     imagePath='fa-plus-circle.png', command=self._onUpdateCluster)
        self.updateClusterBtn.grid(row=0, column=0, padx=5)

        frame.grid(row=1, column=0, sticky='new', padx=5, pady=(5, 10))

    def _onSaveClusterClick(self, e=None):
        if self.saveClusterCallback:
            self.saveClusterCallback(self)

    def _onSimClick(self):
        self._onResetClick()
        traj_axis = self.trajAxisBtn.getValue()
        traj_type = self.trajTypeBtn.getValue()

        data_axis = np.array([p.getData()[traj_axis] for p in self.data])
        mean_axis =data_axis.mean()
        std_axis =data_axis.std()
        min_axis =data_axis.min()
        max_axis =data_axis.max()

        traj_points = np.zeros((self.numberOfPoints, self.dim))
        if traj_type== 0 :
            traj_points[:,traj_axis] = np.array(
                [np.percentile(data_axis,100*(i+1)/(self.numberOfPoints+1)) for i in range(self.numberOfPoints)]
            )
        elif traj_type== 1 :
            traj_points[:,traj_axis] = np.linspace(min_axis,max_axis,self.numberOfPoints)
        elif traj_type== 2:
            traj_points[:,traj_axis] = np.linspace(-2*std_axis,+2*std_axis,self.numberOfPoints)
        elif traj_type== 3:
            distribution = sp.stats.norm(loc=mean_axis, scale=std_axis)
            bounds_for_range = distribution.cdf([min_axis, max_axis])
            gaussTraj = distribution.ppf(np.linspace(*bounds_for_range, num=self.numberOfPoints))
            traj_points[:, traj_axis] = gaussTraj
        elif traj_type== 4:
            distribution = sp.stats.norm(loc=mean_axis, scale=std_axis)
            bounds_for_range = distribution.cdf([-2*std_axis, +2*std_axis])
            gaussTraj = distribution.ppf(np.linspace(*bounds_for_range, num=self.numberOfPoints))
            traj_points[:, traj_axis] = gaussTraj

        for i in range(self.numberOfPoints):
            self.pathData.addPoint(Point(pointId=i + 1, data=traj_points[i], weight=0))

        self._checkNumberOfPoints()
        self._onUpdateClick()

    def _onUpdateCluster(self):
        traj_arr = np.array([p.getData() for p in self.pathData])
        selection = np.array(self.listbox.curselection())
        traj_sel = traj_arr[:, selection]

        for point in self.data:
            point_sel = point.getData()[selection]
            closet_point = np.argmin(np.linalg.norm(traj_sel - point_sel, axis=1))
            point._weight = closet_point + 1

        self.saveClusterBtn.config(state=tk.NORMAL)
        self._onUpdateClick()
        self.setClusterNumber(self.numberOfPoints)


    def _onCreateCluster(self):
        self.setClusterNumber(self.getClusterNumber() +1)
        for point in self.data:
            if point.getState() == Point.SELECTED:
                point._weight =self.getClusterNumber()
        self.saveClusterBtn.config(state=tk.NORMAL)
        ClusteringWindow._onResetClick(self)

    def setClusterNumber(self, n):
        self._clusterNumber = n

    def _onErase(self):
        for point in self.data:
            if point.getState() == Point.SELECTED:
                point._weight =0.0
        self.saveClusterBtn.config(state=tk.NORMAL)
        ClusteringWindow._onResetClick(self)

    def _checkNumberOfPoints(self):
        TrajectoriesWindow._checkNumberOfPoints(self)
        self.updateClusterBtn.config(state=tk.NORMAL)

    def _onResetClick(self, e=None):
        self.updateClusterBtn.config(state=tk.DISABLED)
        self.saveClusterBtn.config(state=tk.DISABLED)
        self.setClusterNumber(0)

        for point in self.data:
            point._weight = 0
        TrajectoriesWindow._onResetClick(self, e)

    def getClusterName(self):
        return self.clusterName.get().strip()

    def getAnimationType(self):
        return self.comboBtn.getValue()

    def getClusterNumber(self):
        return self._clusterNumber