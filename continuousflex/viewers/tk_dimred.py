from continuousflex.viewers.nma_gui import TrajectoriesWindow, ClusteringWindow
import tkinter as tk
from pyworkflow.gui.widgets import Button, HotButton, ComboBox
from pyworkflow.utils.properties import Icon
import numpy as np
import scipy as sp
from continuousflex.protocols.data import Point, Data

class ClusteringWindowDimred(ClusteringWindow):

    def __init__(self, **kwargs):
        ClusteringWindow.__init__(self, **kwargs)
        self._clusterNumber = 1

    def _createClusteringBox(self, content):
        frame = tk.LabelFrame(content, text='Clustering')
        frame.columnconfigure(0, minsize=50)
        frame.columnconfigure(1, weight=1)
        # Animation name
        self._addLabel(frame, 'Name', 0, 0)
        self.clusterName = tk.StringVar()
        clusterEntry = tk.Entry(frame, textvariable=self.clusterName,
                                width=30, bg='white')
        clusterEntry.grid(row=0, column=1, sticky='nw', pady=5)

        buttonsFrame = tk.Frame(frame)
        buttonsFrame.grid(row=1, column=0,
                          sticky='se', padx=5, pady=5)
        buttonsFrame.columnconfigure(0, weight=1)

        self.createBtn = HotButton(buttonsFrame, text='Create cluster', state=tk.DISABLED,
                                     tooltip='Create new cluster',
                                     imagePath='fa-plus-circle.png', command=self._onCreateCluster)
        self.createBtn.grid(row=0, column=1, padx=5)

        self.saveClusterBtn = Button(buttonsFrame, text='Export', state=tk.DISABLED,
                              tooltip='export clusters to scipion', command=self._onSaveClusterClick)
        self.saveClusterBtn.grid(row=0, column=2, padx=5)

        frame.grid(row=2, column=0, sticky='new', padx=5, pady=(10, 5))

    def _onCreateCluster(self):
        for point in self.data:
            if point.getState() == Point.SELECTED:
                point._weight =self.getClusterNumber()
        self.setClusterNumber(self.getClusterNumber()+1)
        self.saveClusterBtn.config(state=tk.NORMAL)
        ClusteringWindow._onResetClick(self)

    def _onSaveClusterClick(self, e=None):
        if self.callback:
            self.callback(self)

    def getClusterName(self):
        return self.clusterName.get().strip()

    def getClusterNumber(self):
        return self._clusterNumber

    def setClusterNumber(self, n):
        self._clusterNumber =n

    def _onResetClick(self, e=None):
        for point in self.data:
            point._weight = 0
        ClusteringWindow._onResetClick(self, e)

class TrajectoriesWindowDimred(TrajectoriesWindow):

    def __init__(self, **kwargs):
        TrajectoriesWindow.__init__(self, **kwargs)
        self.saveClusterCallback = kwargs.get('saveClusterCallback', None)

    def _createContent(self, content):
        TrajectoriesWindow._createContent(self, content)
        self._createClusteringBox(content)

    def _createClusteringBox(self, content):
        frame = tk.LabelFrame(content, text='Clustering')
        frame.columnconfigure(0, minsize=50)
        frame.columnconfigure(1, weight=1)
        # Animation name
        self._addLabel(frame, 'Name', 0, 0)
        self.clusterName = tk.StringVar()
        clusterEntry = tk.Entry(frame, textvariable=self.clusterName,
                                width=30, bg='white')
        clusterEntry.grid(row=0, column=1, sticky='nw', pady=5)

        buttonsFrame = tk.Frame(frame)
        buttonsFrame.grid(row=1, column=0,
                          sticky='se', padx=5, pady=5)
        buttonsFrame.columnconfigure(0, weight=1)

        self.updateClusterBtn = HotButton(buttonsFrame, text='Update clusters', state=tk.DISABLED,
                                     tooltip='Generate clusters based on selected points',
                                     imagePath='fa-plus-circle.png', command=self._onCreateCluster)
        self.updateClusterBtn.grid(row=0, column=1, padx=5)

        self.saveClusterBtn = Button(buttonsFrame, text='Export', state=tk.DISABLED,
                              tooltip='export clusters to scipion', command=self._onSaveClusterClick)
        self.saveClusterBtn.grid(row=0, column=2, padx=5)

        frame.grid(row=2, column=0, sticky='new', padx=5, pady=(10, 5))

    def _createTrajectoriesBox(self, content):
        frame = tk.LabelFrame(content, text='Trajectories')
        frame.columnconfigure(0, minsize=50)
        frame.columnconfigure(1, weight=1)  # , minsize=30)

        # Animation name
        self._addLabel(frame, 'Name', 0, 0)
        self.animationVar = tk.StringVar()
        clusterEntry = tk.Entry(frame, textvariable=self.animationVar,
                                width=30, bg='white')
        clusterEntry.grid(row=0, column=1, sticky='nw', pady=5)

        self.loadBtn = Button(frame, text='Load', imagePath='fa-folder-open.png',
                              tooltip='Load a generated animation.', command=self._onLoadClick)
        self.loadBtn.grid(row=0, column=2, padx=5)


        buttonsFrame = tk.Frame(frame)
        buttonsFrame.grid(row=1, column=0,
                          sticky='se', padx=5, pady=5)
        buttonsFrame.columnconfigure(0, weight=1)
        self.generateBtn = HotButton(buttonsFrame, text='Show in VMD', state=tk.DISABLED,
                                     tooltip='Select trajectory points to generate the animations',
                                     imagePath='fa-plus-circle.png', command=self._onCreateClick)
        self.generateBtn.grid(row=0, column=0, padx=5)
        self.comboBtn = ComboBox(buttonsFrame, choices=["Inverse transformation", "cluster average", "cluster PCA"])
        self.comboBtn.grid(row=0, column=1, padx=(5, 10))

        buttonsFrame2 = tk.Frame(frame)
        buttonsFrame2.grid(row=2, column=0,
                          sticky='se', padx=5, pady=5)
        buttonsFrame2.columnconfigure(0, weight=1)
        self.trajSimBtn = HotButton(buttonsFrame2, text='Generate points', state=tk.NORMAL,
                                     tooltip='Generate trajectory points based on axis and trajectory type', command=self._onSimClick)
        self.trajSimBtn.grid(row=0, column=0, padx=5)
        self.trajAxisBtn = ComboBox(buttonsFrame2, choices=["axis %i"%(i+1) for i in range(self.dim)])
        self.trajAxisBtn.grid(row=0, column=1, padx=(5, 10))
        self.trajTypeBtn = ComboBox(buttonsFrame2, choices=["percentiles",
                                                            "linear betmeen min and max", "Linear betmeen -2*std and +2*std"
                                                            , "Gaussian betmeen min and max", "Gaussian betmeen -2*std and +2*std"])
        self.trajTypeBtn.grid(row=0, column=2, padx=(5, 5))

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



    def _onCreateCluster(self):
        traj_arr = np.array([p.getData() for p in self.pathData])
        selection = np.array(self.listbox.curselection())
        traj_sel = traj_arr[:,selection]

        for point in self.data:
            point_sel = point.getData()[selection]
            closet_point = np.argmin(np.linalg.norm(traj_sel - point_sel, axis=1))
            point._weight =closet_point +1

        self.saveClusterBtn.config(state=tk.NORMAL)
        self._onUpdateClick()

    def _checkNumberOfPoints(self):
        TrajectoriesWindow._checkNumberOfPoints(self)
        self.updateClusterBtn.config(state=tk.NORMAL)

    def _onResetClick(self, e=None):
        self.updateClusterBtn.config(state=tk.DISABLED)
        self.saveClusterBtn.config(state=tk.DISABLED)

        for point in self.data:
            point._weight = 0
        TrajectoriesWindow._onResetClick(self, e)

    def getClusterName(self):
        return self.clusterName.get().strip()

    def getAnimationType(self):
        return self.comboBtn.getValue()

