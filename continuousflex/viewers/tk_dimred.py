from continuousflex.viewers.nma_gui import TrajectoriesWindow, ClusteringWindow
import tkinter as tk
from pyworkflow.gui.widgets import Button, HotButton
from pyworkflow.utils.properties import Icon
import numpy as np

class ClusteringWindowDimred(ClusteringWindow):
    pass

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

        self.saveClusterBtn = Button(buttonsFrame, text='Save', state=tk.DISABLED,
                              tooltip='Save cluster', command=self._onSaveClusterClick)
        self.saveClusterBtn.grid(row=0, column=2, padx=5)

        frame.grid(row=2, column=0, sticky='new', padx=5, pady=(10, 5))

    def _onSaveClusterClick(self, e=None):
        if self.saveClusterCallback:
            self.saveClusterCallback()

    def _onCreateCluster(self):
        traj_arr = np.array([p.getData() for p in self.pathData])
        selection = np.array(self.listbox.curselection())
        traj_sel = traj_arr[:,selection]

        for point in self.data:
            point_sel = point.getData()[selection]
            closet_point = np.argmin(np.linalg.norm(traj_sel - point_sel, axis=1))
            point._weight =closet_point

        self.saveClusterBtn.config(state=tk.NORMAL)
        self._onUpdateClick()

    def _checkNumberOfPoints(self):
        TrajectoriesWindow._checkNumberOfPoints(self)
        self.updateClusterBtn.config(state=tk.NORMAL)

    def _onResetClick(self, e=None):
        self.updateClusterBtn.config(state=tk.DISABLED)
        self.saveClusterBtn.config(state=tk.DISABLED)

        for point in self.data:
            point._weight = 0.0
        TrajectoriesWindow._onResetClick(self, e)

    def getClusterName(self):
        return self.clusterName.get().strip()


