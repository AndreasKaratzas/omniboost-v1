
import re
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from typing import List
from PyQt5.QtWidgets import QDialog, QApplication, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class Window(QDialog):
    def __init__(self, names: List[str], dnns: np.ndarray, emb_dim: int, parent=None):
        super(Window, self).__init__(parent)
        self._names = names
        self._dnns = dnns
        self._emb_dim = emb_dim
        self._num_iter = 0

        # a figure instance to plot on
        self.figure = plt.figure(figsize=(15, 5))

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        # set the layout
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def test(self, i):
        """Plot random matrices

        Parameters
        ----------
        i : int
            The running loop index.
        """

        self.cached_workload = np.random.randn(3, 5, 3)

        self.figure.clear()
        ax_1 = self.figure.add_subplot(131)
        ax_2 = self.figure.add_subplot(132)
        ax_3 = self.figure.add_subplot(133)

        sns.heatmap(ax=ax_1, data=self.cached_workload[0, :, :], center=0)
        ax_1.set_title(self._names[0])
        sns.heatmap(ax=ax_2, data=self.cached_workload[1, :, :], center=0)
        ax_2.set_title(self._names[1])
        sns.heatmap(ax=ax_3, data=self.cached_workload[2, :, :], center=0)
        ax_3.set_title(self._names[2])

        self.figure.tight_layout()

        # plt.savefig(
        #     './epoch-' + str(i) + '.png',
        #     dpi=400
        # )

        # refresh canvas
        self.canvas.draw()
    
    def render(self, names: List[str], dev_workload: np.ndarray, export_path: str, epochs: int, verbose: int):
        
        self.figure.clear()

        ax_1 = self.figure.add_subplot(131)
        ax_2 = self.figure.add_subplot(132)
        ax_3 = self.figure.add_subplot(133)

        self.figure.suptitle('Platform Workload', fontsize=16)

        sns.heatmap(ax=ax_1, data=dev_workload[0, :, :], center=0)
        ax_1.set_title(names[0])

        if self._num_iter == 0:
            self.x_labels = [item.get_text() for item in ax_1.get_xticklabels()]
            self.y_labels = [item.get_text() for item in ax_1.get_yticklabels()]
            self.x_tick_labels = [self._dnns[i] for i in range(len(self.x_labels))]
            self.y_tick_labels = ["layer-" + str((i + 1) * np.ceil(self._emb_dim / len(
                self.y_labels)).astype(int)).zfill(2) for i in range(len(self.y_labels))]

            if int(re.search(r'\d+', self.y_tick_labels[-1]).group(0)) != self._emb_dim:
                self.y_tick_labels[-1] = "layer-" + str(self._emb_dim).zfill(2)

        ax_1.set_xticklabels(self.x_tick_labels)
        ax_1.set_yticklabels(self.y_tick_labels)
        plt.setp(ax_1.xaxis.get_majorticklabels(), rotation=60,
                 ha="right", rotation_mode="anchor")
        plt.setp(ax_1.yaxis.get_majorticklabels(), rotation=0,
                 ha="right", rotation_mode="anchor")

        sns.heatmap(ax=ax_2, data=dev_workload[1, :, :], center=0)
        ax_2.set_title(names[1])
        ax_2.set_xticklabels(self.x_tick_labels)
        ax_2.set_yticklabels(self.y_tick_labels)
        plt.setp(ax_2.xaxis.get_majorticklabels(), rotation=60,
                 ha="right", rotation_mode="anchor")
        plt.setp(ax_2.yaxis.get_majorticklabels(), rotation=0,
                 ha="right", rotation_mode="anchor")
        
        sns.heatmap(ax=ax_3, data=dev_workload[2, :, :], center=0)
        ax_3.set_title(names[2])
        ax_3.set_xticklabels(self.x_tick_labels)
        ax_3.set_yticklabels(self.y_tick_labels)
        plt.setp(ax_3.xaxis.get_majorticklabels(), rotation=60,
                 ha="right", rotation_mode="anchor")
        plt.setp(ax_3.yaxis.get_majorticklabels(), rotation=0,
                 ha="right", rotation_mode="anchor")

        self.figure.supxlabel(t='Models', y=0.02, ha='center')
        self.figure.supylabel(t='Embeddings', ha='center', x=0.02, rotation='vertical')
        self.figure.tight_layout()

        if epochs % verbose == 0:
            plt.savefig(
                str(export_path) + '/epoch-' + str(epochs) + '.png',
                dpi=400
            )

        self.canvas.draw()

        if self._num_iter == 0:
            self._num_iter += 1


class Render:
    def __init__(self, names: List[str], dnns: np.ndarray, emb_dim: int):
        self._names = names
        self._dnns = dnns
        self._emb_dim = emb_dim
        self._hikey_app = QApplication(sys.argv)
        self._stage_window = Window(names=self._names, dnns=dnns, emb_dim=emb_dim)
        self._stage_window.show()
        self._is_rendering = True

    def render(self, cached_workload: np.ndarray, export_path: str, epochs: int, verbose: int):
        self._stage_window.render(
            names=self._names,
            dev_workload=cached_workload,
            export_path=export_path,
            epochs=epochs,
            verbose=verbose)
        self._hikey_app.processEvents()
    
    def close(self):
        self._stage_window.close()
        self._hikey_app.quit()
        self._is_rendering = False


if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = Window()
    main.show()

    for i in range(5):
        main.test(i)
        app.processEvents()
    
    app.quit()
    # sys.exit(app.exec_())
