import gc
import sys
import glob
import h5py
import numpy as np
import pandas as pd
from scipy import spatial
import ruamel.yaml as yaml
from operator import itemgetter
from scipy.interpolate import interp1d
from pykdtree.kdtree import KDTree
from pyevtk.hl import gridToVTK
from scipy.ndimage import gaussian_filter

class readOutput():

    def __init__(self, filename=None, step=None):

        # Check input file exists
        try:
            with open(filename) as finput:
                pass
        except IOError as exc:
            print("Unable to open file: ",filename)
            raise IOError('The input file is not found...')

        # Open YAML file
        with open(filename, 'r') as finput:
            self.input = yaml.load(finput, Loader=yaml.Loader)

        self.radius = 6378137.
        self._inputParser()

        self.nbCPUs = len(glob.glob1(self.outputDir+"/h5/","topology.p*"))

        self._readElevationData(step)

        return


    def _inputParser(self):

        try:
            timeDict = self.input['time']
        except KeyError as exc:
            print("Key 'time' is required and is missing in the input file!")
            raise KeyError('Key time is required in the input file!')

        try:
            self.tStart = timeDict['start']
        except KeyError as exc:
            print("Key 'start' is required and is missing in the 'time' declaration!")
            raise KeyError('Simulation start time needs to be declared.')

        try:
            self.tEnd = timeDict['end']
        except KeyError as exc:
            print("Key 'end' is required and is missing in the 'time' declaration!")
            raise KeyError('Simulation end time needs to be declared.')

        try:
            outDict = self.input['output']
            try:
                self.outputDir = outDict['dir']
            except KeyError as exc:
                self.outputDir = 'output'
        except KeyError as exc:
            self.outputDir = 'output'

        return

    def _getCoordinates(self,step):

        for k in range(self.nbCPUs):
            df = h5py.File('%s/h5/topology.p%s.h5'%(self.outputDir,k), 'r')
            coords = np.array((df['/coords']))

            df2 = h5py.File('%s/h5/gLEM.%s.p%s.h5'%(self.outputDir,step,k), 'r')
            elev = np.array((df2['/elev']))

            if k == 0:
                self.x, self.y, self.z = np.hsplit(coords, 3)
                self.elev = elev
            else:
                self.x = np.append(self.x, coords[:,0])
                self.y = np.append(self.y, coords[:,1])
                self.z = np.append(self.z, coords[:,2])
                self.elev = np.append(self.elev, elev)
            df.close()

        self.nbPts = len(self.x)
        self.coords = np.zeros((self.nbPts,3))
        self.coords[:,0] = self.x.ravel()
        self.coords[:,1] = self.y.ravel()
        self.coords[:,2] = self.z.ravel()
        del coords, elev
        gc.collect()


        return

    def _readElevationData(self, step):

        self._getCoordinates(step)

        return
