import gc
import sys
import glob
import h5py
import math
import meshio
import numpy as np
import pandas as pd
from scipy import spatial
import ruamel.yaml as yaml
import os.path as path
from time import clock
from scipy.linalg import norm
from scipy import sum, average
from operator import itemgetter
from scipy.interpolate import interp1d
from pyevtk.hl import gridToVTK
from netCDF4 import Dataset
from scipy import ndimage
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import scipy.optimize
from scipy.stats import pearsonr
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import fileinput

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import warnings
warnings.filterwarnings("ignore")

class buildInput():

    def __init__(self, fmesh=None, fcoords=None):

        # First we read the spherical mesh structure for the considered resolution:
        mesh_struct = np.load(fmesh+'.npz')

        self.ngbIDs = mesh_struct['n']
        self.vertices = mesh_struct['v']
        self.cells = mesh_struct['c']
        self.fixtree = spatial.cKDTree(self.vertices,leafsize=10)

        print("Load spherical mesh {}".format(fmesh+'.npz'))
        #print("Field names: {}".format(mesh_struct.files))

        # Then we load the corresponding coordinates in longitudes and latitudes to map it on the eTOPO1 dataset
        lonlat = np.load(fcoords+'.npz')
        self.icoords = lonlat['v']
        self.ids = None

        self.deg = 0.
        self.ilons = None
        self.ilats = None
        self.ilon = None
        self.ilat = None
        self.mesh_z = None

        print("Load lon/lat grid {}".format(fcoords+'.npz'))

        return

    def buildPaleoTopo(self, ncfile=None, outfile=None, visvtk=False, filter=2):
        '''
        Reading paleotopography grid
        '''

        t0 = clock()
        paleotopo = Dataset(ncfile, "r", format="NETCDF4")
        paleoz = np.fliplr(paleotopo['z'][:,:].T)
        print("Load NETCDF paleofile {}".format(ncfile))

        # Apply some smoothing is necessary...
        if filter>0:
            paleoz = ndimage.gaussian_filter(paleoz,sigma=filter)

        # Interpolate the paleogrid on global mesh
        self.mesh_z = ndimage.map_coordinates(paleoz, self.icoords , order=2, mode='nearest').astype(np.float)

        # Save the mesh as compressed numpy file for global simulation
        np.savez_compressed(outfile, v=self.vertices, c=self.cells, n=self.ngbIDs.astype(int), z=self.mesh_z)
        print("Processing {} to create {} done in {}s".format(ncfile,outfile+".npz",int(clock()-t0)))

        if visvtk:
            t0 = clock()
            vis_mesh = meshio.Mesh(self.vertices, {'triangle': self.cells}, point_data={"z":self.mesh_z})
            meshio.write(outfile+".vtk", vis_mesh)
            print("Writing VTK file {} in {}s".format(outfile+".vtk",int(clock()-t0)))

        return

    def buildPaleoDisp(self, gPlates=None, outfile=None, visvtk=False):
        '''
        Reading paleo-displacement grids
        '''

        t0 = clock()

        # Open gPlates 1 degree 3D displacement maps (xy files)
        data = pd.read_csv(gPlates, sep=r'\s+', engine='c', header=None, skiprows=[0,1,2,3,4,5,65166],
                           error_bad_lines=True, na_filter=False, dtype=np.float, low_memory=False)

        # Read dataset
        lon = data.values[:,0]
        lat = data.values[:,1]

        # Conversion from cm/yr to m/yr
        tmpx = data.values[:,2]/100.
        tmpy = data.values[:,3]/100.
        tmpz = data.values[:,4]/100.

        # Reshape dataset
        tmpx = np.fliplr(tmpx.reshape((181,360)).T)
        tmpy = np.fliplr(tmpy.reshape((181,360)).T)
        tmpz = np.fliplr(tmpz.reshape((181,360)).T)
        dispX = np.zeros((361,181))
        dispY = np.zeros((361,181))
        dispZ = np.zeros((361,181))
        dispX[:360,:] = tmpx
        dispX[-1,:] = tmpx[0,:]
        dispY[:360,:] = tmpy
        dispY[-1,:] = tmpy[0,:]
        dispZ[:360,:] = tmpz
        dispZ[-1,:] = tmpz[0,:]
        
        # Interpolate the paleo displacement on global mesh
        dX = ndimage.map_coordinates(dispX , self.icoords/10., order=3, mode='nearest').astype(np.float)
        dY = ndimage.map_coordinates(dispY , self.icoords/10., order=3, mode='nearest').astype(np.float)
        dZ = ndimage.map_coordinates(dispZ , self.icoords/10., order=3, mode='nearest').astype(np.float)
        disps = np.stack((dX, dY, dZ)).T

        # Save the mesh as compressed numpy file for global simulation
        np.savez_compressed(outfile, xyz=disps)

        if visvtk:
            t0 = clock()
            if self.mesh_z is not None:
                vis_mesh = meshio.Mesh(self.vertices, {'triangle': self.cells}, point_data={"z":self.mesh_z,"ux":dX,"uy":dY,"uz":dZ})
            else:
                vis_mesh = meshio.Mesh(self.vertices, {'triangle': self.cells}, point_data={"ux":dX,"uy":dY,"uz":dZ})
            meshio.write(outfile+".vtk", vis_mesh)
            print("Writing VTK file {} in {}s".format(outfile+".vtk",int(clock()-t0)))

        print("Processing {} to create {} done in {}s".format(gPlates,outfile+".npz",int(clock()-t0)))

        return

    def backwardAdvection(self, dispfile=None, outfile=None, zdisp=None, lastfile=None, zforce=None, dt=1.e6):

        kk1 = 1
        kk2 = 3
        advectXYZ = self.vertices.copy()

        for k in range(len(dispfile)):

            # Advect backward in time to previous coordinates
            if k==0 :
                # Reading the last advection velocity
                vel3d = np.load(dispfile[k])
                advectXYZ -= vel3d['xyz']*dt
            else :
                advectXYZ -= nadvect*dt

            # Interpolate from backward advected coordinates
            mvtree = spatial.cKDTree(advectXYZ,leafsize=10)

            distances, indices = mvtree.query(self.vertices, k=kk1)
            if kk1 == 1:
                nzdisp = zdisp[indices]
            else:
                # Inverse weighting distance...
                weights = 1.0 / distances**2
                onIDs = np.where(distances[:,0] == 0)[0]
                nzdisp = np.sum(weights*zdisp[indices],axis=1)/np.sum(weights, axis=1)

                if len(onIDs)>0:
                    nzdisp[onIDs] = zdisp[indices[onIDs,0]]


            if lastfile is not None:
                # Previous vertical forcing file
                lastdisp = np.load(lastfile[k]+".npz")
                nzdisp += lastdisp['z']

            # Write vertical displacement at previous time:
            if k == 0 and zforce is not None:
                np.savez_compressed(outfile[k], z=nzdisp+zforce*self.timeall/dt)
            else:
                np.savez_compressed(outfile[k], z=nzdisp)
            print("Processing {}".format(outfile[k]+".npz"))

            # Reading the previous advection velocity
            if k+1 == len(dispfile):
                return

            vel3d = np.load(dispfile[k+1])
            ndisps = vel3d['xyz']

            distances, indices = self.fixtree.query(advectXYZ, k=kk2)
            nadvect = np.zeros(advectXYZ.shape)

            # Inverse weighting distance...
            weights = 1.0 / distances**2
            onIDs = np.where(distances[:,0] == 0)[0]
            nadvect[:,0] = np.sum(weights*ndisps[indices,0],axis=1)/np.sum(weights, axis=1)
            nadvect[:,1] = np.sum(weights*ndisps[indices,1],axis=1)/np.sum(weights, axis=1)
            nadvect[:,2] = np.sum(weights*ndisps[indices,2],axis=1)/np.sum(weights, axis=1)

            if len(onIDs)>0:
                nadvect[onIDs,0] = ndisps[indices[onIDs,0],0]
                nadvect[onIDs,1] = ndisps[indices[onIDs,0],1]
                nadvect[onIDs,2] = ndisps[indices[onIDs,0],2]

        return

    def compareWithMap(self, filename=None,time=None):

        paleo_struct = np.load(filename)

        self.mapZ = paleo_struct['z']
        self.mapV = paleo_struct['v']
        self.mapC = paleo_struct['c']
        self.timeall = time
        if self.ids is None:
            tree = spatial.cKDTree(self.coords,leafsize=10)
            distances, self.ids = tree.query(self.mapV, k=1)

        diff_elev = (self.mapZ-self.elev[self.ids])/time # conversion in displacement (m/yr)

        return diff_elev

    def readOutputs(self, filename=None, step=None):

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

    def plotDiffMap(self, idiff=None):

        fig = plt.figure(figsize=(10, 8))

        im = plt.imshow(np.flipud(idiff),
                   vmin=-idiff.max(), vmax=idiff.max(),
                   interpolation="bicubic",
                   cmap='RdBu')

        plt.xlabel('longitude',size=10,fontweight='bold')
        plt.ylabel('latitude',size=10,fontweight='bold')

        ax = plt.gca()

        xvals = np.arange(0,self.ilon.shape[1]+20,20)
        yvals = np.arange(0,self.ilon.shape[0]+10,10)
        xticks = np.linspace(-180,180,num=len(xvals),endpoint=True)
        yticks = -np.linspace(-90,90,num=len(yvals),endpoint=True)

        ax.set_xticks(xvals)
        ax.set_yticks(yvals)
        ax.set_xticklabels(xticks)
        ax.set_yticklabels(yticks)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.3)

        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('zMap',size=10,fontweight='bold')

        fig.tight_layout()

        return

    def computeNorms(self, map1, map2):
        # Manhattan distance is a distance metric between two points in a N dimensional vector space.
        # It is the sum of the lengths of the projections of the line segment between the points onto the
        # coordinate axes. In simple terms, it is the sum of absolute difference between the measures in
        # all dimensions of two points.

        # Normalize to compensate for exposure difference
        rng = map1.max()-map1.min()
        amin = map1.min()
        map1 = (map1-amin)*255/rng
        rng = map2.max()-map2.min()
        amin = map2.min()
        map2 = (map2-amin)*255/rng

        # Calculate the difference and its norms
        diff = map2 - map1  # Elementwise for scipy arrays

        # Manhattan norm
        n_m = sum(abs(diff))
        # # Zero norm
        # self.n_0 = norm(diff.ravel(), 0)

        # print("  + Manhattan norm: {} / per pixel: {}".format(n_m, n_m/map1.size))
        # print("  + Zero norm: {} / per pixel: {}".format(self.n_0, self.n_0*1.0/map1.size))

        return n_m

    def predictionDiff(self, deg=2., diff=None, plot=True, method='kdtree'):
        '''
        Method options are 'linear', 'nearest', 'cubic'
        '''
        t0 = clock()
        if self.ilons is None or self.deg != deg:
            self.ilons = (self.icoords[0,:])*360./3601.-180.
            self.ilats = (self.icoords[1,:])*180./1801.-90.

            # target grid to interpolate to
            ilon = np.arange(-180.,180.,deg)
            ilat = np.arange(-90.,90.,deg)
            self.ilon,self.ilat = np.meshgrid(ilon,ilat)

        # interpolate
        if method == 'kdtree':
            if self.deg != deg:
                XY = np.zeros((self.ilons.shape[0],2))
                XY[:,0] = self.ilons
                XY[:,1] = self.ilats

                pXY = np.zeros((self.ilon.size,2))
                pXY[:,0] = self.ilon.ravel()
                pXY[:,1] = self.ilat.ravel()

                self.fftree = spatial.cKDTree(XY,leafsize=10)
                self.distances, self.indices = self.fftree.query(pXY, k=3)

            weights = 1.0 / self.distances**2
            onIDs = np.where(self.distances[:,0] == 0)[0]
            self.idiff = np.sum(weights*diff[self.indices],axis=1)/np.sum(weights, axis=1)
            self.ielev = np.sum(weights*self.mapZ[self.indices],axis=1)/np.sum(weights, axis=1)

            if len(onIDs)>0:
                self.idiff[onIDs] = diff[self.indices[onIDs,0]]
                self.ielev[onIDs] = self.mapZ[self.indices[onIDs,0]]

            self.idiff = self.idiff.reshape(self.ilon.shape)
            self.ielev = self.ielev.reshape(self.ilon.shape)

        else:
            self.idiff = griddata((self.ilons,self.ilats),diff,(self.ilon,self.ilat),method='cubic')
            self.ielev = griddata((self.ilons,self.ilats),self.mapZ,(self.ilon,self.ilat),method='cubic')
            self.idiff[np.isnan(self.idiff)] = 0.
            self.ielev[np.isnan(self.ielev)] = 0.

        self.deg = deg
        type = np.zeros(self.ielev.shape)
        type[self.ielev<-1000.] = -1
        type[self.ielev>0.] = 1
        diffdata = pd.DataFrame({'lons': self.ilon.ravel(), 'lats': self.ilat.ravel(),
                                 'zMap': self.idiff.ravel(), 'type': type.ravel()})
        if plot:
            self.plotDiffMap(self.idiff)
            #self.plotDiffMap(type)

        #print("Processing done in {}s".format(int(clock()-t0)))

        return diffdata

    def similarity(self, X, Y=None, *, normalise=True, demean=True):
        '''
        Compute similarity between the columns of one or two matrices.

        Covariance: normalise=False, demean=True - https://en.wikipedia.org/wiki/Covariance

        Corrcoef: normalise=True, demean=True - https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient

        Dot product: normalise=False, demean=False

        Cosine similarity: normalise=True, demean=False - https://en.wikipedia.org/wiki/Cosine_similarity
        N.B. also known as the congruence coefficient
        https://en.wikipedia.org/wiki/Congruence_coefficient
        '''

        eps = 1.0e-5
        if Y is None:
            if X.ndim != 2:
                raise ValueError("X must be 2D!")
            Y = X

        if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must be 2D with the same first dimension!")

        if demean:
            X = X - np.mean(X, axis=0)
            Y = Y - np.mean(Y, axis=0)

        if normalise:
            # Set variances to unity
            x = np.sqrt(np.sum(X**2, axis=0)); x[x < eps] = 1.0
            y = np.sqrt(np.sum(Y**2, axis=0)); y[y < eps] = 1.0
            X = X / x; Y = Y / y
        else:
            # Or just divide by no. of observations to make an expectation
            X = X / math.sqrt(X.shape[0]); Y = Y / math.sqrt(Y.shape[0])

        return X.T @ Y

    def rv_coefficient(self, X, Y, *, normalise=True, demean=True):
        '''
        RV coefficient (and related variants) between 2D matrices, columnwise
        https://en.wikipedia.org/wiki/RV_coefficient
        RV normally defined in terms of corrcoefs, but any of the above similarity
        metrics will work
        '''
        # Based on scalar summaries of covariance matrices
        # Sxy = sim(X,Y)
        # covv_xy = Tr(Sxy @ Syx)
        # rv_xy =  covv_xy / sqrt(covv_xx * covv_yy)

        # Calculate correlations
        # N.B. trace(Xt @ Y) = sum(X * Y)
        Sxy = self.similarity(X, Y, normalise=normalise, demean=demean)
        c_xy = np.sum(Sxy ** 2)
        Sxx = self.similarity(X, X, normalise=normalise, demean=demean)
        c_xx = np.sum(Sxx ** 2)
        Syy = self.similarity(Y, Y, normalise=normalise, demean=demean)
        c_yy = np.sum(Syy ** 2)

        # And put together
        rv =  c_xy / math.sqrt(c_xx * c_yy)

        return rv

    def concordance_correlation_coefficient(self, y_true, y_pred, sample_weight=None, multioutput='uniform_average'):
        """Concordance correlation coefficient.
        The concordance correlation coefficient is a measure of inter-rater agreement.
        It measures the deviation of the relationship between predicted and true values
        from the 45 degree angle.
        Read more: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
        Original paper: Lawrence, I., and Kuei Lin. "A concordance correlation coefficient to evaluate reproducibility." Biometrics (1989): 255-268.
        Parameters
        ----------
        y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
            Ground truth (correct) target values.
        y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
            Estimated target values.
        Returns
        -------
        loss : A float in the range [-1,1]. A value of 1 indicates perfect agreement
        between the true and the predicted values.
        Examples
        --------
        >>> from sklearn.metrics import concordance_correlation_coefficient
        >>> y_true = [3, -0.5, 2, 7]
        >>> y_pred = [2.5, 0.0, 2, 8]
        >>> concordance_correlation_coefficient(y_true, y_pred)
        0.97678916827853024
        """
        cor=np.corrcoef(y_true,y_pred)[0][1]

        mean_true=np.mean(y_true)
        mean_pred=np.mean(y_pred)

        var_true=np.var(y_true)
        var_pred=np.var(y_pred)

        sd_true=np.std(y_true)
        sd_pred=np.std(y_pred)

        numerator=2*cor*sd_true*sd_pred

        denominator=var_true+var_pred+(mean_true-mean_pred)**2

        return numerator/denominator

    def nonMatching(self, diff, elevrange=100.):

        nomatch = len(np.where(abs(diff)>elevrange)[0])/diff.size

        return nomatch

    def accuracyScores(self,elevrange=100.):

        obs = self.ielev.copy()
        sim = obs - self.idiff
        regions = ['global','lands','shelf','ocean']

        landIDs = np.where(self.ielev>=0.)
        oceanIDs = np.where(self.ielev<=-1000.)
        shelfIDs = np.where(np.logical_and(self.ielev<0.,self.ielev>-1000.))

        L1 = []
        CV = []
        CR = []
        DP = []
        CS = []
        P = []
        R2 = []
        MAE = []
        RMSE = []
        LCC = []
        NN = []

        # Global
        L1.append(self.computeNorms(obs,sim))
        CV.append(self.rv_coefficient(obs, sim, normalise=False, demean=True))
        CR.append(self.rv_coefficient(obs, sim, normalise=True, demean=True))
        DP.append(self.rv_coefficient(obs, sim, normalise=False, demean=False))
        CS.append(self.rv_coefficient(obs, sim, normalise=True, demean=False))
        P.append(pearsonr(obs.ravel(), sim.ravel())[0])
        R2.append(r2_score(obs.ravel(), sim.ravel(),multioutput='variance_weighted'))
        MAE.append(mean_absolute_error(obs.ravel(), sim.ravel(), multioutput='uniform_average'))
        RMSE.append(np.sqrt(mean_squared_error(obs.ravel(), sim.ravel())))
        LCC.append(self.concordance_correlation_coefficient(obs.ravel(), sim.ravel()))
        NN.append(self.nonMatching(sim-obs, elevrange=100.))

        # Land
        Lobs = obs.copy()
        Lsim = sim.copy()
        Lobs[self.ielev<0] = 0.
        Lsim[self.ielev<0] = 0.
        land_obs = obs[landIDs].ravel()
        land_sim = sim[landIDs].ravel()
        CV.append(self.rv_coefficient(Lobs, Lsim, normalise=False, demean=True))
        CR.append(self.rv_coefficient(Lobs, Lsim, normalise=True, demean=True))
        DP.append(self.rv_coefficient(Lobs, Lsim, normalise=False, demean=False))
        CS.append(self.rv_coefficient(Lobs, Lsim, normalise=True, demean=False))
        P.append(pearsonr(land_obs.ravel(), land_sim.ravel())[0])
        L1.append(self.computeNorms(land_obs,land_sim))
        R2.append(r2_score(land_obs, land_sim,multioutput='variance_weighted'))
        MAE.append(mean_absolute_error(land_obs, land_sim, multioutput='uniform_average'))
        RMSE.append(np.sqrt(mean_squared_error(land_obs, land_sim)))
        LCC.append(self.concordance_correlation_coefficient(land_obs, land_sim))
        NN.append(self.nonMatching(land_sim-land_obs, elevrange=100.))

        # Shelf
        Sobs = obs.copy()
        Ssim = sim.copy()
        Sobs[self.ielev>0] = 0.
        Ssim[self.ielev>0] = 0.
        Sobs[self.ielev<-1000.] = 0.
        Ssim[self.ielev<-1000.] = 0.
        shelf_obs = obs[shelfIDs].ravel()
        shelf_sim = sim[shelfIDs].ravel()
        CV.append(self.rv_coefficient(Sobs, Ssim, normalise=False, demean=True))
        CR.append(self.rv_coefficient(Sobs, Ssim, normalise=True, demean=True))
        DP.append(self.rv_coefficient(Sobs, Ssim, normalise=False, demean=False))
        CS.append(self.rv_coefficient(Sobs, Ssim, normalise=True, demean=False))
        L1.append(self.computeNorms(shelf_obs,shelf_sim))
        P.append(pearsonr(shelf_obs, shelf_sim)[0])
        R2.append(r2_score(shelf_obs, shelf_sim,multioutput='variance_weighted'))
        MAE.append(mean_absolute_error(shelf_obs, shelf_sim, multioutput='uniform_average'))
        RMSE.append(np.sqrt(mean_squared_error(shelf_obs, shelf_sim)))
        LCC.append(self.concordance_correlation_coefficient(shelf_obs, shelf_sim))
        NN.append(self.nonMatching(shelf_sim-shelf_obs, elevrange=100.))

        # Ocean
        Oobs = obs.copy()
        Osim = sim.copy()
        Lobs[self.ielev>=-1000.] = 0.
        Lsim[self.ielev>=-1000.] = 0.
        ocean_obs = obs[oceanIDs].ravel()
        ocean_sim = sim[oceanIDs].ravel()
        CV.append(self.rv_coefficient(Oobs, Osim, normalise=False, demean=True))
        CR.append(self.rv_coefficient(Oobs, Osim, normalise=True, demean=True))
        DP.append(self.rv_coefficient(Oobs, Osim, normalise=False, demean=False))
        CS.append(self.rv_coefficient(Oobs, Osim, normalise=True, demean=False))
        L1.append(self.computeNorms(ocean_obs,ocean_sim))
        P.append(pearsonr(ocean_obs.ravel(), ocean_sim.ravel())[0])
        R2.append(r2_score(ocean_obs.ravel(), ocean_sim.ravel(),multioutput='variance_weighted'))
        MAE.append(mean_absolute_error(ocean_obs.ravel(), ocean_sim.ravel(), multioutput='uniform_average'))
        RMSE.append(np.sqrt(mean_squared_error(ocean_obs.ravel(), ocean_sim.ravel())))
        LCC.append(self.concordance_correlation_coefficient(ocean_obs.ravel(), ocean_sim.ravel()))
        NN.append(self.nonMatching(ocean_sim-ocean_obs, elevrange=100.))

        similarity = pd.DataFrame({'region':regions, 'covariance': CV, 'corrcoef': CR,
                                 'dotproduct': DP, 'cosine': CS,
                                 'pearson': P, 'L1': L1})

        accuracy = pd.DataFrame({'region':regions,  'RMSE': RMSE, 'R2': R2,
                         'MAE': MAE,  'LCC': LCC, 'nonmatching': NN})

        return similarity, accuracy

    def updateInput(self, input, run):

        with fileinput.FileInput(input, inplace=True, backup=str(run)) as file:
            for line in file:
                print(line.replace('_r'+str(run-1)+'_', '_r'+str(run)+'_'), end='')

        return

    def improvementModels(self, rmse1, rmse2):

        regions = ['global','lands','shelf','ocean']
        RI = (rmse1-rmse2)*100./rmse1
        fRI = pd.DataFrame({'region':regions, 'relative improvement': RI})

        return fRI
