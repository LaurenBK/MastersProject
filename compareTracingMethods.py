import numpy as np
import tracingMethods as tm
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial import distance
from numpy import random
import sys,os,scipy
from scipy import stats as ss
from matplotlib import rc
from numpy import random
import h5py
import seaborn as sns
#mpl.use('Agg')
import scipy.spatial as ssp
import time
from memory_profiler import memory_usage

sns.set_context("paper")#sns.set_style("whitegrid")
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# ## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})

with h5py.File('wholeCube1Positions','r') as data: 
    print data.keys()
    wholeCube = data['pos'][:]
print wholeCube.shape
print wholeCube[0]


def lightRay(startDistance,piece,vectorPc):
    """
    Takes in the source positions all at once and returns vectors along the LOS from observer to source 
    """
    pcs = np.linspace(0,startDistance,piece+1)#[:, np.newaxis]   #array to use to find vectors to sources
    a = np.multiply(vectorPc.reshape(3,1),pcs)   # construct the light ray
    return a 

def normalise(a):
	"""Find normalised vector of each row"""
	return a/ np.sqrt(a.dot(a))

Ids = np.random.choice(range(len(wholeCube)),10)

# xIds = np.where(wholeCube[:,0]<lim)[0]
# yIds = np.where(wholeCube[:,1]<lim)[0]
# zIds = np.where(wholeCube[:,2]<lim)[0]
# Ids = np.intersect1d(xIds,yIds)
# Ids = np.intersect1d(Ids,zIds)
x = wholeCube[:,0][Ids]
y = wholeCube[:,1][Ids]
z = wholeCube[:,2][Ids]

sourcePositions = np.array([np.max(wholeCube[:,0]),np.max(wholeCube[:,1]),np.max(wholeCube[:,2])])
print sourcePositions
absD = np.linalg.norm(sourcePositions)  #absolute distance in Mpc
normSourceVector = normalise(sourcePositions)  #find the unit vector for each source

vectorPiece = np.copy(normSourceVector)  #deep copy
pieces = np.divide(absD,10).astype(np.int64) #divide by 10 to get number of checkpoints along ray

ray = lightRay(absD,pieces,vectorPiece).T
radius = 100.0
numParticles = [10,100,1000,10000,100000,1000000]

makeTreeMem = []
searchTreeMem = []
bruteMem = []
broadcastMem = []

print 'Testing trees'

timesTree = []
timesBuildingTree = []
for i in numParticles:
    Ids = np.random.choice(range(len(wholeCube)),i)
    t0 = time.time()
    tree = tm.makeTree(wholeCube[Ids])
    makeTreeMem.append(max(memory_usage((tm.makeTree,(wholeCube[Ids],)))))
    t1 = time.time()
    timesBuildingTree.append(t1-t0)
    timesTree.append(tm.treeMethod(tree, ray, radius))
    searchTreeMem.append(max(memory_usage((tm.treeMethod,(tree, ray, radius,)))))

print makeTreeMem,searchTreeMem

print 'Testing brute'

timesBrute = []
for i in numParticles:
    Ids = np.random.choice(range(len(wholeCube)),i)
    space = wholeCube[Ids]
    timesBrute.append(tm.bruteForce(space,len(space), ray, radius))
    bruteMem.append(max(memory_usage((tm.bruteForce,(space,len(space), ray, radius,)))))

print 'Testing breadcast'

timesBroadcast = []
for i in numParticles:
    Ids = np.random.choice(range(len(wholeCube)),i)
    space = wholeCube[Ids]
    timesBroadcast.append(tm.broadcastMethod(ray,space,radius))
    broadcastMem.append(max(memory_usage((tm.broadcastMethod,(ray,space,radius,)))))

print 'Done testing'

print timesTree
print timesBrute
print timesBroadcast

plt.figure(figsize=(6,6))
plt.plot(numParticles,timesBuildingTree,label ='Building tree')
plt.plot(numParticles,timesTree,label ='TM')
plt.plot(numParticles,timesBrute,label ='BFM')
plt.plot(numParticles,timesBroadcast,label ='BM')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number Particles')
plt.ylabel('Time (s)')

plt.savefig('compareTracingMethods.pdf')

plt.figure(figsize=(6,6))
plt.plot(numParticles,makeTreeMem,label ='Building tree')
plt.plot(numParticles,searchTreeMem,label ='TM')
plt.plot(numParticles,bruteMem,label ='BFM')
plt.plot(numParticles,broadcastMem,label ='BM')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number Particles')
plt.ylabel('Memory usage (MiB)')

plt.savefig('compareTracingMethodsUsage.pdf')

#plt.show()

