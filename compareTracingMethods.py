import numpy as np
import tracingMethods as tm
import matplotlib.pyplot as plt
import h5py
import seaborn as sns
import time
from memory_profiler import memory_usage

sns.set_context("paper")

with h5py.File('wholeCube1Positions', 'r') as data:
    print data.keys()
    wholeCube = data['pos'][:]
print wholeCube.shape
print wholeCube[0]


def light_ray(start_distance, piece, vector_pc):
    """
    Takes in the source positions all at once and returns vectors along the
    LOS from observer to source
    """
    # array to use to find vectors to sources
    pcs = np.linspace(0, start_distance, piece+1)
    # construct the light ray
    a = np.multiply(vector_pc.reshape(3, 1), pcs)
    return a 


def normalise(a):
    """Find normalised vector of each row"""
    return a/ np.sqrt(a.dot(a))


Ids = np.random.choice(range(len(wholeCube)), 10)

x = wholeCube[:, 0][Ids]
y = wholeCube[:, 1][Ids]
z = wholeCube[:, 2][Ids]

sourcePositions = np.array([np.max(wholeCube[:, 0]),
                            np.max(wholeCube[:, 1]),
                            np.max(wholeCube[:, 2])])
print sourcePositions
# absolute distance in Mpc
absD = np.linalg.norm(sourcePositions)
# find the unit vector for each source
normSourceVector = normalise(sourcePositions)

vectorPiece = np.copy(normSourceVector)
# divide by 10 to get number of checkpoints along ray
pieces = np.divide(absD, 10).astype(np.int64)
ray = light_ray(absD, pieces, vectorPiece).T
radius = 100.0
numParticles = [10, 100, 1000, 10000, 100000, 1000000]

makeTreeMem = []
searchTreeMem = []
bruteMem = []
broadcastMem = []

print 'Testing trees'

timesTree = []
timesBuildingTree = []
for i in numParticles:
    Ids = np.random.choice(range(len(wholeCube)), i)
    t0 = time.time()
    tree = tm.makeTree(wholeCube[Ids])
    makeTreeMem.append(max(memory_usage((tm.makeTree, (wholeCube[Ids],)))))
    t1 = time.time()
    timesBuildingTree.append(t1-t0)
    timesTree.append(tm.treeMethod(tree, ray, radius))
    searchTreeMem.append(max(memory_usage((tm.treeMethod,
                                          (tree, ray, radius,)))))

print makeTreeMem, searchTreeMem

print 'Testing brute'
timesBrute = []
for i in numParticles:
    Ids = np.random.choice(range(len(wholeCube)), i)
    space = wholeCube[Ids]
    timesBrute.append(tm.bruteForce(space, len(space), ray, radius))
    bruteMem.append(max(memory_usage((tm.bruteForce,
                                     (space, len(space), ray, radius,)))))

print 'Testing broadcast'
timesBroadcast = []
for i in numParticles:
    Ids = np.random.choice(range(len(wholeCube)), i)
    space = wholeCube[Ids]
    timesBroadcast.append(tm.broadcastMethod(ray, space, radius))
    broadcastMem.append(max(memory_usage((tm.broadcastMethod,
                                         (ray, space, radius,)))))

print 'Done testing'

print timesTree
print timesBrute
print timesBroadcast

plt.figure(figsize=(6, 6))
plt.plot(numParticles, timesBuildingTree, label ='Building tree')
plt.plot(numParticles, timesTree, label ='TM')
plt.plot(numParticles, timesBrute, label ='BFM')
plt.plot(numParticles, timesBroadcast, label ='BM')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number Particles')
plt.ylabel('Time (s)')

plt.savefig('compareTracingMethods.pdf')

plt.figure(figsize=(6, 6))
plt.plot(numParticles, makeTreeMem, label='Building tree')
plt.plot(numParticles, searchTreeMem, label='TM')
plt.plot(numParticles, bruteMem, label='BFM')
plt.plot(numParticles, broadcastMem, label='BM')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number Particles')
plt.ylabel('Memory usage (MiB)')

plt.savefig('compareTracingMethodsUsage.pdf')

