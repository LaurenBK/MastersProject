import time
import scipy.interpolate as sip
from os import path
import sys
import h5py
from definitionsForTracing import *
plt.switch_backend('agg')

start = time.time()

# ===============================================================
#                     READ IN PARAMETERS
# ===============================================================
cmdargs = str(sys.argv)
print sys.argv
numCube = np.int(sys.argv[1])
sectionOfRays = int(sys.argv[2])
outPath = sys.argv[3]
coherenceScale = sys.argv[4]

B_Direction = check_direction(outPath)
B0 = check_B0(outPath)
eta = check_eta(outPath)

inPath = '/mnt/lustre/users/lhunt/CurrentCode/structureAlongRays/'

print "-----------------------------------"
print "--------READING PARAMETERS---------"
print "-----------------------------------"

with open(path.join(dir, outPath + "params.txt")) as f:
    content = f.readlines()
    for i in range(len(content)):
        line = content[i].split("=")
        # Running parameters
        if line[0] == "NUMCUBES":
            numCubes = int(line[1])
        elif line[0] == "LIGHTCONE_PATH":
            datPath = line[1][:-1]
        elif line[0] == "SOURCES":
            numSources = int(line[1])
        elif line[0] == "LOS":
            lineOfSight = normalise(np.array((line[1]).split(","), np.float))   
        elif line[0] == "PARALLELCHUNKSIZE":
            parallelChunkSize = int(line[1])        
        # Testing parameters
        elif line[0] == "RAYINTERVAL":
            rayInterval = float(line[1])
        elif line[0] == "SAMPLING_PARTICLE":
            particleSampleRadius = float(line[1]) 
        elif line[0] == "SAMPLING_GALAXY":
            galaxySampleRadius = float(line[1])  
        elif line[0] == "SAMPLING_HALO":
            haloSampleRadius = float(line[1])   
        elif line[0] == "ALPHATOP":
            alphaTop = float(line[1])
        elif line[0] == "ALPHABOTTOM":
            alphaBottom = float(line[1])
        elif line[0] == "PSCALEFACTOR":
            pScale = float(line[1])
        elif line[0] == "LIMITING_ATTRIBUTE":
            LA = ((line[1]).split("\n")[0]).split(',')
        elif line[0] == "LA_CUT":
            LA_cut = ((line[1]).split("\n")[0]).split(',')
            LA_cut = [float(x) for x in LA_cut]
        elif line[0] == "LIMITING_ORIGIN":
            LO = ((line[1]).split("\n")[0]).split(',')
        elif line[0] == "X0":
            voight_x0 = float(line[1])
        elif line[0] == "HWHM":
            voight_HWHM = float(line[1])

alpha = alphaTop / alphaBottom
dl = rayInterval * 1000.0  # Mpc to kpc
print "Sampling with a radius of", galaxySampleRadius,\
      "for galaxies and", haloSampleRadius, "for halos"

sys.stdout.flush()

indicesInCone = {}
with h5py.File(os.path.expanduser(
        datPath + 'indicesInCone_' + str(numCube) + '_PARTICLE_0'),
        'r') as data:
    for i in data.keys():
        indicesInCone[i] = data[i][:]
with h5py.File(os.path.expanduser(
        datPath + 'indicesInCone_' + str(numCube) + 'GALAXY'), 'r') as data:
    for i in data.keys():
        indicesInCone[i] = data[i][:]
with h5py.File(os.path.expanduser(
        datPath + 'indicesInCone_' + str(numCube) + 'HALO'), 'r') as data:
    for i in data.keys():
        indicesInCone[i] = data[i][:]
print 'Indices in Cone'
for i in indicesInCone.keys():
    print i, indicesInCone[i].shape

# ===============================================================
#                     READ IN FIELD DIRECTIONS
# ===============================================================
dlDict = {}
with h5py.File(os.path.expanduser(
        'directionOfField_' + coherenceScale + '/Bdirection_' + str(numCube) +
        '_' + str(sectionOfRays) + '_' + coherenceScale+'kpc'), 'r') as data:
    print 'field direction keys', data.keys()
    lengthOfRayHere = [len(x) for x in data.values()]
    for keys in data.keys():
        dlDict[keys] = data[keys][:]

# ===============================================================
#                     READ IN PARTICLE DATA
# ===============================================================
particlePositions, particleAttribs = readLightcone('PARTICLE',
                                                   indicesInCone,
                                                   datPath,
                                                   numCube)
minDistance = np.min(particleAttribs['distance_gas'])
maxDistance = np.max(particleAttribs['distance_gas'])
print 'Distance in range:', minDistance, maxDistance
print particleAttribs.keys(), particleAttribs.values()
for i in range(len(particleAttribs.keys())):
    print particleAttribs.keys()[i], len(particleAttribs.values()[i]),\
        'eg:', particleAttribs.values()[i][0]
print 'Max hsml', np.max(particleAttribs['hsml_gas'])
plt.hist(particleAttribs['hsml_gas'])
plt.savefig('hsml.pdf')
# ===============================================================
#                     READ IN GALAXY DATA
# ===============================================================
galaxyPositions, galaxyAttribs, galaxyGlist = readLightcone('GALAXY',
                                                            indicesInCone,
                                                            datPath,
                                                            numCube)
for i in range(len(galaxyAttribs.keys())):
    print 'Galaxy Attributes'
    print galaxyAttribs.keys()[i], len(galaxyAttribs.values()[i]), \
        galaxyAttribs.values()[i][0]

# ===============================================================
#                       READ IN HALO DATA
# ===============================================================
haloPositions, haloAttribs, haloGlist = readLightcone('HALO',
                                                      indicesInCone,
                                                      datPath,
                                                      numCube)
for i in range(len(haloAttribs.keys())):
    print 'Halo Attributes'
    print haloAttribs.keys()[i], len(haloAttribs.values()[i]), \
        haloAttribs.values()[i][0]

# ===============================================================
#                     READ IN STRUCTURE ALONG RAYS
# ===============================================================
print "-----------------------------------"
print "--------READING IN STRUCTURE-------"
print "-----------------------------------"

particleID, galaxyID, haloID, particleDist, galaxyDist, haloDist = \
    readStructure(numCube, sectionOfRays, inPath, numSources)
print particleID.keys()
print galaxyID.keys()
print haloID.keys()
print particleDist.keys()
print galaxyDist.keys()
print haloDist.keys()

# ===============================================================
print "-----------------------------------"
print "-------CALCULATING REDSHIFTS-------"
print "-----------------------------------"
# create linear interpolation function using zr.txt to get Mpcs/H -> Z
points = [line.split(" ") for line in open("zr.txt")]
zrFunc = sip.UnivariateSpline(map(float, zip(*points)[1]),
                              map(float, zip(*points)[2]), s=0)
z = zrFunc(maxDistance)
print "Max redshift is", z

# ===============================================================
#                     SEARCH THE TREE IN PARALLEL
# ===============================================================
print "-----------------------------------"
print "--------- RM CALCULATIONS ---------"
print "-----------------------------------"
rotationMeasure = []
galaxyRM = []
haloRM = []
numberHit = []
B = []
Ne = []
dl = []
proc = 0
for i in particleID.keys():
        proc += 1
        sourcekey = str(i)
        if len(particleID[sourcekey]) == 0:
            # these rays were too short to be in the current section so we
            # can ignore them
            print 'Processing ray', i, 'which is empty'
            ignoreKey = np.array([])  # Dont want to waste memory on the rays
            # that are too long
            rotationMeasure.append(ignoreKey)
            galaxyRM.append(ignoreKey)
            haloRM.append(ignoreKey)
            numberHit.append([0, 0, 0])
            B.append(ignoreKey)
            Ne.append(ignoreKey)
            dl.append(ignoreKey)
        else:
            out_q = 'NOTPARALLEL'
            print 'Processing ray', i, 'of shape', len(particleID[sourcekey])
            returned = obtainRM(particleID[sourcekey], galaxyID[sourcekey],
                                haloID[sourcekey], particleDist[sourcekey],
                                galaxyDist[sourcekey], haloDist[sourcekey],
                                particleAttribs, B0, eta, pScale, alpha,
                                dlDict[sourcekey], B_Direction, LA, LA_cut, LO,
                                galaxyAttribs, galaxyGlist, haloAttribs,
                                haloGlist, indicesInCone, out_q)
            rmVec = returned[0]
            rotationMeasure.append(rmVec)  # get the calculation's output
            galVec = returned[1]
            galaxyRM.append(galVec)
            haloVec = returned[2]
            haloRM.append(haloVec)
            numberHit.append(returned[3])
            B.append(returned[4])
            Ne.append(returned[5])
            dl.append(returned[6])

print 'lengths B,Ne', len(B), len(Ne), len(dl)
print 'lengths', [len(x) for x in B], [len(x) for x in Ne]
end = time.time()
print "total time", (end-start)/60

# ===============================================================
#                                 WRITE OUT DATA
# ===============================================================
print "-----------------------------------"
print "-----------WRITING DATA------------"
print "-----------------------------------"

loc = path.dirname(__file__)

cubeDistanceRadii = path.join(
    loc, outPath + 'cubeDistanceRadii' + str(numCube) + '_'
    + str(sectionOfRays))
rotationMeasureFile = path.join(
    loc, outPath + 'rotationMeasure' + str(numCube) + '_' + str(sectionOfRays))
particles_galaxies_halos_hitPerSourceFile = path.join(
    loc, outPath + 'particles_galaxies_halos_hitPerSource_' + str(numCube) +
    '_' + str(sectionOfRays))
NeBFile = path.join(
    loc, outPath + 'NeBprofiles' + str(numCube) + '_' + str(sectionOfRays))

print 'Files allocated'
sys.stdout.flush()

stringToWrite = str(numCube) + ',' + \
                str(np.max(particleAttribs['redshift_gas'])) + ',' + \
                str(np.max(particleAttribs['distance_gas'])) + ',' + \
                str(np.max(galaxyAttribs['radii_r200'])) + ',' + \
                str(np.max(haloAttribs['radii_r200'])) + '\n'
with open(cubeDistanceRadii, 'a') as f:
    f.write(stringToWrite)

print numberHit
numberHit = np.array(numberHit)
with h5py.File(particles_galaxies_halos_hitPerSourceFile) as hf:
    hf.create_dataset(str(numCube), data=numberHit, chunks=True,
                      compression='gzip', compression_opts=4)

rotationMeasure = np.array(rotationMeasure)
galaxyRM = np.array(galaxyRM)
haloRM = np.array(haloRM)
print "rotationMeasure", rotationMeasure.shape
print "galaxyRM", galaxyRM.shape
print "haloRM", haloRM.shape

with h5py.File(rotationMeasureFile) as hf:
        grp = hf.create_group('rotationMeasure')
        for i in range(len(rotationMeasure)):
            grp.create_dataset(str(i), data=rotationMeasure[i], chunks=True,
                               compression='gzip', compression_opts=4)
        grp = hf.create_group('rotationMeasure_galaxy')
        for i in range(len(galaxyRM)):
            grp.create_dataset(str(i), data=galaxyRM[i], chunks=True,
                               compression='gzip', compression_opts=4)
        grp = hf.create_group('rotationMeasure_halo')
        for i in range(len(haloRM)):
            grp.create_dataset(str(i), data=haloRM[i], chunks=True,
                               compression='gzip', compression_opts=4)

print 'lengths B,Ne', len(B), len(Ne)
with h5py.File(NeBFile) as hf:
    grp = hf.create_group('B')
    for i in range(len(B)):
        grp.create_dataset(str(i), data=B[i], chunks=True, compression='gzip',
                           compression_opts=4)
    grp = hf.create_group('Ne')
    for i in range(len(Ne)):
        grp.create_dataset(str(i), data=Ne[i], chunks=True, compression='gzip',
                           compression_opts=4)
    grp = hf.create_group('dl')
    for i in range(len(dl)):
        grp.create_dataset(str(i), data=dl[i], chunks=True, compression='gzip',
                           compression_opts=4)

print 'Files written'
sys.stdout.flush()
