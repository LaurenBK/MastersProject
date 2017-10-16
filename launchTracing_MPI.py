import shutil
from mpi4py import MPI
import numpy as np
import os
import sys
import scipy
from scipy.interpolate import UnivariateSpline
import numpy as np
import math
import os
import time
import platform
import glob

z=0
box=0
axis=0
toStart=0
numCubes=0
running=0
zBins=0
doDM= False
pixElems = 11
pixels = 0
zBins = 0
genSources = 'NO'

def intro():

    print "-----------------------------------------------------------"
    print "                Ray Tracing tool v0.1                       "
    print "-----------------------------------------------------------"

    print "Checking files..."

    if not os.path.isfile("params.txt"):
        print "ERROR, params.txt file not found!"
        sys.exit()

    if not os.path.isfile("combineContributions.py"):
        print "ERROR, combineContributions.py file not found!"
        sys.exit()

    print "Done."
    print ""
    time.sleep(2)

def params():

    global numCubes,numSources,genSources,outPath,raypieces
    print "Checking parameters..."

    #read params
    with open("params.txt") as f:
        content = f.readlines()

        for i in range(len(content)):
            line = content[i].split("=")

            if(line[0] == "NUMCUBES"):
                numCubes = int(line[1])

            elif(line[0] == "SOURCES"):
                numSources = int(line[1])

            elif(line[0] == "B_DIRECTION"):
                B_Direction = str(line[1])

            elif(line[0] == "B0"):
                B0 = float(line[1])

            elif(line[0] == "ETA"):
                eta = float(line[1])

            elif(line[0] == "RAYPIECES"):
                raypieces = int(line[1])

            elif(line[0] == "GENERATE_SOURCES"):
                genSources = line[1].split('\n')[0]

            elif(line[0] == "OUTPUT_PATH"):
                outPath = line[1][:-1]

    if(numCubes == 0):
        print "Could not find NUMCUBES in param.txt"
        sys.exit()

    print "Done."
    print ""
    time.sleep(2)

def clean():
    #empty the offsets file
    open("cube_offsets.txt", 'w').close()

    #delete old data files
    for the_file in os.listdir(outPath):
        file_path = os.path.join(outPath, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception, e:
            print e

    #delete old log files
    for the_file in os.listdir(outPath+'/logs/'):
        file_path = os.path.join(outPath+'/logs/', the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception, e:
            print e

def makeSources():

    dir = os.path.dirname(__file__)
    filename = os.path.join(dir, 'logs/sources.log')
    os.system("python sourcesFromCone.py > " +filename )
    os.system("python makeLightRay.py")
    print "Sources have been generated."

sys.stdout.flush()


#----------------------------------------------
cmdargs = str(sys.argv)
print cmdargs
print sys.argv
startCube = np.int(sys.argv[1])
endCube = np.int(sys.argv[2])
direction = sys.argv[3]
B0 = sys.argv[4]
exponent = sys.argv[5]
coherenceScale = sys.argv[6]

comm = MPI.COMM_WORLD
rank = comm.rank
num_procs = comm.size
wt = MPI.Wtime()

comm.Barrier()

arg2 = str(rank)
script_path = "/mnt/lustre/users/lhunt/CurrentCode/structureToRM.py" # path to external python script

#check versions and files
intro()

#check params
params()
outPath = outPath[:-1]+direction+'_'+str(B0)+'_'+str(exponent)+'_'+str(coherenceScale)

print 'number ray pieces',raypieces
print 'outPath',outPath
print os.path.exists(outPath)
print os.path.exists(outPath+"/logs")

if not os.path.exists(outPath):
        try:
            os.makedirs(outPath)
        except:
            print 'Folder seems to exist already...'

if not os.path.exists(outPath+"/logs"):
    try:
        os.makedirs(outPath+"/logs")
    except:
        pass

if startCube == 0:
    #wipe old data
    clean()
    
    shutil.copyfile('params.txt', outPath+'/params.txt')
    if endCube == 0:
        N = raypieces
    else:
        N = endCube*raypieces  #one task per section of rays
elif startCube >0:
    N = (endCube-startCube+1)*raypieces #number of independent things to process
else: 
    pass
print 'rank',rank
print 'number of processors',num_procs
print 'number of independent things to process',N

with open('rayTracingInfo.txt','a') as f:
    l = str(str(startCube+N)+','+str(rank)+','+str(num_procs)+'\n')
    f.write(l)

# let rank 0 do chunk+rem number of items, the others just do chunk amounts
if N>num_procs:
    chunk = int(np.floor(float(N)/float(num_procs)))
    rem = N - chunk*num_procs
else:
    chunk = 1
    rem = 0

comm.Barrier()
if rank==0:
   print "Chunk/rem: ", chunk, rem
else:
   pass
comm.Barrier()

# Calculate which rank does what list of items
comm.Barrier()
if rank==0:
   start = startCube
   end = start+ chunk + rem
   tasklist = range(start,end)
   print "Tasklist for rank " + str(rank), tasklist
else:
   start = startCube + (chunk + rem) + (rank-1)*chunk
   end = start + chunk
   print 'start/end',start,end
   tasklist = range(start,end)
   print "Tasklist for rank " + str(rank), tasklist
comm.Barrier()
sys.stdout.flush()

# proceed with the main calculation
dir = os.path.dirname(__file__)
comm.Barrier()
                               
# proceed with the main calculation
dir = os.path.dirname(__file__)
comm.Barrier()
if rank==0:
   for i in tasklist:
      print i, startCube
      cubeToRun = startCube + (i-startCube)/raypieces
      raySectionToRun = (i-startCube)%raypieces
      if not os.path.exists(outPath+'/rotationMeasure'+str(cubeToRun)+'_'+str(raySectionToRun)):
        print 'With starting cube:',startCube,'we process cube',cubeToRun,'with subsection',raySectionToRun
        logfile = os.path.join(dir, outPath+'/logs/tracing_'+direction+'_'+str(cubeToRun)+'_'+str(raySectionToRun)+'.log')
        exec_command = "python " + script_path + " " + str(cubeToRun) + ' '+ str(raySectionToRun) + ' ' + outPath + '/ ' + coherenceScale +  " > " +logfile
        os.system(exec_command)
      else:
        print 'rotationMeasure'+str(cubeToRun)+'_'+str(raySectionToRun),'exists, so skipping this one'
else:
   for i in tasklist:
      print i, startCube,(startCube + (chunk + rem) + (rank-1)*chunk) ,i-startCube
      cubeToRun = startCube + (i-startCube)/raypieces
      #cubeToRun = (startCube + (chunk + rem) + (rank-1)*chunk) + (i-startCube)/raypieces
      raySectionToRun = (i-startCube)%raypieces
      if not os.path.exists(outPath+'/rotationMeasure'+str(cubeToRun)+'_'+str(raySectionToRun)):
        print 'With starting cube:',startCube,'we process cube',cubeToRun,'with subsection',raySectionToRun
        logfile = os.path.join(dir, outPath+'/logs/tracing_'+direction+'_'+str(cubeToRun)+'_'+str(raySectionToRun)+'.log')
        exec_command = "python " + script_path + " " + str(cubeToRun) + ' '+ str(raySectionToRun) + ' ' + outPath + '/ ' + coherenceScale +  " > " +logfile
        os.system(exec_command)
      else:
        print 'rotationMeasure'+str(cubeToRun)+'_'+str(raySectionToRun),'exists, so skipping this one'
comm.Barrier()

wt = MPI.Wtime() - wt
comm.Barrier()
if rank == 0:
   print "Elapsed time (seconds):", wt
else:
   pass
comm.Barrier()
sys.exit()
