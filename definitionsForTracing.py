import numpy as np
import matplotlib.pyplot as plt
import itertools
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FuncFormatter, MaxNLocator
# import pyfits as pyfits
from numpy import random
import astropy.coordinates.distances as acd
import astropy.units as u
import csv
import time
import multiprocessing as mp
import scipy.spatial as ssp
import scipy.spatial.distance as ssd
import scipy.interpolate as sip
import matplotlib.colors as colors
import numpy.ma as ma
import os,sys
from os import path
import h5py
from numpy import random
plt.switch_backend('agg')

#-----------------------------------
#           MATH FUNCTIONS
#-----------------------------------

def normalise(a):
        """Find normalised vector of each row"""
        return a/ np.sqrt(a.dot(a))

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2':"""
    if (v1 == v2).all():
        return 0.0
    v1_u = normalise(v1)
    v2_u = normalise(v2)
    angle = np.arccos(np.dot(v1_u, v2_u))
    if np.isnan(angle):
            return np.pi
    return angle

def quadratic(a,b,c):  #from http://stackoverflow.com/questions/15398427/solving-quadratic-equation
        d = b**2-4*a*c # discriminant

        if d < 0:
            #This equation has no real solution
            return None
        elif d == 0:
            x = (-b+math.sqrt(b**2-4*a*c))/2*a
            #This equation has one solutions: x
            return x
        else:
            x1 = (-b+math.sqrt((b**2)-(4*(a*c))))/(2*a)
            x2 = (-b-math.sqrt((b**2)-(4*(a*c))))/(2*a)
            #This equation has two solutions: x1 or x2
            return (x1,x2)

def dTheta(x0,HWHM):
    """
    A Lorenzian Profile fits this distribution, so this is the quantile function (inverse cumulative distribution) where the number that is returned is a value for dtheta. 
    """
    r = random.random()
    while r>0.993557 or r<0.006448:  #keeps the angle change between -pi and pi with less calculation
        r = random.random()
    angleChange = x0+HWHM*np.tan(np.pi*(r-0.5))
    return angleChange

def distance(x,y):
    '''Takes in positions in Mpc and returns the real distance between elements(Mpc) because hsml,radii and distances are in Mpc'''
    return np.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2+(x[2]-y[2])**2)

def spline(particleNumber,rayDist,particleAttribs):
    hsml = particleAttribs['hsml_gas'][particleNumber] #this was converted to Mpc when it was read in
    r = rayDist   #in Mpc
    q = np.absolute(r)/hsml
    #print hsml,r,q
    if q>=1:
        bit = 0
    elif 0.5 <= q < 1:
        bit = 2*(1-q)**3
    elif 0 <= q < 0.5:
        bit = 1+6*q**2*(q-1)
    else:
        print 'Error with q value:',q
    w = bit
    return w

#-----------------------------------
#           IMPORT  FUNCTIONS
#-----------------------------------

def addToDict(key,dictionary,toAdd,printCommand):
    try:
        dk = dictionary[key]
    except KeyError:
        dictionary[key] = []
        dk = dictionary[key]
    if isinstance(toAdd, dict):
        dictionary[key] = toAdd
    #    if printCommand == 'y':
     #       print 'Adding ',key,'to dictionary',len(toAdd.keys())
    else:
       toAdd = np.array(toAdd)
       dictionary[key] = toAdd
     #  if printCommand == 'y':
      #     print 'Adding ',key,'to dictionary of length',len(toAdd)

def readLightcone(objectType,indicesInCone,datPath,numCube):
    print "-----------------------------------"
    print "----------",objectType,"-----------"
    print "-----------------------------------"
    attributeFile = objectType+"_"
    positionFile = objectType+"_positions_"
    if objectType=='GALAXY':
        listFile = objectType+"AVERAGES_"
        numbersFile = objectType+'_glist_'
    if objectType=='HALO':
        listFile = objectType+"GALAXIES_"
        numbersFile = objectType+'_glist_'
    Attribs = {}
    Pos = {}
    Glist = {}
    print "Reading in particle data from lightcone..."

    if objectType=="PARTICLE":
        ind = indicesInCone[objectType+'_0']
    else:
        ind = indicesInCone[objectType]

    with h5py.File(os.path.expanduser(datPath+attributeFile+str(numCube)),'r') as data:
        names =  data.keys()
        for name in names:
            if name in ('radii_gas', 'radii_r200', 'radii_stellar','hsml_gas'):
                dat = data[name][:]
                dat = dat/1000.0    #Positions are in Mpc, so radii must be too
                addToDict(name,Attribs,dat,'y')
            elif name == 'rho_gas':
                dat = data[name][:]  #code units were read in so we need to convert from 10e10M*/kpc3 to g/cm3
                UnitLength_in_cm = 3.085678e21
                UnitMass_in_g = 1.989e43
                dat = (dat*UnitMass_in_g)/(UnitLength_in_cm**3)    #Positions are in Mpc, so radii must be too
                addToDict(name,Attribs,dat,'y')
            else:
                dat = data[name][:]
                addToDict(name,Attribs,dat,'y')

    with h5py.File(os.path.expanduser(datPath+positionFile+str(numCube)),'r') as data:
        if objectType=="PARTICLE":
            xpos = data['gpos'][:,0]
            ypos = data['gpos'][:,1]
            zpos = data['gpos'][:,2]
            addToDict('x',Pos,xpos,'y')
            addToDict('y',Pos,ypos,'y')
            addToDict('z',Pos,zpos,'y')
        else:
            xpos = data['pos'][:,0]
            ypos = data['pos'][:,1]
            zpos = data['pos'][:,2]
            addToDict('x',Pos,xpos,'y')
            addToDict('y',Pos,ypos,'y')
            addToDict('z',Pos,zpos,'y')
    if objectType == "GALAXY":
        with h5py.File(os.path.expanduser(datPath+listFile+str(numCube)),'r') as data:
            names =  data.keys()
            for number in names:
                dat = data[number][:]
                dat = dat[ind]
                addToDict(number,Glist,dat,'y')
        with h5py.File(os.path.expanduser(datPath+numbersFile+str(numCube)),'r') as data:
            names =  data.keys()
            for number in names:
                dat = data[number][:]
                addToDict(number+'_gasID',Glist,dat,'y')

    if objectType == "HALO":
        with h5py.File(os.path.expanduser(datPath+listFile+str(numCube)),'r') as data:
            names =  data.keys()
            print 'keys in this file are:',names
            for number in names:
                print number,type(data[number])
                dat = data[number]
                if  number in ['gasNumbers','starNumbers','galaxyNumbers']:
                    dat = data[number][:]
                    dat = dat[ind]
                    addToDict(number,Glist,dat,'y')
               # elif '_' in number:
                #    print 'The weird extra key',number,'has the form:',type(data[number]),'and keys',dat[:]
                else:
                    d = {}
                    for n in dat.keys():
                        key = n.split('_')[1]
                        if int(key) in ind:
                            d[key]=dat[n][:]
                    addToDict(number,Glist,d,'y')
        with h5py.File(os.path.expanduser(datPath+numbersFile+str(numCube)),'r') as data:
            names =  data.keys()
            for number in names:
                dat = data[number][:]
                addToDict(number+'_gasID',Glist,dat,'y')


    if objectType=="PARTICLE":
        print 'maximum particle x,y,z = (',max(Pos['x']),max(Pos['y']),max(Pos['z']),')'
        print 'number of redshift entries= ',len(Attribs['redshift_gas']),'and number of particles is',len(Pos['x'])
    positionData = np.array((Pos['x'],Pos['y'],Pos['z']))
    positionData = positionData.T
    print "position data shape is",positionData.shape
    if objectType=="PARTICLE":
            return positionData,Attribs
    else:
            return positionData,Attribs,Glist


def readStructure(numCube,sectionOfRays,inPath,numSources):
    particleID = {}
    particleDist = {}
    galaxyID = {}
    galaxyDist = {}
    haloID = {}
    haloDist = {}
    IDDicts = [particleID,galaxyID,haloID]
    distDicts = [particleDist,galaxyDist,haloDist]
    dataKeys = ['particle','galaxy','halo']
    print "Reading in structure data"

    with h5py.File(os.path.expanduser(inPath+'objectsAlongRay'+str(numCube)+'_'+str(sectionOfRays)),'r') as data:
        names =  dataKeys
        print 'Structure keys are:',names
        for i in range(len(names)):
            print '\n\n ------------',names[i],'------------  \n\n'
            if names[i]=='particle':
                dat = data[names[i]]
                print 'Structure subkeys keys are:',dat.keys()
                for key in dat.keys():
                    dat2 = dat[key]
                    key = str(key)
                    k = key.split('_')
                    lengthOfKeys = len(dat2.keys())/2
                    IdList = [[]]*lengthOfKeys
                    distList = [[]]*lengthOfKeys
                    for m in dat2.keys():
                       datSmall = dat2[m]
                       m = str(m)
                       key2 = m.split('_')
                       key2[1] = int(key2[1])
                       if key2[0] == 'raySection':
                          IdList[key2[1]] = list(datSmall[:])
                       else:
                          distList[key2[1]] = list(datSmall[:])
                    addToDict(k[1],IDDicts[i],IdList,'y')
                    addToDict(k[1],distDicts[i],distList,'y')
            else:
                dat = data[names[i]]
                print 'Structure subkeys keys are:',dat.keys()
                lengthOfKeys = len(dat.keys())/2
                IdList = [[]]*lengthOfKeys
                distList = [[]]*lengthOfKeys
                for key in dat.keys():
                    key = str(key)
                    k = key.split('_')
                    if k[0] == 'raySection':
                        addToDict(k[1],IDDicts[i],dat[key],'y')
                        print 'IDs added',dat[key].shape
                    else:
                        addToDict(k[1],distDicts[i],dat[key],'y')
                        print 'Distances added',dat[key].shape

    return particleID,galaxyID,haloID,particleDist,galaxyDist,haloDist

#-----------------------------------
#     SPATIAL SEARCHING FUNCTIONS
#-----------------------------------

def combineParticleIDs(length,nearList1,nearList2,nearList3):
        nearCombined = []
        for i in range(length):
            if len(nearList1[i])>0:
                if len(nearList2[i])>0:
                    if len(nearList3[i])>0:
                        temp = nearList1[i]+nearList2[i]+nearList3[i]
                    else:
                        temp = []
                elif len(nearList2[i])>0:
                    temp = nearList1[i]+nearList3[i]
                else:
                    temp = nearList1[i]
            elif len(nearList2[i])>0:
                if len(nearList3[i])>0:
                    temp = nearList2[i]+nearList3[i]
                else:
                    temp = nearList2[i]
            elif len(nearList3[i])>0:
                temp = nearList3[i]
            else:
                temp = []
            nearCombined.append(temp)
        return nearCombined

def checkDistanceToRay(nearList, ray,positions, attributes, radiusKey):
        temp = []; distBetween=[]
        for i in range(len(ray)):
            temp2 = []
            rayPiece = ray[i]
            if len(nearList[i])>0:
                npPos = positions[nearList[i]]
                for j in npPos:
                    temp2.append(distance(rayPiece,j))
                radiusList = attributes[radiusKey][nearList[i]]
                closeEnoughTest = np.greater_equal(radiusList,temp2)
                closeEnough = np.array(nearList[i])[closeEnoughTest]
                temp.append(list(closeEnough))
                distBetween.append(list(np.array(temp2)[closeEnoughTest]))
            else:
                temp.append(nearList[i])
                distBetween.append(nearList[i])
        #print 'nearlist,output',sum([len(j) for j in nearList]),sum([len(j) for j in temp])
        return temp, distBetween


#-----------------------------------
#    FIELD DIRECTION FUNCTIONS
#-----------------------------------

def findNe(lengthOfRay,particle,distances,particleAttribs):
        Ne = np.zeros(lengthOfRay)  #electron density
        for i in range(len(particle)):
            if len(particle[i])>0:
                rhoTemp = particleAttribs['rho_gas'][particle[i]]
                neTemp = particleAttribs['ne_gas'][particle[i]]
     #           print 'density,ne',np.max(rhoTemp),np.max(neTemp)
                neTemp = np.multiply(rhoTemp,neTemp)/(1.6726219e-24) # find physical electron dnesity at my position#divide by proton mass to get particles/cm3

                kernelList = np.zeros(len(particle[i]))  #the particles are modelled as blobs with a certain density falloff
                for pn in range(len(particle[i])):
                    kernelList[pn] = spline(particle[i][pn],distances[i][pn],particleAttribs)
                    neTemp = np.multiply(neTemp,kernelList)  #find dinsity at my distance from particle center
      #              print 'ne',np.max(neTemp)
                    Ne[i] = sum(neTemp)
        return Ne


#def BdirectionContribution(lengthOfRay,particle,distances,particleAttribs,voight_x0,voight_HWHM,rayInterval):
#        Ne = findNe(lengthOfRay,particle,distances,particleAttribs)
#        contributions = []
#        angles = []
#        initTheta = 2*np.pi*random.random()
#        contribution = rayInterval*np.cos(initTheta)
#        contributions.append(contribution)
#        angles.append(initTheta)
#        cFs = [0,0]
#        for i in range(lengthOfRay-1):
#            changeFlag = True
#            if i ==0:  #direction will change unless the electron density is the same in the next step
#                if Ne[i] ==0 and Ne[i+1] ==0:
#                    changeFlag = False
#                elif Ne[i]>0 and Ne[i+1]>0:
#                    changeFlag = False
#            elif i==lengthOfRay-2: #direction will change unless the electron density is the same in the previous step
#                if Ne[i] ==0 and Ne[i-1] ==0 and Ne[i+1] ==0:
#                    changeFlag = False
#                elif Ne[i]>0 and Ne[i-1]>0 and Ne[i+1]>0:
#                    changeFlag = False
#            else: #direction will change unless the electron density is the same in the next and previous step
#                if Ne[i] ==0 and Ne[i-1] ==0:
#                    changeFlag = False
#                elif Ne[i]>0 and Ne[i-1]>0:
#                    changeFlag = False
#            if changeFlag == True:
#                changeInTheta = dTheta(voight_x0,voight_HWHM)
#                cFs[0]+=1
#            else:
#                changeInTheta = 0.0
#                cFs[1]+=1
#            initTheta = initTheta+changeInTheta
#            contribution = rayInterval*np.cos(initTheta)
#            contributions.append(contribution)
#            angles.append(initTheta)
#        print cFs[0],'changed direction and',cFs[1],'directions unchanged'
#        if len(contributions)!=lengthOfRay:
#            print "Error in contribution length!"
#        print 'range of contributions',np.min(contributions),np.max(contributions)
#        return contributions
#
#def checkOtherRays(r,dlkeys,coherenceScale,rays,dlDict,particle,distances,particleAttribs):
#    """
#    This function will check the current ray against all of the nearby ones that have already been processed
#    - If there are any points in other rays near this one with positive electron densities, then the directions of the fields in those rays will be saved(matchToNearby)
#    - A mask will block the direction changes in a new direction vector so the nearby directions can be added from matchToNearby
#    """
#    rayToConsider = rays[r]
#    NeToConsider = findNe(len(rays[r]),particle,distances,particleAttribs)
#    matchToNearby = np.zeros(len(rayToConsider))
#    mask = np.ones(len(rayToConsider))
#    if np.sum(NeToConsider)==0:
#        pass  #if no elements along this ray have a positive electron density, then there will be no use in comparing field directions
#    else:
#        for d in dlkeys:
#            Ne = findNe(len(rays[d]),particle,distances,particleAttribs)
#            distBetween = ssd.cdist(rays[d],rayToConsider)
#            closeEnough = []
#            for db in range(distBetween.shape[0]):
#                closeEnoughCheck = np.where(distBetween[db,:]<coherenceScale)[0]
#                NeCheck = np.where(Ne>0)[0]
#                directionsToMatch = np.intersect1d(closeEnoughCheck,NeCheck)
#                print sum(Ne),NeCheck,closeEnoughCheck,directionsToMatch
#                closeEnough.append(closeEnoughCheck)
#            closeEnough = np.array(closeEnough)
#            for c in closeEnough:
#                matchToNearby[c] = dlDict[d][c]
#                mask[c] = 0
#        matchToNearby[NeToConsider==0] = 0.0  #since we only want to find coherent fields along a structure, it doesn't help to change the direction of a field in this ray if the electron density is 0 
#        mask[NeToConsider==0] = 1
#    return mask,matchToNearby
#
#def directionContribution(lengthOfRay,Ne):
#        global voight_x0,voight_HWHM,dl
#        contributions = []
#        angles = []
#        initTheta = 2*np.pi*random.random()
#        contribution = dl*np.cos(initTheta)
#        contributions.append(contribution)
#        angles.append(initTheta)
#        cFs = [0,0]
#        for i in range(lengthOfRay-1):
#            changeFlag = True
#            if i ==0:
#                if Ne[i] ==0 and Ne[i+1] ==0:
#                    changeFlag = False
#                elif Ne[i]>0 and Ne[i+1]>0:
#                    changeFlag = False
#            elif i==lengthOfRay-2:
#                if Ne[i] ==0 and Ne[i-1] ==0 and Ne[i+1] ==0:
#                    changeFlag = False
#                elif Ne[i]>0 and Ne[i-1]>0 and Ne[i+1]>0:
#                    changeFlag = False
#            else:
#                if Ne[i] ==0 and Ne[i-1] ==0:
#                    changeFlag = False
#                elif Ne[i]>0 and Ne[i-1]>0:
#                    changeFlag = False
#            if changeFlag == True:
#                changeInTheta = dTheta(voight_x0,voight_HWHM)
#                cFs[0]+=1
#            else:
#                changeInTheta = 0.0
#                cFs[1]+=1
#            initTheta = initTheta+changeInTheta
#            contribution = dl*np.cos(initTheta)
#            contributions.append(contribution)
#            angles.append(initTheta)
#        print cFs[0],'changed direction and',cFs[1],'directions unchanged'
#        if len(contributions)!=lengthOfRay:
#            print "Error in contribution length!"
#        print 'range of contributions',np.min(contributions),np.max(contributions)
#        return contributions

def directionSampling(lengthOfRay,Ne,B_Direction,realistic_dl,dl_length):
        """ Direction of B dotted with direction of light ray will give positive or negative effect on RM  """
        dlVec = np.full((lengthOfRay),dl_length)
        if B_Direction == "RANDOM":
                randomBit = random.normal(0,np.sqrt(0.3),lengthOfRay)
                while np.any(randomBit>1.0):
                    tooLarge = np.where(randomBit>1.0)[0]
                    for j in tooLarge:
                        randomBit[j] = random.normal(0,np.sqrt(0.3))
                dlVec = randomBit*dlVec  #normal distribution...tested this in ipython
        elif B_Direction == "REALISTIC":
                dlVec = realistic_dl
        elif B_Direction == "ALIGNED":
                dlVec = dlVec
        print B_Direction,dl_length,dlVec
        return dlVec

#-----------------------------------
#      MAGNETIC FIELD FUNCTIONS
#-----------------------------------

def outflowB(B,density,pScale,alpha):
    """ Outflow parameters from F. Stasyszyn et al (Measuring cosmic magnetic fields by rotation measure-galaxy cross-correlations in cosmological simulations)"""
    factor = (density/pScale)**alpha
    #print "shapes",B.shape, density.shape
    return B*factor

def filamentFields(lengthOfRay,particleID,distances,particleAttribs,B0,eta,pScale,alpha):
        B0 = B0*1.0e6    #faraday rotation formulaa takes magnetic field in microgauss, so we need to convert from gauss to microgauss
        #print len(particleID),len(distances)
        lengthOfRay = len(particleID)
        Ne = np.zeros(lengthOfRay)  #electron density
        outflow = np.zeros(lengthOfRay) #metallicity > 0 implies that this particle used to be part of galaxy
        density = np.zeros(lengthOfRay) #gas density at this point
        BfromLSS = np.zeros(lengthOfRay)
        BfromOutflows = np.zeros(lengthOfRay)
        for i in range(len(particleID)):
                if len(particleID[i])>0:
                        neTemp = particleAttribs['ne_gas'][particleID[i]]
                        rhoTemp = particleAttribs['rho_gas'][particleID[i]] 
          #              print 'density,ne',np.max(rhoTemp),np.max(neTemp)
                        neTemp = np.multiply(rhoTemp,neTemp)/(1.6726219e-24)  #divide by proton mass to get particles/cm3
         #               print 'ne',np.max(neTemp)
                        outflowTemp = particleAttribs['z_gas'][particleID[i]]
                        kernelList = np.zeros(len(particleID[i]))  #the particles are modelled as blobs with a certain density falloff
                        for pn in range(len(particleID[i])):
                            kernelList[pn] = spline(particleID[i][pn],distances[i][pn],particleAttribs)
        #                print 'kernel',min(kernelList),max(kernelList)
       #                 print neTemp.shape,rhoTemp.shape,kernelList.shape
                        neTemp = np.multiply(neTemp,kernelList)
                        #metallicity is defined for whole particle
                        Ne[i] = sum(neTemp)
                        outflow[i] = np.mean(outflowTemp)
                        density[i] = sum(rhoTemp)

                        if np.max(neTemp) != 0:
                            Btemp = B0*neTemp*1000000.0    # the multiplication by 100000 is necessary to pull the magnetic field back to the desired field strength
                            noOutflow = np.where(outflowTemp == 0)
                            BfromLSS_masked = Btemp[:]
                            BfromLSS_masked[noOutflow[0]] = 0 #we do this to isolate only the places in the ray where metallicity is present
                        else:  #ie, if the electron density is 0 for this ray piece - should not happen
                            Btemp = np.array([0.0])
                            BfromLSS_masked = np.array([0.0])

                        BfromLSS[i] = sum(Btemp)
                        BfromOutflows[i] = sum(outflowB(BfromLSS_masked,rhoTemp,pScale,alpha))

                else:
                        outflow[i] = 0.0
                        Ne[i] = 0.0
                        density[i] = 0.0
                        BfromLSS[i] = 0.0
                        BfromOutflows[i] = 0.0

        #print "Min/max Ne",np.min(Ne),np.max(Ne),'\n',"Min/max metallicity",np.min(outflow),np.max(outflow),'\n',"Min/max density",np.min(density),np.max(density)

        particleB = BfromLSS + BfromOutflows # normalised and scaled so that the upper limit matches that of Akahori, Ryu et al, http://iopscience.iop.org/article/10.1088/0004-637X/723/1/476/meta
        toReturn  = [particleB,Ne]
        return toReturn

#def galaxyFields(lengthOfRay,usefulGalaxies,nearGalaxies):
#         B = np.zeros(lengthOfRay)
#         Ne = 0
#         for i in usefulGalaxies:
#             inc = 180.0*random.random()-90.0      #internal rotation measure
#             PA = 180.0*random.random()-90.0
#             count = np.where(nearGalaxies==i)
#             sumCount = len(count)
#             B[i] += 1.0e-6/sumCount  # from "Magnetic Fields in Galaxies, Rainer Beck & Richard Wielebinski"
# #galaxyMagnetic(source2obs[-1],inc,PA,galaxyPos[nearGalaxies[j]],0.02,0.005)
#             Ne = 0
#         toReturn  = [B,Ne]
#         return toReturn

#-----------------------------------
#      OBJECT CHECK FUNCTIONS
#-----------------------------------

def getIndices(listToCheck,associatedDist):
        '''Makes sure that every galaxy or halo is only counted once'''
        indices = []
        distBetween = []
        for i in range(len(listToCheck)):
            if len(listToCheck[i])>0:
                indices.extend(listToCheck[i])
                distBetween.extend(associatedDist[i])
        indices = np.array(indices)
        distBetween = np.array(distBetween)
        if len(indices)>0:
            unq,unq_in = np.unique(indices,return_index=True)
            indices = unq
            distBetween = distBetween[unq_in]
        indices = indices.astype(int)
        indices = indices.T
        distBetween = distBetween.T
        return list(indices),list(distBetween)

def emptyHalo(haloID):
    global LA,LA_cut,haloGlist
    for i in range(len(LA)):
        if LA[i] == 'galaxyNumbers':
            full = np.where(haloGlist[LA[i]]>LA_cut[i])
            empty = np.where(haloGlist[LA[i]]<=LA_cut[i])
    return full,empty

def checkCentralGalaxy(nearGalaxies):
    print nearGalaxies
    global galaxyAttribs
    print 'central galaxy check:',len(nearGalaxies),galaxyAttribs['central'].shape
    print 'central galaxy check:',nearGalaxies,galaxyAttribs['central']
    centralGal = []
    for l in nearGalaxies:
        for galID in l:
            if galaxyAttribs['central'][galID]==1:
                centralGal.append(galID)
    return centralGal

def checkUseful(indices,objectType,LA,LA_cut,LO,attributes):
    """
    - Indices come from the nearest neighbour searches
    - Attribs of halo or galaxy
    - List, either halo or galaxy
    - LA = limiting attribute (list of strings)
    - LA_cut = vlaue of limit that results must be greater than (list of floats)
    - LO = origin of the limiting factor, either galaxy or halogalaxy (list of strings)
    """
    useful = {}
    if objectType == 'GALAXY':
        for i in range(len(LO)):
            if LO[i] == 'galaxy':
                attributesList = attributes[LA[i]]
                print LA_cut,i
                print LA_cut[i]
                if type(attributesList[0]) == 'numpy.int64' or 'numpy.float64':
                    useful[LA[i]] = indices[np.where(attributes[LA[i]][indices]>LA_cut[i])[0]]
                elif type(attributesList[0]) == 'numpy.ndarray':
                    useful[LA[i]] = indices[np.where(attributes[LA[i]][0][indices]>LA_cut[i])[0]]
                else:
                    print "Weird errors with attributes;", attributesList, type(attributesList[0])
        for i in useful.keys():
            print 'number of galaxies that can be used with key',i,'is:',len(useful[i])
        usefulList = np.intersect1d(useful.values()[0],useful.values()[1])
    elif objectType == 'HALO':
        for i in range(len(LO)):
            if LO[i] == 'halogalaxy':
                try:
                    useful[LA[i]] = indices[np.where(attributes[LA[i]][indices]>LA_cut[i])[0]]
                except:
                    useful[LA[i]] = indices[np.where(attributes[LA[i]][0][indices]>LA_cut[i])[0]]
                usefulList = useful[LA[i]]
        for i in useful.keys():
            print 'number of halos that can be used with key',i,'is:',len(useful[i])
    return usefulList

def overlapping(indicesInCone,particleID,distances,usefulIndices,Glist):
         """
         Find all the particles that the ray intersected, and remove them from the nearParticle list
         """
         gasIndices = indicesInCone['PARTICLE_0']
         if len(usefulIndices) ==0:
             particleID_updated = particleID
             distances_updated = distances
         else:
             particleID_updated = [[]]*len(particleID)
             distances_updated = [[]]*len(particleID)
             print len(distances_updated)
             for i in usefulIndices:
                 gasIDs = Glist[str(i)+'_gasID']
                 overlap = np.where(gasIDs == gasIndices)[0] #find out which gas particles in the cone are in this halo
                 if len(overlap) == 0:    #if there is no overlap, then we do not need to worry about eliminating intersecting particles
                     particleID_updated = particleID
                     distances_updated = distances
                 else:
                     print len(gasIndices),len(gasIDs),sum([len(i) for i in particleID])
                     for j in range(len(particleID)):
                         if  len(particleID[j])>0:
                             particle_temp = []
                             dists_temp = []
                             for k in range(len(particleID[j])):
                                 if particleID[j][k] in overlap:
                                     pass
                                 else:
                                     if particleID[j][k] in particle_temp:
                                         pass
                                     else:
                                         particle_temp.append(particleID[j][k])
                                         dists_temp.append(distances[j][k])
                             particleID_updated[j].extend(particle_temp)
                             distances_updated[j].extend(dists_temp)
                         else:
                             pass
         return particleID_updated,distances_updated

#-----------------------------------
#         MAIN FUNCTIONS
#-----------------------------------

def rayTrace(source2obs,args,out_q):
        rayInterval, particleTreeSmall,particleTreeMed,particleTreeLarge,galaxyTree,haloTree,galaxySampleRadius,haloSampleRadius,particlePositions, particleAttribs, galaxyPositions, galaxyAttribs, haloPositions, haloAttribs,galaxyGlist, haloGlist = args

        printString = ''
        printString += "-----------------------------------\n"
        printString += "-----------TRACING RAY-------------\n"
        printString += "-----------------------------------\n"

        lenS20 = len(source2obs)
        #Search the tree
#        inputList = [[particlePositions[i],particleAttribs['hsml_gas'][i]] for i in range(len(particleAttribs['hsml_gas']))]
#        originalNumberParticles = np.apply_along_axis(searchPerRadius,1,inputList,str(source2obs[0]))  

        nearParticlesSmall = particleTreeSmall.query_ball_point(source2obs,1.0)  #gives list of particles close to ray
        nearParticlesMed = particleTreeMed.query_ball_point(source2obs,1.0)
        nearParticlesLarge = particleTreeLarge.query_ball_point(source2obs, 1.0)

        printString += 'small,medium,large particles before distance check '+str(sum([len(j) for j in nearParticlesSmall]))+','+str(sum([len(j) for j in nearParticlesMed]))+','+str(sum([len(j) for j in nearParticlesLarge]))+'\n'
        originalNumberParticles =sum([sum([len(j) for j in nearParticlesSmall]),sum([len(j) for j in nearParticlesMed]),sum([len(j) for j in nearParticlesLarge])])


        nearParticlesSmall,distBetSmall = checkDistanceToRay(nearParticlesSmall,source2obs,particlePositions, particleAttribs, 'hsml_gas')
        nearParticlesMed,distBetMed = checkDistanceToRay(nearParticlesMed,source2obs,particlePositions, particleAttribs, 'hsml_gas')
        nearParticlesLarge,distBetLarge = checkDistanceToRay(nearParticlesLarge,source2obs,particlePositions, particleAttribs, 'hsml_gas')

        printString +=  'small,medium,large particles after check '+str(sum([len(j) for j in nearParticlesSmall]))+','+str(sum([len(j) for j in nearParticlesMed]))+','+str(sum([len(j) for j in nearParticlesLarge]))+'\n'
            
        nearParticles = combineParticleIDs(lenS20,nearParticlesSmall,nearParticlesMed,nearParticlesLarge)
        particleDist = combineParticleIDs(lenS20,distBetSmall,distBetMed,distBetLarge)
        print len(nearParticles),len(particleDist)
        nearGalaxies = galaxyTree.query_ball_point(source2obs, galaxySampleRadius)
        nearHalos = haloTree.query_ball_point(source2obs, haloSampleRadius)
        printString +=  'initial numbers of galaxies and halos '+str(sum([len(j) for j in nearGalaxies]))+','+str(sum([len(j) for j in nearHalos]))+'\n'

        nearGalaxies,distBetGal = checkDistanceToRay(nearGalaxies,source2obs,galaxyPositions, galaxyAttribs, 'radii_r200')
        nearHalos,distBetHal = checkDistanceToRay(nearHalos,source2obs,haloPositions, haloAttribs, 'radii_r200')


        nearGalaxies = np.array(nearGalaxies)
        distBetGal = np.array(distBetGal)
        nearHalos = np.array(nearHalos)
        distBetHal = np.array(distBetHal)

        #Get a set of the halos and galaxies to avoid duplicates
        galaxyIndices,galaxyDist = getIndices(nearGalaxies,distBetGal)
        haloIndices,haloDist = getIndices(nearHalos,distBetHal)

        printString +=  'secondary numbers of galaxies and halos (actually close enough to ray) '+str(len(galaxyIndices))+','+str(len(haloIndices))+'\n'

        numberHitPerSource = [originalNumberParticles,len(galaxyIndices),len(haloIndices)]
        print printString
        toReturn = [nearParticles,particleDist,galaxyIndices,galaxyDist,haloIndices,haloDist,numberHitPerSource]

        if out_q == 'NOTPARALLEL':
            return toReturn
        else:
            out_q.put(toReturn)

def BdirectionFromTree(rays,coherenceLength,particleTreeSmall,particleTreeMed,particleTreeLarge,particlePositions,particleAttribs,particleID,particleDist,voight_x0,voight_HWHM,rayInterval):
    """
    Alternative way to assign directions by looking for particles within cooherence length and trying to align direction with the structure
   """
    print [sum([len(i) for i in r]) for r in particleID.values()]
    dlDict = {}
    for r in rays.keys():
        if len(rays[r])==0:
            dlDict[r] = []
        else:
            NeToConsider = findNe(len(rays[r]),particleID[r],particleDist[r],particleAttribs)        
            nearParticlesSmall = particleTreeSmall.query_ball_point(rays[r],coherenceLength)  #gives list of particles close to ray
            nearParticlesMed = particleTreeMed.query_ball_point(rays[r],coherenceLength)
            nearParticlesLarge = particleTreeLarge.query_ball_point(rays[r],coherenceLength)
            nearParticles = combineParticleIDs(len(rays[r]),nearParticlesSmall,nearParticlesMed,nearParticlesLarge)
            dlVec = [0]*len(rays[r])
            try:
                print 'nearParticles',len(nearParticles),len(nearParticles[0])
            except:
                pass
            sys.stdout.flush()
            for i in range(len(rays[r])):
                if NeToConsider[i] == 0:
                     dlVec[i] = 0
                     initTheta = 2*np.pi*random.random()
                    #if i ==0:
                    #    initTheta = 2*np.pi*random.random()
                    #    dlVec[0] = rayInterval*np.cos(initTheta)
                    #else:
                    #    if NeToConsider[i-1] == 0:
                    #        dlVec[i] = dlVec[i-1]
                    #    else:
                    #        changeInTheta = dTheta(voight_x0,voight_HWHM)
                    #        initTheta = initTheta+changeInTheta
                    #        dlVec[i] = rayInterval*np.cos(initTheta)
                else:
                    if len(nearParticles[i])==0:
                        changeInTheta = dTheta(voight_x0,voight_HWHM)
                        initTheta = initTheta+changeInTheta
                        dlVec[i] = rayInterval*np.cos(initTheta)
                    else:
                    #put in on 16/7/17
                        if len(nearParticles[i])>1e6:
                            percent01 = int(0.001*len(nearParticles[i]))
                            nearParticlesTemp = random.choice(nearParticles[i],size = percent01)
                        elif len(nearParticles[i])>1e5:
                            percent1 = int(0.01*len(nearParticles[i]))
                            nearParticlesTemp = random.choice(nearParticles[i],size = percent1)
                        elif len(nearParticles[i])>1e4:
                            percent10 = int(0.1*len(nearParticles[i]))
                            nearParticlesTemp = random.choice(nearParticles[i],size = percent10)
                    #------------------
                        vectors = particlePositions[nearParticlesTemp]-rays[r][i]
                        #print len(particlePositions[nearParticles[i]]), particlePositions[nearParticles[i]],vectors
                        normedVectors = np.apply_along_axis(normalise,0,np.array(vectors))
                        resultant_vec = [sum(normedVectors[:,0]),sum(normedVectors[:,1]),sum(normedVectors[:,1])]
                        normed_resultant_vec = normalise(np.array(resultant_vec))
                        angularContribution = angle_between(normed_resultant_vec,rays[r][-1])
                        dlVec[i] = rayInterval*np.cos(angularContribution)
        #                print 'angular contribution from LSS=',angularContribution, 'and resultant dl=',dlVec[i]
            dlDict[r] = dlVec
            print 'Length of directions for ray',r,'is',len(dlVec),len(rays[r])
    return dlDict

def obtainRM(particleID,galaxyID,haloID,particleDist,galaxyDist,haloDist,particleAttribs,B0,eta,pScale,alpha,realistic_dl,B_Direction,LA,LA_cut,LO,galaxyAttribs,galaxyGlist,haloAttribs,haloGlist,indicesInCone,out_q):
        """
        - Adds electron density and metallicity contributions along ray length
        - Calculates magnetic field
        - Ramdomises magnetic field direction
        - Returns Rotation Measure vector
        """
#        print distance((2.356467515040207672e3,4.500005498139384486e2,7.221242449709170614e+02),source2obs[-1])

        printString = ''
        printString += "-----------------------------------\n"
        printString += "---------PROCESSING RAY------------\n"
        printString += "-----------------------------------\n"

        lenS2O = len(particleID)

#        centralGalaxies = checkCentralGalaxy(galaxyID)  #galaxies that are central to their halos

#        fullHalos,emptyHalos = emptyHalo(haloID) #Different types of halos get different treatment

        print 'len PID before overlaps',len(particleID)
        #Clean up particle list so no double-sampling

        #--------ELIMINATE GALAXY OVERLAPS--------# 

        if len(galaxyID)>0:
            usefulGalaxies = checkUseful(galaxyID,'GALAXY',LA,LA_cut,LO,galaxyAttribs)
            particleID,particleDist = overlapping(indicesInCone,particleID,particleDist,usefulGalaxies,galaxyGlist)

        else:
            usefulGalaxies = galaxyID
        #--------ELIMINATE HALO OVERLAPS--------#       

        if len(haloID)>0:
            usefulHalos = checkUseful(haloID,'HALO',LA,LA_cut,LO,haloGlist)
            print [len(i) for i in particleID[:10]]
            particleID,particleDist = overlapping(indicesInCone,particleID,particleDist,usefulHalos,haloGlist) 
            print [len(i) for i in particleID[:10]]
        else:
            usefulHalos = haloID

        #Keep numbers for stats
        numberParticles = sum([len(j) for j in particleID])
        numberGalaxies = len(galaxyID) #different because galaxy and halo B calculations should only be done once each per source because of nature of calculation
        numberHalos = len(haloID)

        #B, Ne from gas particles
        print 'len PID after overlaps',len(particleID)
        filamentData = filamentFields(lenS2O,particleID,particleDist,particleAttribs,B0,eta,pScale,alpha)

        B = filamentData[0]
        Ne = filamentData[1]
        print "Ne size",len(Ne),'length ray',lenS2O
        dl = directionSampling(lenS2O,Ne,B_Direction,realistic_dl,10)
        print 'max dl',np.max(dl)
        print 'max B %.3e microGauss'%np.max(B)
        print 'max ne %.3e'%np.max(Ne)
        RM = 811.9*Ne*B*dl  #ray interval is given in mpc but we need it in kpc
        RM= np.nan_to_num(RM)
        print 'max RM',np.max(RM)
        galaxyData = usefulGalaxies  #list of galaxies close to the ray
        haloData = usefulHalos     #list of halos close to the ray
        numHit = [numberParticles,len(usefulGalaxies),len(usefulHalos)]
        print max(B),max(Ne)
        toReturn = (RM,galaxyData,haloData,numHit,B,Ne,dl)

        printString += str(numberParticles)+" particles, "+str(len(usefulGalaxies))+" galaxies, and "+str(len(usefulHalos))+" halos have been found near this LOS.\n"
        printString += "-----------------------------------\n"
        print printString

        if out_q == 'NOTPARALLEL':
            return toReturn
        else:
            out_q.put(toReturn)

#-----------------------------------
#         LIGHTRAY FUNCTIONS
#-----------------------------------

def lightRay(startDistance,piece,vectorPc,i,out_q):
    """
    Takes in the source positions all at once and returns vectors along the LOS from observer to source 
    """
    pcs = np.linspace(0,startDistance,piece+1)#[:, np.newaxis]   #array to use to find vectors to sources
    a = np.multiply(vectorPc.reshape(3,1),pcs)   # construct the light ray
    if out_q == 'NOTPARALLEL':
        return a
    else:
        out_q.put(a)

#-----------------------------------
#         OTHER FUNCTIONS
#-----------------------------------

def printExtremes(x):
        print np.min(x),np.max(x)

def chunks(l, n):  #http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def replaceLine(file_name, parameter, newNumber):
        """
        Replacing some parameters in the params.txt file to reflect the lightcone being used
        """
        lines = open(file_name,'r').readlines()
        for i in range(len(lines)):
                words = content[i].split("=")
                if words[0] == parameter:
                        lines[i] = parameter+"="+str(newNumber)+"\n"
        out = open(file_name,'w')
        out.writelines(lines)
        out.close()

#-----------------------------------
#         SYSARG FUNCTIONS
#-----------------------------------

def checkDirection(path):
    if 'REALISTIC' in path:
      B_Direction = 'REALISTIC'
    elif 'ALIGNED' in path:
      B_Direction = 'ALIGNED'
    elif 'RANDOM' in path:
      B_Direction = 'RANDOM'
    return B_Direction

def checkB0(path):
    if '06' in path:
      B0 = 1.0e-6
    if '07' in path:
      B0 = 1.0e-7
    if '08' in path:
      B0 = 1.0e-8
    if '09' in path:
      B0 = 1.0e-9
    return B0

def checkEta(path):
    if '0.4' in path:
      eta = 0.4
    if '0.5' in path:
      eta = 0.5
    if '0.6' in path:
      eta = 0.6
    return eta


