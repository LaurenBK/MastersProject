import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import math
import h5py
from numpy import random
plt.switch_backend('agg')

# -----------------------------------
#           MATH FUNCTIONS
# -----------------------------------


def normalise(a):
    """Find normalised vector of each row"""
    return a / np.sqrt(a.dot(a))


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


def quadratic(a, b, c):
    """
    from http://stackoverflow.com/questions/15398427/solving-quadratic-equation
    """
    # discriminant
    d = b**2-4*a*c

    if d < 0:
        # This equation has no real solution
        return None
    elif d == 0:
        x = (- b + math.sqrt(b**2 - 4 * (a * c))) / (2 * a)
        # This equation has one solutions: x
        return x
    else:
        x1 = (- b + math.sqrt((b**2) - (4 * (a * c)))) / (2 * a)
        x2 = (- b - math.sqrt((b**2) - (4 * (a * c)))) / (2 * a)
        # This equation has two solutions: x1 or x2
        return x1, x2


def dTheta(x0, hwhm):
    """
    A Lorenzian Profile fits this distribution, so this is the quantile
    function (inverse cumulative distribution) where the number that is
    returned is a value for dtheta.
    """
    r = random.random()
    # keep the angle change between -pi and pi with less calculation
    while r > 0.993557 or r < 0.006448:
        r = random.random()
    angle_change = x0 + hwhm * np.tan(np.pi * (r - 0.5))
    return angle_change


def distance(x, y):
    """
    Takes in positions in Mpc and returns the real distance between
    elements(Mpc) because hsml,radii and distances are in Mpc'''
    """
    return np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2 + (x[2] - y[2])**2)


def spline(particle_number, ray_dist, particle_attribs):
    """
    Calculate the strength of the magnetic field
    :param particle_number: int
        id for the particle
    :param ray_dist: float
        distance from ray to particle center
    :param particle_attribs: dict
        dictionary of particle attributes
    :return:
    """
    # this was converted to Mpc when it was read in
    hsml = particle_attribs['hsml_gas'][particle_number]
    r = ray_dist   # in Mpc
    q = np.absolute(r)/hsml
    if q >= 1:
        bit = 0
    elif 0.5 <= q < 1:
        bit = 2 * (1 - q)**3
    elif 0 <= q < 0.5:
        bit = 1 + 6 * q**2 * (q - 1)
    else:
        print 'Error with q value:', q
        bit = None
    w = bit
    return w

# -----------------------------------
#           IMPORT  FUNCTIONS
# -----------------------------------


def addToDict(key, dictionary, to_add):
    """
    Function to add details to dictionaries
    :param key: str
    :param dictionary: dict
    :param to_add: data
    :return:
    """
    try:
        dk = dictionary[key]
    except KeyError:
        dictionary[key] = []
    if isinstance(to_add, dict):
        dictionary[key] = to_add
    else:
        to_add = np.array(to_add)
        dictionary[key] = to_add


def readLightcone(object_type, indices_in_cone, dat_path, num_cube):
    print "-----------------------------------"
    print "----------", object_type, "-----------"
    print "-----------------------------------"
    attribute_file = object_type+"_"
    position_file = object_type+"_positions_"
    if object_type == 'GALAXY':
        list_file = object_type+"AVERAGES_"
        numbers_file = object_type+'_glist_'
    if object_type == 'HALO':
        list_file = object_type+"GALAXIES_"
        numbers_file = object_type+'_glist_'
    attribs = {}
    pos = {}
    gal_list = {}
    print "Reading in particle data from lightcone..."

    if object_type == "PARTICLE":
        ind = indices_in_cone[object_type+'_0']
    else:
        ind = indices_in_cone[object_type]

    with h5py.File(
            os.path.expanduser(dat_path + attribute_file + str(num_cube)),
            'r') as data:
        names = data.keys()
        for name in names:
            if name in ('radii_gas', 'radii_r200', 'radii_stellar', 'hsml_gas'):
                dat = data[name][:]
                dat = dat/1000.0    # positions are in Mpc, so radii must be too
                addToDict(name, attribs, dat)
            elif name == 'rho_gas':
                dat = data[name][:]  # code units were read in so we need to
                #  convert from 10e10M*/kpc3 to g/cm3
                UnitLength_in_cm = 3.085678e21
                UnitMass_in_g = 1.989e43
                dat = (dat*UnitMass_in_g)/(UnitLength_in_cm**3)
                addToDict(name, attribs, dat)
            else:
                dat = data[name][:]
                addToDict(name, attribs, dat)

    with h5py.File(os.path.expanduser(dat_path+position_file+str(num_cube)),
                   'r') as data:
        if object_type == "PARTICLE":
            xpos = data['gpos'][:, 0]
            ypos = data['gpos'][:, 1]
            zpos = data['gpos'][:, 2]
            addToDict('x', pos, xpos)
            addToDict('y', pos, ypos)
            addToDict('z', pos, zpos)
        else:
            xpos = data['pos'][:, 0]
            ypos = data['pos'][:, 1]
            zpos = data['pos'][:, 2]
            addToDict('x', pos, xpos)
            addToDict('y', pos, ypos)
            addToDict('z', pos, zpos)
    if object_type == "GALAXY":
        with h5py.File(os.path.expanduser(dat_path+list_file+str(num_cube)),
                       'r') as data:
            names = data.keys()
            for number in names:
                dat = data[number][:]
                dat = dat[ind]
                addToDict(number, gal_list, dat)
        with h5py.File(os.path.expanduser(
                dat_path + numbers_file + str(num_cube)), 'r') as data:
            names = data.keys()
            for number in names:
                dat = data[number][:]
                addToDict(number + '_gasID', gal_list, dat)

    if object_type == "HALO":
        with h5py.File(os.path.expanduser(dat_path+list_file+str(num_cube)),
                       'r') as data:
            names = data.keys()
            print 'keys in this file are:', names
            for number in names:
                print number, type(data[number])
                dat = data[number]
                if number in ['gasNumbers', 'starNumbers', 'galaxyNumbers']:
                    dat = data[number][:]
                    dat = dat[ind]
                    addToDict(number, gal_list, dat)
                else:
                    d = {}
                    for n in dat.keys():
                        key = n.split('_')[1]
                        if int(key) in ind:
                            d[key] = dat[n][:]
                    addToDict(number, gal_list, d)
        with h5py.File(os.path.expanduser(
                dat_path+numbers_file + str(num_cube)), 'r') as data:
            names = data.keys()
            for number in names:
                dat = data[number][:]
                addToDict(number + '_gasID', gal_list, dat)

    if object_type == "PARTICLE":
        print 'maximum particle x,y,z = (', max(pos['x']), max(pos['y']), \
            max(pos['z']), ')'
        print 'number of redshift entries= ', len(attribs['redshift_gas']), \
            'and number of particles is', len(pos['x'])
    position_data = np.array((pos['x'], pos['y'], pos['z']))
    position_data = position_data.T
    print "position data shape is", position_data.shape
    if object_type == "PARTICLE":
            return position_data, attribs
    else:
            return position_data, attribs, gal_list


def readStructure(num_cube, section_of_rays, in_path, numSources):
    particleID = {}
    particleDist = {}
    galaxyID = {}
    galaxyDist = {}
    haloID = {}
    haloDist = {}
    IDDicts = [particleID, galaxyID, haloID]
    distDicts = [particleDist, galaxyDist, haloDist]
    dataKeys = ['particle', 'galaxy', 'halo']
    print "Reading in structure data"

    with h5py.File(os.path.expanduser(in_path+'objectsAlongRay'+str(num_cube)+'_'+str(section_of_rays)),'r') as data:
        names = dataKeys
        print 'Structure keys are:', names
        for i in range(len(names)):
            print '\n\n ------------', names[i], '------------  \n\n'
            if names[i] == 'particle':
                dat = data[names[i]]
                print 'Structure subkeys keys are:', dat.keys()
                for key in dat.keys():
                    dat2 = dat[key]
                    key = str(key)
                    k = key.split('_')
                    lengthOfKeys = len(dat2.keys()) / 2
                    IdList = [[]] * lengthOfKeys
                    distList = [[]] * lengthOfKeys
                    for m in dat2.keys():
                       datSmall = dat2[m]
                       m = str(m)
                       key2 = m.split('_')
                       key2[1] = int(key2[1])
                       if key2[0] == 'raySection':
                          IdList[key2[1]] = list(datSmall[:])
                       else:
                          distList[key2[1]] = list(datSmall[:])
                    addToDict(k[1], IDDicts[i], IdList)
                    addToDict(k[1], distDicts[i], distList)
            else:
                dat = data[names[i]]
                print 'Structure subkeys keys are:', dat.keys()
                lengthOfKeys = len(dat.keys()) / 2
                for key in dat.keys():
                    key = str(key)
                    k = key.split('_')
                    if k[0] == 'raySection':
                        addToDict(k[1], IDDicts[i], dat[key])
                        print 'IDs added', dat[key].shape
                    else:
                        addToDict(k[1], distDicts[i], dat[key])
                        print 'Distances added', dat[key].shape

    return particleID, galaxyID, haloID, particleDist, galaxyDist, haloDist

# -----------------------------------
#     SPATIAL SEARCHING FUNCTIONS
# -----------------------------------


def combineParticleIDs(length, nearList1, nearList2, nearList3):
        nearCombined = []
        for i in range(length):
            if len(nearList1[i]) > 0:
                if len(nearList2[i]) > 0:
                    if len(nearList3[i]) > 0:
                        temp = nearList1[i] + nearList2[i] + nearList3[i]
                    else:
                        temp = []
                elif len(nearList2[i]) > 0:
                    temp = nearList1[i] + nearList3[i]
                else:
                    temp = nearList1[i]
            elif len(nearList2[i]) > 0:
                if len(nearList3[i]) > 0:
                    temp = nearList2[i] + nearList3[i]
                else:
                    temp = nearList2[i]
            elif len(nearList3[i]) > 0:
                temp = nearList3[i]
            else:
                temp = []
            nearCombined.append(temp)
        return nearCombined

def checkDistanceToRay(nearList, ray, positions, attributes, radiusKey):
        temp = []; distBetween=[]
        for i in range(len(ray)):
            temp2 = []
            rayPiece = ray[i]
            if len(nearList[i]) > 0:
                npPos = positions[nearList[i]]
                for j in npPos:
                    temp2.append(distance(rayPiece, j))
                radiusList = attributes[radiusKey][nearList[i]]
                closeEnoughTest = np.greater_equal(radiusList, temp2)
                closeEnough = np.array(nearList[i])[closeEnoughTest]
                temp.append(list(closeEnough))
                distBetween.append(list(np.array(temp2)[closeEnoughTest]))
            else:
                temp.append(nearList[i])
                distBetween.append(nearList[i])
        return temp, distBetween

# -----------------------------------
#    FIELD DIRECTION FUNCTIONS
# -----------------------------------


def findNe(lengthOfRay, particle, distances, particle_attribs):
        Ne = np.zeros(lengthOfRay)  #electron density
        for i in range(len(particle)):
            if len(particle[i])>0:
                rhoTemp = particle_attribs['rho_gas'][particle[i]]
                # find physical electron dnesity at my position#divide by
                # proton mass to get particles/cm3
                neTemp = particle_attribs['ne_gas'][particle[i]]
                neTemp = np.multiply(rhoTemp, neTemp)/(1.6726219e-24)
                # the particles are modelled as blobs with a certain density
                # falloff
                kernelList = np.zeros(len(particle[i]))
                for pn in range(len(particle[i])):
                    # find density at my distance from particle center
                    kernelList[pn] = spline(particle[i][pn], distances[i][pn],
                                            particle_attribs)
                    neTemp = np.multiply(neTemp, kernelList)
                    Ne[i] = sum(neTemp)
        return Ne


def directionSampling(length_of_ray, Ne, B_Direction, realistic_dl, dl_length):
        """ Direction of B dotted with direction of light ray will give
        positive or negative effect on RM  """
        dlVec = np.full((length_of_ray), dl_length)
        if B_Direction == "RANDOM":
                randomBit = random.normal(0, np.sqrt(0.3), length_of_ray)
                while np.any(randomBit > 1.0):
                    tooLarge = np.where(randomBit > 1.0)[0]
                    for j in tooLarge:
                        randomBit[j] = random.normal(0, np.sqrt(0.3))
                dlVec = randomBit * dlVec
        elif B_Direction == "REALISTIC":
            dlVec = realistic_dl
        elif B_Direction == "ALIGNED":
            dlVec = dlVec
        print B_Direction, dl_length, dlVec
        return dlVec

# -----------------------------------
#      MAGNETIC FIELD FUNCTIONS
# -----------------------------------


def outflowB(b, density, p_scale, alpha):
    """
    Outflow parameters from F. Stasyszyn et al (Measuring cosmic magnetic
    fields by rotation measure-galaxy cross-correlations in cosmological
    simulations)
    """
    factor = (density / p_scale)**alpha
    #print "shapes",B.shape, density.shape
    return b*factor

def filamentFields(length_of_ray,particle_id,distances,particle_attribs,B0,eta,pScale,alpha):
        B0 = B0*1.0e6    #faraday rotation formulaa takes magnetic field in microgauss, so we need to convert from gauss to microgauss
        #print len(particle_id),len(distances)
        length_of_ray = len(particle_id)
        Ne = np.zeros(length_of_ray)  #electron density
        outflow = np.zeros(length_of_ray) #metallicity > 0 implies that this particle used to be part of galaxy
        density = np.zeros(length_of_ray) #gas density at this point
        BfromLSS = np.zeros(length_of_ray)
        BfromOutflows = np.zeros(length_of_ray)
        for i in range(len(particle_id)):
                if len(particle_id[i])>0:
                        neTemp = particle_attribs['ne_gas'][particle_id[i]]
                        rhoTemp = particle_attribs['rho_gas'][particle_id[i]]
          #              print 'density,ne',np.max(rhoTemp),np.max(neTemp)
                        neTemp = np.multiply(rhoTemp,neTemp)/(1.6726219e-24)  #divide by proton mass to get particles/cm3
         #               print 'ne',np.max(neTemp)
                        outflowTemp = particle_attribs['z_gas'][particle_id[i]]
                        kernelList = np.zeros(len(particle_id[i]))  #the particles are modelled as blobs with a certain density falloff
                        for pn in range(len(particle_id[i])):
                            kernelList[pn] = spline(particle_id[i][pn],distances[i][pn],particle_attribs)
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


def emptyHalo(halo_id):
    global LA, LA_cut, haloGlist
    for i in range(len(LA)):
        if LA[i] == 'galaxyNumbers':
            full = np.where(haloGlist[LA[i]] > LA_cut[i])
            empty = np.where(haloGlist[LA[i]] <= LA_cut[i])
    return full, empty


def checkCentralGalaxy(near_galaxies):
    global galaxyAttribs
    print 'central galaxy check:', len(near_galaxies), \
        galaxyAttribs['central'].shape
    print 'central galaxy check:', near_galaxies, galaxyAttribs['central']
    central_gal = []
    for l in near_galaxies:
        for gal_id in l:
            if galaxyAttribs['central'][gal_id] == 1:
                central_gal.append(gal_id)
    return central_gal


def checkUseful(indices,object_type,LA,LA_cut,LO,attributes):
    """
    - Indices come from the nearest neighbour searches
    - Attribs of halo or galaxy
    - List, either halo or galaxy
    - LA = limiting attribute (list of strings)
    - LA_cut = vlaue of limit that results must be greater than (list of floats)
    - LO = origin of the limiting factor, either galaxy or halogalaxy (list of strings)
    """
    useful = {}
    if object_type == 'GALAXY':
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
    elif object_type == 'HALO':
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


def overlapping(indices_in_cone,particle_id,distances,usefulIndices,Glist):
         """
         Find all the particles that the ray intersected, and remove them from the nearParticle list
         """
         gasIndices = indices_in_cone['PARTICLE_0']
         if len(usefulIndices) ==0:
             particle_id_updated = particle_id
             distances_updated = distances
         else:
             particle_id_updated = [[]]*len(particle_id)
             distances_updated = [[]]*len(particle_id)
             print len(distances_updated)
             for i in usefulIndices:
                 gasIDs = Glist[str(i)+'_gasID']
                 overlap = np.where(gasIDs == gasIndices)[0] #find out which gas particles in the cone are in this halo
                 if len(overlap) == 0:    #if there is no overlap, then we do not need to worry about eliminating intersecting particles
                     particle_id_updated = particle_id
                     distances_updated = distances
                 else:
                     print len(gasIndices),len(gasIDs),sum([len(i) for i in particle_id])
                     for j in range(len(particle_id)):
                         if  len(particle_id[j])>0:
                             particle_temp = []
                             dists_temp = []
                             for k in range(len(particle_id[j])):
                                 if particle_id[j][k] in overlap:
                                     pass
                                 else:
                                     if particle_id[j][k] in particle_temp:
                                         pass
                                     else:
                                         particle_temp.append(particle_id[j][k])
                                         dists_temp.append(distances[j][k])
                             particle_id_updated[j].extend(particle_temp)
                             distances_updated[j].extend(dists_temp)
                         else:
                             pass
         return particle_id_updated,distances_updated

# -----------------------------------
#         MAIN FUNCTIONS
# -----------------------------------


def rayTrace(source2obs,args,out_q):
        rayInterval, particleTreeSmall,particleTreeMed,particleTreeLarge,galaxyTree,haloTree,galaxySampleRadius,haloSampleRadius,particlePositions, particle_attribs, galaxyPositions, galaxyAttribs, haloPositions, haloAttribs,galaxyGlist, haloGlist = args

        printString = ''
        printString += "-----------------------------------\n"
        printString += "-----------TRACING RAY-------------\n"
        printString += "-----------------------------------\n"

        lenS20 = len(source2obs)
        #Search the tree
#        inputList = [[particlePositions[i],particle_attribs['hsml_gas'][i]] for i in range(len(particle_attribs['hsml_gas']))]
#        originalNumberParticles = np.apply_along_axis(searchPerRadius,1,inputList,str(source2obs[0]))  

        nearParticlesSmall = particleTreeSmall.query_ball_point(source2obs,1.0)  #gives list of particles close to ray
        nearParticlesMed = particleTreeMed.query_ball_point(source2obs,1.0)
        nearParticlesLarge = particleTreeLarge.query_ball_point(source2obs, 1.0)

        printString += 'small,medium,large particles before distance check '+str(sum([len(j) for j in nearParticlesSmall]))+','+str(sum([len(j) for j in nearParticlesMed]))+','+str(sum([len(j) for j in nearParticlesLarge]))+'\n'
        originalNumberParticles =sum([sum([len(j) for j in nearParticlesSmall]),sum([len(j) for j in nearParticlesMed]),sum([len(j) for j in nearParticlesLarge])])


        nearParticlesSmall,distBetSmall = checkDistanceToRay(nearParticlesSmall,source2obs,particlePositions, particle_attribs, 'hsml_gas')
        nearParticlesMed,distBetMed = checkDistanceToRay(nearParticlesMed,source2obs,particlePositions, particle_attribs, 'hsml_gas')
        nearParticlesLarge,distBetLarge = checkDistanceToRay(nearParticlesLarge,source2obs,particlePositions, particle_attribs, 'hsml_gas')

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


def BdirectionFromTree(rays,coherenceLength,particleTreeSmall,particleTreeMed,particleTreeLarge,particlePositions,particle_attribs,particle_id,particleDist,voight_x0,voight_HWHM,rayInterval):
    """
    Alternative way to assign directions by looking for particles within cooherence length and trying to align direction with the structure
   """
    print [sum([len(i) for i in r]) for r in particle_id.values()]
    dlDict = {}
    for r in rays.keys():
        if len(rays[r])==0:
            dlDict[r] = []
        else:
            NeToConsider = findNe(len(rays[r]),particle_id[r],particleDist[r],particle_attribs)
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


def obtainRM(particle_id,galaxyID,haloID,particleDist,galaxyDist,haloDist,particle_attribs,B0,eta,pScale,alpha,realistic_dl,B_Direction,LA,LA_cut,LO,galaxyAttribs,galaxyGlist,haloAttribs,haloGlist,indices_in_cone,out_q):
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

        lenS2O = len(particle_id)

#        centralGalaxies = checkCentralGalaxy(galaxyID)  #galaxies that are central to their halos

#        fullHalos,emptyHalos = emptyHalo(haloID) #Different types of halos get different treatment

        print 'len PID before overlaps',len(particle_id)
        #Clean up particle list so no double-sampling

        #--------ELIMINATE GALAXY OVERLAPS--------# 

        if len(galaxyID)>0:
            usefulGalaxies = checkUseful(galaxyID,'GALAXY',LA,LA_cut,LO,galaxyAttribs)
            particle_id,particleDist = overlapping(indices_in_cone,particle_id,particleDist,usefulGalaxies,galaxyGlist)

        else:
            usefulGalaxies = galaxyID
        #--------ELIMINATE HALO OVERLAPS--------#       

        if len(haloID)>0:
            usefulHalos = checkUseful(haloID,'HALO',LA,LA_cut,LO,haloGlist)
            print [len(i) for i in particle_id[:10]]
            particle_id,particleDist = overlapping(indices_in_cone,particle_id,particleDist,usefulHalos,haloGlist)
            print [len(i) for i in particle_id[:10]]
        else:
            usefulHalos = haloID

        #Keep numbers for stats
        numberParticles = sum([len(j) for j in particle_id])
        numberGalaxies = len(galaxyID) #different because galaxy and halo B calculations should only be done once each per source because of nature of calculation
        numberHalos = len(haloID)

        #B, Ne from gas particles
        print 'len PID after overlaps',len(particle_id)
        filamentData = filamentFields(lenS2O,particle_id,particleDist,particle_attribs,B0,eta,pScale,alpha)

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

# -----------------------------------
#         LIGHTRAY FUNCTIONS
# -----------------------------------


def lightRay(start_distance, piece, vector_pc, out_q):
    """
    Takes in the source positions all at once and returns vectors along
    the LOS from observer to source
    """
    # array to use to find vectors to sources
    pcs = np.linspace(0, start_distance, piece+1)
    # construct the light ray
    a = np.multiply(vector_pc.reshape(3, 1), pcs)
    if out_q == 'NOTPARALLEL':
        return a
    else:
        out_q.put(a)

# -----------------------------------
#         OTHER FUNCTIONS
# -----------------------------------


def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-
    evenly-sized-chunks-in-python
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]


def replaceLine(file_name, parameter, new_number):
        """
        Replacing some parameters in the params.txt file to reflect the
        lightcone being used
        """
        lines = open(file_name, 'r').readlines()
        for i in range(len(lines)):
                words = lines[i].split("=")
                if words[0] == parameter:
                        lines[i] = parameter + "=" + str(new_number) + "\n"
        out = open(file_name, 'w')
        out.writelines(lines)
        out.close()


# -----------------------------------
#         SYSARG FUNCTIONS
# -----------------------------------


def check_direction(path):
    """
    Parse the path to find the method use to choose magnetic field directions
    :param path:
    :return:
    """
    if 'REALISTIC' in path:
        B_direction = 'REALISTIC'
    elif 'ALIGNED' in path:
        B_direction = 'ALIGNED'
    elif 'RANDOM' in path:
        B_direction = 'RANDOM'
    return B_direction


def check_B0(path):
    """
    Parse the path to find the value of B0, the initial magnetic field
    :param path:
    :return:
    """
    if '06' in path:
        B0 = 1.0e-6
    if '07' in path:
        B0 = 1.0e-7
    if '08' in path:
        B0 = 1.0e-8
    if '09' in path:
        B0 = 1.0e-9
    return B0


def check_eta(path):
    """
    Parse the path to find the value of eta
    :param path: str
    :return:
    """
    if '0.4' in path:
        eta = 0.4
    if '0.5' in path:
        eta = 0.5
    if '0.6' in path:
        eta = 0.6
    return eta


