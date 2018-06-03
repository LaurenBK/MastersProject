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


def d_theta(x0, hwhm):
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


def add_to_dict(key, dictionary, to_add):
    """
    Function to add details to dictionaries
    :param key: str
    :param dictionary: dict
    :param to_add: data
    :return:
    """
    try:
        _ = dictionary[key]
    except KeyError:
        dictionary[key] = []
    if isinstance(to_add, dict):
        dictionary[key] = to_add
    else:
        to_add = np.array(to_add)
        dictionary[key] = to_add


def read_lightcone(object_type, indices_in_cone, dat_path, num_cube):
    print "-----------------------------------"
    print "----------", object_type, "-----------"
    print "-----------------------------------"
    attribute_file = object_type+"_"
    position_file = object_type+"_positions_"
    if object_type == 'GALAXY':
        list_file = object_type+"AVERAGES_"
        numbers_file = object_type+'_gas_list_'
    if object_type == 'HALO':
        list_file = object_type+"GALAXIES_"
        numbers_file = object_type+'_gas_list_'
    attribs = {}
    pos = {}
    gas_list = {}
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
                add_to_dict(name, attribs, dat)
            elif name == 'rho_gas':
                dat = data[name][:]  # code units were read in so we need to
                #  convert from 10e10M*/kpc3 to g/cm3
                unit_length_in_cm = 3.085678e21
                unit_mass_in_g = 1.989e43
                dat = (dat*unit_mass_in_g)/(unit_length_in_cm**3)
                add_to_dict(name, attribs, dat)
            else:
                dat = data[name][:]
                add_to_dict(name, attribs, dat)

    with h5py.File(os.path.expanduser(dat_path+position_file+str(num_cube)),
                   'r') as data:
        if object_type == "PARTICLE":
            xpos = data['gpos'][:, 0]
            ypos = data['gpos'][:, 1]
            zpos = data['gpos'][:, 2]
            add_to_dict('x', pos, xpos)
            add_to_dict('y', pos, ypos)
            add_to_dict('z', pos, zpos)
        else:
            xpos = data['pos'][:, 0]
            ypos = data['pos'][:, 1]
            zpos = data['pos'][:, 2]
            add_to_dict('x', pos, xpos)
            add_to_dict('y', pos, ypos)
            add_to_dict('z', pos, zpos)
    if object_type == "GALAXY":
        with h5py.File(os.path.expanduser(dat_path+list_file+str(num_cube)),
                       'r') as data:
            names = data.keys()
            for number in names:
                dat = data[number][:]
                dat = dat[ind]
                add_to_dict(number, gas_list, dat)
        with h5py.File(os.path.expanduser(
                dat_path + numbers_file + str(num_cube)), 'r') as data:
            names = data.keys()
            for number in names:
                dat = data[number][:]
                add_to_dict(number + '_gasID', gas_list, dat)

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
                    add_to_dict(number, gas_list, dat)
                else:
                    d = {}
                    for n in dat.keys():
                        key = n.split('_')[1]
                        if int(key) in ind:
                            d[key] = dat[n][:]
                    add_to_dict(number, gas_list, d)
        with h5py.File(os.path.expanduser(
                dat_path+numbers_file + str(num_cube)), 'r') as data:
            names = data.keys()
            for number in names:
                dat = data[number][:]
                add_to_dict(number + '_gasID', gas_list, dat)

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
            return position_data, attribs, gas_list


def read_structure(num_cube, section_of_rays, in_path):
    particle_id = {}
    particle_dist = {}
    galaxy_id = {}
    galaxy_dist = {}
    halo_id = {}
    halo_dist = {}
    id_dicts = [particle_id, galaxy_id, halo_id]
    dist_dicts = [particle_dist, galaxy_dist, halo_dist]
    data_keys = ['particle', 'galaxy', 'halo']
    print "Reading in structure data"

    with h5py.File(os.path.expanduser(
            in_path + 'objectsAlongRay' + str(num_cube) + '_' +
            str(section_of_rays)), 'r') as data:
        names = data_keys
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
                    length_of_keys = len(dat2.keys()) / 2
                    id_list = [[]] * length_of_keys
                    dist_list = [[]] * length_of_keys
                    for m in dat2.keys():
                        dat_small = dat2[m]
                        m = str(m)
                        key2 = m.split('_')
                        key2[1] = int(key2[1])
                        if key2[0] == 'raySection':
                            id_list[key2[1]] = list(dat_small[:])
                        else:
                            dist_list[key2[1]] = list(dat_small[:])
                    add_to_dict(k[1], id_dicts[i], id_list)
                    add_to_dict(k[1], dist_dicts[i], dist_list)
            else:
                dat = data[names[i]]
                print 'Structure subkeys keys are:', dat.keys()
                for key in dat.keys():
                    key = str(key)
                    k = key.split('_')
                    if k[0] == 'raySection':
                        add_to_dict(k[1], id_dicts[i], dat[key])
                        print 'IDs added', dat[key].shape
                    else:
                        add_to_dict(k[1], dist_dicts[i], dat[key])
                        print 'Distances added', dat[key].shape

    return particle_id, galaxy_id, halo_id, particle_dist, galaxy_dist,\
        halo_dist

# -----------------------------------
#     SPATIAL SEARCHING FUNCTIONS
# -----------------------------------


def combine_particle_ids(length, near_list1, near_list2, near_list3):
        near_combined = []
        for i in range(length):
            if len(near_list1[i]) > 0:
                if len(near_list2[i]) > 0:
                    if len(near_list3[i]) > 0:
                        temp = near_list1[i] + near_list2[i] + near_list3[i]
                    else:
                        temp = []
                elif len(near_list2[i]) > 0:
                    temp = near_list1[i] + near_list3[i]
                else:
                    temp = near_list1[i]
            elif len(near_list2[i]) > 0:
                if len(near_list3[i]) > 0:
                    temp = near_list2[i] + near_list3[i]
                else:
                    temp = near_list2[i]
            elif len(near_list3[i]) > 0:
                temp = near_list3[i]
            else:
                temp = []
            near_combined.append(temp)
        return near_combined


def check_distance_to_ray(near_list, ray, positions, attributes, radius_key):
        temp = []
        dist_between = []
        for i in range(len(ray)):
            temp2 = []
            ray_piece = ray[i]
            if len(near_list[i]) > 0:
                np_pos = positions[near_list[i]]
                for j in np_pos:
                    temp2.append(distance(ray_piece, j))
                radius_list = attributes[radius_key][near_list[i]]
                close_enough_test = np.greater_equal(radius_list, temp2)
                close_enough = np.array(near_list[i])[close_enough_test]
                temp.append(list(close_enough))
                dist_between.append(list(np.array(temp2)[close_enough_test]))
            else:
                temp.append(near_list[i])
                dist_between.append(near_list[i])
        return temp, dist_between

# -----------------------------------
#    FIELD DIRECTION FUNCTIONS
# -----------------------------------


def find_ne(length_of_ray, particle, distances, particle_attribs):
        ne = np.zeros(length_of_ray)  # electron density
        for i in range(len(particle)):
            if len(particle[i]) > 0:
                rho_temp = particle_attribs['rho_gas'][particle[i]]
                # find physical electron density at my position#divide by
                # proton mass to get particles/cm3
                ne_temp = particle_attribs['ne_gas'][particle[i]]
                ne_temp = np.multiply(rho_temp, ne_temp) / 1.6726219e-24
                # the particles are modelled as blobs with a certain density
                # falloff
                kernel_list = np.zeros(len(particle[i]))
                for pn in range(len(particle[i])):
                    # find density at my distance from particle center
                    kernel_list[pn] = spline(particle[i][pn], distances[i][pn],
                                             particle_attribs)
                    ne_temp = np.multiply(ne_temp, kernel_list)
                    ne[i] = sum(ne_temp)
        return ne


def direction_sampling(length_of_ray, B_direction, realistic_dl, dl_length):
        """ Direction of B dotted with direction of light ray will give
        positive or negative effect on RM  """
        dl_vec = np.full(length_of_ray, dl_length)
        if B_direction == "RANDOM":
                random_bit = random.normal(0, np.sqrt(0.3), length_of_ray)
                while np.any(random_bit > 1.0):
                    too_large = np.where(random_bit > 1.0)[0]
                    for j in too_large:
                        random_bit[j] = random.normal(0, np.sqrt(0.3))
                dl_vec = random_bit * dl_vec
        elif B_direction == "REALISTIC":
            dl_vec = realistic_dl
        elif B_direction == "ALIGNED":
            dl_vec = dl_vec
        print B_direction, dl_length, dl_vec
        return dl_vec

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
    return b*factor


def filament_fields(particle_id, distances, particle_attribs, B0,
                    p_scale, alpha):
        B0 = B0 * 1.0e6
        # faraday rotation formula takes magnetic field in microgauss, so we
        #  need to convert from gauss to microgauss
        length_of_ray = len(particle_id)
        ne = np.zeros(length_of_ray)  # electron density
        outflow = np.zeros(length_of_ray)
        # metallicity > 0 implies that this particle used to be part of galaxy
        density = np.zeros(length_of_ray)  # gas density at this point
        BfromLSS = np.zeros(length_of_ray)
        Bfrom_outflows = np.zeros(length_of_ray)
        for i in range(len(particle_id)):
                if len(particle_id[i]) > 0:
                        ne_temp = particle_attribs['ne_gas'][particle_id[i]]
                        rho_temp = particle_attribs['rho_gas'][particle_id[i]]
                        ne_temp = np.multiply(rho_temp, ne_temp) / 1.6726219e-24
                        # divide by proton mass to get particles/cm3
                        outflow_temp = particle_attribs['z_gas'][particle_id[i]]
                        kernel_list = np.zeros(len(particle_id[i]))
                        # the particles are modelled as blobs with a certain
                        # density falloff
                        for pn in range(len(particle_id[i])):
                            kernel_list[pn] = spline(particle_id[i][pn],
                                                     distances[i][pn],
                                                     particle_attribs)
                        ne_temp = np.multiply(ne_temp, kernel_list)
                        # metallicity is defined for whole particle
                        ne[i] = sum(ne_temp)
                        outflow[i] = np.mean(outflow_temp)
                        density[i] = sum(rho_temp)

                        if np.max(ne_temp) != 0:
                            Btemp = B0 * ne_temp * 1000000.0
                            # the multiplication by 100000 is necessary to pull
                            #  the magnetic field back to the desired field
                            # strength
                            no_outflow = np.where(outflow_temp == 0)
                            BfromLSS_masked = Btemp[:]
                            BfromLSS_masked[no_outflow[0]] = 0
                            # we do this to isolate only the places in the ray
                            #  where metallicity is present
                        else:  # ie, if the electron density is 0 for this ray
                            # piece (should not happen)
                            Btemp = np.array([0.0])
                            BfromLSS_masked = np.array([0.0])

                        BfromLSS[i] = sum(Btemp)
                        Bfrom_outflows[i] = sum(outflowB(BfromLSS_masked,
                                                         rho_temp, p_scale,
                                                         alpha))

                else:
                        outflow[i] = 0.0
                        ne[i] = 0.0
                        density[i] = 0.0
                        BfromLSS[i] = 0.0
                        Bfrom_outflows[i] = 0.0


        particleB = BfromLSS + Bfrom_outflows
        # normalised and scaled so that the upper limit matches that of
        # Akahori, Ryu et al, http://iopscience.iop.org/article/10.1088/0004
        # -637X/723/1/476/meta
        to_return = [particleB, ne]
        return to_return

# -----------------------------------
#      OBJECT CHECK FUNCTIONS
# -----------------------------------


def get_indices(list_to_check,associated_dist):
    """
    Makes sure that every galaxy or halo is only counted once
    """
    indices = []
    dist_between = []
    for i in range(len(list_to_check)):
        if len(list_to_check[i]) > 0:
            indices.extend(list_to_check[i])
            dist_between.extend(associated_dist[i])
    indices = np.array(indices)
    dist_between = np.array(dist_between)
    if len(indices) > 0:
        unq, unq_in = np.unique(indices, return_index=True)
        indices = unq
        dist_between = dist_between[unq_in]
    indices = indices.astype(int)
    indices = indices.T
    dist_between = dist_between.T
    return list(indices), list(dist_between)


def empty_halo(halo_id):
    global LA, LA_cut, halo_gas_list
    for i in range(len(LA)):
        if LA[i] == 'galaxyNumbers':
            full = np.where(halo_gas_list[LA[i]] > LA_cut[i])
            empty = np.where(halo_gas_list[LA[i]] <= LA_cut[i])
    return full, empty


def check_central_galaxy(near_galaxies):
    global galaxy_attribs
    print 'central galaxy check:', len(near_galaxies), \
        galaxy_attribs['central'].shape
    print 'central galaxy check:', near_galaxies, galaxy_attribs['central']
    central_gal = []
    for l in near_galaxies:
        for gal_id in l:
            if galaxy_attribs['central'][gal_id] == 1:
                central_gal.append(gal_id)
    return central_gal


def check_useful(indices, object_type, LA, LA_cut, LO, attributes):
    """
    - Indices come from the nearest neighbour searches
    - Attribs of halo or galaxy
    - List, either halo or galaxy
    - LA = limiting attribute (list of strings)
    - LA_cut = vlaue of limit that results must be greater than (list of floats)
    - LO = origin of the limiting factor, either galaxy or halogalaxy (list of
    strings)
    """
    useful = {}
    if object_type == 'GALAXY':
        for i in range(len(LO)):
            if LO[i] == 'galaxy':
                attributes_list = attributes[LA[i]]
                print LA_cut, i
                print LA_cut[i]
                if type(attributes_list[0]) == 'numpy.int64' or 'numpy.float64':
                    useful[LA[i]] = indices[np.where(attributes[LA[i]][indices]
                                                     > LA_cut[i])[0]]
                elif type(attributes_list[0]) == 'numpy.ndarray':
                    useful[LA[i]] = indices[np.where(
                        attributes[LA[i]][0][indices] > LA_cut[i])[0]]
                else:
                    print "Weird errors with attributes;", attributes_list,\
                        type(attributes_list[0])
        for i in useful.keys():
            print 'number of galaxies that can be used with key', i, 'is:', \
                len(useful[i])
        useful_list = np.intersect1d(useful.values()[0],useful.values()[1])
    elif object_type == 'HALO':
        for i in range(len(LO)):
            if LO[i] == 'halogalaxy':
                try:
                    useful[LA[i]] = indices[np.where(
                        attributes[LA[i]][indices] > LA_cut[i])[0]]
                except:
                    useful[LA[i]] = indices[np.where(
                        attributes[LA[i]][0][indices] > LA_cut[i])[0]]
                useful_list = useful[LA[i]]
        for i in useful.keys():
            print 'number of halos that can be used with key', i, 'is:', \
                len(useful[i])
    return useful_list


def overlapping(indices_in_cone, particle_id, distances, useful_indices,
                gas_list):
    """
    Find all the particles that the ray intersected, and remove them
    from the nearParticle list
    """
    gas_indices = indices_in_cone['PARTICLE_0']
    if len(useful_indices) == 0:
        particle_id_updated = particle_id
        distances_updated = distances
    else:
        particle_id_updated = [[]] * len(particle_id)
        distances_updated = [[]] * len(particle_id)
        for i in useful_indices:
            gas_ids = gas_list[str(i) + '_gasID']
            overlap = np.where(gas_ids == gas_indices)[0]
            # find out which gas particles in the cone are in this halo
            if len(overlap) == 0:
                # if there is no overlap, then we do not need to worry
                # about eliminating intersecting particles
                particle_id_updated = particle_id
                distances_updated = distances
            else:
                for j in range(len(particle_id)):
                    if len(particle_id[j]) > 0:
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
    return particle_id_updated, distances_updated

# -----------------------------------
#         MAIN FUNCTIONS
# -----------------------------------


def ray_trace(source2obs, args, out_q):
        ray_interval, particle_tree_small, particle_tree_med, \
        particle_tree_large, galaxy_tree, halo_tree, galaxy_sample_radius, \
        halo_sample_radius, particle_positions, particle_attribs, \
        galaxy_positions, galaxy_attribs, halo_positions, halo_attribs,\
        galaxy_gas_list, halo_gas_list = args

        print_string = ''
        print_string += "-----------------------------------\n"
        print_string += "-----------TRACING RAY-------------\n"
        print_string += "-----------------------------------\n"

        len_s20 = len(source2obs)
        # Search the tree
        near_particles_small = particle_tree_small.query_ball_point(
            source2obs, 1.0)  #gives list of particles close to ray
        near_particles_med = particle_tree_med.query_ball_point(
            source2obs, 1.0)
        near_particles_large = particle_tree_large.query_ball_point(
            source2obs, 1.0)

        print_string += 'small,medium,large particles before distance check ' \
                        + str(sum([len(j) for j in near_particles_small])) + \
                        ',' + str(sum([len(j) for j in near_particles_med])) + \
                        ',' + str(sum([len(j) for j in near_particles_large]))\
                        + '\n'
        original_number_particles =sum(
            [sum([len(j) for j in near_particles_small]),
                sum([len(j) for j in near_particles_med]),
                sum([len(j) for j in near_particles_large])])


        near_particles_small, dist_between_small = check_distance_to_ray(
            near_particles_small,source2obs,particle_positions,
            particle_attribs, 'hsml_gas')
        near_particles_med, dist_between_med = check_distance_to_ray(
            near_particles_med,source2obs,particle_positions, particle_attribs,
            'hsml_gas')
        near_particles_large, dist_between_large = check_distance_to_ray(
            near_particles_large,source2obs,particle_positions,
            particle_attribs, 'hsml_gas')

        print_string +=  'small,medium,large particles after check ' + \
                         str(sum([len(j) for j in near_particles_small])) + \
                         ',' + str(sum([len(j) for j in near_particles_med])) \
                         + ',' + \
                         str(sum([len(j) for j in near_particles_large])) + '\n'
            
        near_particles = combine_particle_ids(
            len_s20, near_particles_small, near_particles_med,
            near_particles_large)
        particle_dist = combine_particle_ids(
            len_s20, dist_between_small, dist_between_med, dist_between_large)
        print len(near_particles), len(particle_dist)
        near_galaxies = galaxy_tree.query_ball_point(
            source2obs, galaxy_sample_radius)
        near_halos = halo_tree.query_ball_point(source2obs, halo_sample_radius)
        print_string += 'initial numbers of galaxies and halos ' + \
                        str(sum([len(j) for j in near_galaxies])) + ',' + \
                        str(sum([len(j) for j in near_halos])) + '\n'

        near_galaxies, dist_between_galaxies = check_distance_to_ray(
            near_galaxies, source2obs, galaxy_positions, galaxy_attribs,
            'radii_r200')
        near_halos, dist_between_halo = check_distance_to_ray(
            near_halos, source2obs, halo_positions, halo_attribs, 'radii_r200')

        near_galaxies = np.array(near_galaxies)
        dist_between_galaxies = np.array(dist_between_galaxies)
        near_halos = np.array(near_halos)
        dist_between_halo = np.array(dist_between_halo)

        # Get a set of the halos and galaxies to avoid duplicates
        galaxy_indices, galaxy_dist = get_indices(near_galaxies,
                                                  dist_between_galaxies)
        halo_indices, halo_dist = get_indices(near_halos, dist_between_halo)

        print_string +=  'secondary numbers of galaxies and halos ' \
                         '(actually close enough to ray) ' + \
                         str(len(galaxy_indices)) + ',' + \
                         str(len(halo_indices)) + '\n'

        number_hit_per_source = [original_number_particles,
            len(galaxy_indices), len(halo_indices)]
        print print_string
        to_return = [near_particles, particle_dist, galaxy_indices, galaxy_dist,
            halo_indices, halo_dist, number_hit_per_source]

        if out_q == 'NOTPARALLEL':
            return to_return
        else:
            out_q.put(to_return)


def B_direction_from_tree(rays, coherence_length, particle_tree_small,
                       particle_tree_med, particle_tree_large,
                       particle_positions, particle_attribs, particle_id,
                       particle_dist, voight_x0, voight_HWHM, ray_interval):
    """
    Alternative way to assign directions by looking for particles within
    cooherence length and trying to align direction with the structure
   """
    print [sum([len(i) for i in r]) for r in particle_id.values()]
    dl_dict = {}
    for r in rays.keys():
        if len(rays[r]) == 0:
            dl_dict[r] = []
        else:
            NeToConsider = find_ne(len(rays[r]), particle_id[r],
                                   particle_dist[r], particle_attribs)
            near_particles_small = particle_tree_small.query_ball_point(
                rays[r], coherence_length)
            #gives list of particles close to ray
            near_particles_med = particle_tree_med.query_ball_point(
                rays[r], coherence_length)
            near_particles_large = particle_tree_large.query_ball_point(
                rays[r], coherence_length)
            near_particles = combine_particle_ids(len(
                rays[r]), near_particles_small, near_particles_med,
                near_particles_large)
            dl_vec = [0] * len(rays[r])
            try:
                print 'near_particles', len(near_particles),\
                    len(near_particles[0])
            except:
                pass
            sys.stdout.flush()
            for i in range(len(rays[r])):
                if NeToConsider[i] == 0:
                     dl_vec[i] = 0
                     init_theta = 2 * np.pi * random.random()
                else:
                    if len(near_particles[i]) == 0:
                        change_in_theta = d_theta(voight_x0, voight_HWHM)
                        init_theta = init_theta + change_in_theta
                        dl_vec[i] = ray_interval * np.cos(init_theta)
                    else:
                    #put in on 16/7/17
                        if len(near_particles[i])>1e6:
                            percent01 = int(0.001 * len(near_particles[i]))
                            near_particles_temp = random.choice(
                                near_particles[i], size = percent01)
                        elif len(near_particles[i]) > 1e5:
                            percent1 = int(0.01 * len(near_particles[i]))
                            near_particles_temp = random.choice(
                                near_particles[i], size = percent1)
                        elif len(near_particles[i]) > 1e4:
                            percent10 = int(0.1 * len(near_particles[i]))
                            near_particles_temp = random.choice(
                                near_particles[i], size = percent10)
                    #------------------
                        vectors = particle_positions[near_particles_temp] - \
                                  rays[r][i]
                        normed_vectors = np.apply_along_axis(normalise, 0,
                                                             np.array(vectors))
                        resultant_vec = [sum(normed_vectors[:, 0]),
                            sum(normed_vectors[:, 1]),
                            sum(normed_vectors[:, 1])]
                        normed_resultant_vec = normalise(
                            np.array(resultant_vec))
                        angular_contribution = angle_between(
                            normed_resultant_vec, rays[r][-1])
                        dl_vec[i] = ray_interval * np.cos(angular_contribution)
            dl_dict[r] = dl_vec
            print 'Length of directions for ray',r,'is',len(dl_vec),len(rays[r])
    return dl_dict


def obtain_rm(particle_id, galaxy_id, halo_id, particle_dist, galaxy_dist,
              halo_dist, particle_attribs, B0, eta, p_scale, alpha,
              realistic_dl, B_direction, LA, LA_cut, LO, galaxy_attribs,
              galaxy_gas_list, halo_attribs, halo_gas_list, indices_in_cone,
              out_q):
        """
        - Adds electron density and metallicity contributions along ray length
        - Calculates magnetic field
        - Ramdomises magnetic field direction
        - Returns Rotation Measure vector
        """

        print_string = ''
        print_string += "-----------------------------------\n"
        print_string += "---------PROCESSING RAY------------\n"
        print_string += "-----------------------------------\n"

        len_s20 = len(particle_id)

        print 'len PID before overlaps', len(particle_id)

        #--------ELIMINATE GALAXY OVERLAPS--------#
        if len(galaxy_id) > 0:
            useful_galaxies = check_useful(galaxy_id, 'GALAXY', LA, LA_cut, LO,
                                          galaxy_attribs)
            particle_id, particle_dist = overlapping(
                indices_in_cone, particle_id, particle_dist, useful_galaxies,
                galaxy_gas_list)

        else:
            useful_galaxies = galaxy_id

        #--------ELIMINATE HALO OVERLAPS--------#
        if len(halo_id) > 0:
            useful_halos = check_useful(halo_id, 'HALO', LA, LA_cut, LO,
                                        halo_gas_list)
            print [len(i) for i in particle_id[:10]]
            particle_id, particle_dist = overlapping(
                indices_in_cone, particle_id, particle_dist, useful_halos,
                halo_gas_list)
            print [len(i) for i in particle_id[:10]]
        else:
            useful_halos = halo_id

        #Keep numbers for stats
        number_particles = sum([len(j) for j in particle_id])
        number_galaxies = len(galaxy_id)
        number_halos = len(halo_id)

        #B, Ne from gas particles
        print 'len PID after overlaps',len(particle_id)
        filament_data = filament_fields(particle_id, particle_dist,
                                        particle_attribs, B0, p_scale, alpha)
        B = filament_data[0]
        Ne = filament_data[1]
        print "Ne size", len(Ne), 'length ray', len_s20
        dl = direction_sampling(len_s20, Ne, B_direction, realistic_dl, 10)
        print 'max dl', np.max(dl)
        print 'max B %.3e microGauss' % np.max(B)
        print 'max ne %.3e' % np.max(Ne)
        RM = 811.9 * Ne * B * dl
        # ray interval is given in mpc but we need it in kpc
        RM= np.nan_to_num(RM)
        print 'max RM',np.max(RM)
        galaxy_data = useful_galaxies  # list of galaxies close to the ray
        halo_data = useful_halos     # list of halos close to the ray
        num_hit = [number_particles, len(useful_galaxies), len(useful_halos)]
        print max(B), max(Ne)
        to_return = (RM, galaxy_data, halo_data, num_hit, B, Ne, dl)

        print_string += str(number_particles) + " particles, " + \
            str(len(useful_galaxies)) + " galaxies, and " + \
            str(len(useful_halos)) + \
            " halos have been found near this LOS.\n"
        print_string += "-----------------------------------\n"
        print print_string

        if out_q == 'NOTPARALLEL':
            return to_return
        else:
            out_q.put(to_return)

# -----------------------------------
#         LIGHTRAY FUNCTIONS
# -----------------------------------


def light_ray(start_distance, piece, vector_pc, out_q):
    """
    Takes in the source positions all at once and returns vectors along
    the LOS from observer to source
    """
    # array to use to find vectors to sources
    pcs = np.linspace(0, start_distance, piece + 1)
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
        yield l[i:i + n]


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


