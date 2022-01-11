import os

import pandas as pd
import numpy as np

from sklearn import preprocessing 

def prep_travel_forecasting_data(
    path_to_data='data/public/travel time forecasting/',
    dataset_name='Uber movement',
    city_name='Amsterdam',
    test_split=0.7,
    time_encoding='ORD',
    normalization=True,
    standardization=True,
    silent=True,
    plot=False,
):

    """
    """
    
    ### Create all required paths to where chosen data is stored
    
    # create the base path to data
    base_path = path_to_data + dataset_name + '/' + city_name + '/'
    file_list = os.listdir(base_path)
    
    # search directory for .json files
    json_file_name = [file for file in file_list if file.endswith('.json')][0]
    
    # search directory for .csv files
    csv_file_name = [file for file in file_list if file.endswith('.csv')][0]
    
    # create the full paths to json and csv data
    path_to_json_data = base_path + json_file_name
    path_to_rawdata = base_path + csv_file_name
    
    
    ### Save all parameters in dictionary
    raw_data = {
        'path_to_json_data': path_to_json_data,
        'path_to_rawdata': path_to_rawdata,
        'test_split': test_split,
        'time_encoding': time_encoding,
        'normalization': normalization,
        'standardization': standardization,
        'silent': silent,
        'plot': plot
    }
    
    
    ### Import data
    
    # import travel time data
    travel_data = pd.read_csv(path_to_rawdata)
    
    # import spatial data
    (
        map_movement_id_to_latitude_coordinates,
        map_movement_id_to_longitude_coordinates
    ) = import_geojson(path_to_json_data)
    
    # Calculate city zone centroids from polygons
    (
        map_movement_id_to_centroid_lat,
        map_movement_id_to_centroid_long
    ) = calc_centroids(
        map_movement_id_to_latitude_coordinates,
        map_movement_id_to_longitude_coordinates
    )
    
    # merge into city_zone coordinates
    city_zone_coordinates = create_city_zone_coordinates(path_to_json_data)
    
    
    ### Pair features and labels
    dataset = create_feature_label_pairs(
        city_zone_coordinates,
        travel_data
    )
    
    
    ### Encode temporal features
    dataset = encode_time_features(raw_data, dataset)
    
    
    ### Normalize all data
    dataset = normalize_features(raw_data, dataset) 
    
    
    ### Split data into available and candidate datasets
    avail_data, cand_data = split_avail_cand(raw_data, dataset)
    
    
    ### Standardize features
    cand_data = standardize_features(
        raw_data, 
        cand_data, 
        avail_data
    )
    avail_data = standardize_features(
        raw_data, 
        avail_data, 
        avail_data
    )
    
    
    ### Create return datasets
    datasets = {
      'avail_data': avail_data,
      'cand_data': cand_data
    }
    
    return datasets
    
def normalize_features(raw_data, dataset):

    """
    """

    if raw_data['normalization']:

        if not raw_data['silent']:
        
            # tell us what we do
            print('Normalizing features')

        # get min-max scaler from the sklearn preprocessing package
        min_max_scaler = preprocessing.MinMaxScaler()

        # normalize x_t in the case that it is not OHE
        if raw_data['time_encoding'] != 'OHE':
            dataset['x_t'] = min_max_scaler.fit_transform(dataset['x_t'])

        # normalize x_s
        for column in range(dataset['x_s'].shape[1]):
            dataset['x_s'][:, column, :] = min_max_scaler.fit_transform(
                dataset['x_s'][:, column, :]
            )

    return dataset

def standardize_features(
    raw_data, 
    dataset, 
    reference_data, 
):

    """ Converts the population of each feature into a standard score using mean 
    and std deviations. For x_st, the past time steps of each meteorological 
    condition are transformed separately. For x_s1, the histogram or average 
    values of each channel are transformed separately.
    """

    if raw_data['standardization']:

        if not raw_data['silent']:

            # tell us what we do
            print('Standardizing data')

        # get StandardScaler from the sklearn preprocessing package
        standard_scaler = preprocessing.StandardScaler()

        # standardize x_t in the case that it is not OHE
        if raw_data['time_encoding'] != 'OHE':

            standard_scaler.fit(reference_data['x_t'])
            dataset['x_t'] = standard_scaler.transform(dataset['x_t'])


        # standardize x_s1
        for column in range(dataset['x_s'].shape[1]):

            standard_scaler.fit(reference_data['x_s'][:, column, :])
            dataset['x_s'][:, column, :] = standard_scaler.transform(
                dataset['x_s'][:, column, :]
            )

    return dataset
    
def encode_time_features(raw_data, dataset):

    """ 
    """

    if not raw_data['silent']:

        # tell us what we do
        print('Encoding temporal features')
        print('x_t before:', dataset['x_t'][0])


    ###
    #  If chosen so, transform encoding here ###
    ###

    if raw_data['time_encoding'] == 'OHE':

        # get OHE encoder
        enc = preprocessing.OneHotEncoder()

        # fit encoder
        enc.fit(dataset['x_t'])

        # encode temporal features
        dataset['x_t'] = enc.transform(dataset['x_t']).toarray().astype(int)

    if not raw_data['silent']:
        print('x_t after: {} ({})'.format(dataset['x_t'][0], raw_data['time_encoding']))

    return dataset
    
    
def create_feature_label_pairs(
    city_zone_coordinates,
    travel_data
):
    
    """
    """
    
    rename_sourceid = {
        'sourceid': 'zone_id'
    }
    rename_dstid = {
        'dstid': 'zone_id'
    }
    travel_data.rename(columns=rename_sourceid, inplace=True)
    travel_data = pd.merge(travel_data, city_zone_coordinates, on='zone_id')
    travel_data.drop(columns=['zone_id'], inplace=True)
    travel_data.rename(columns=rename_dstid, inplace=True)
    travel_data = pd.merge(travel_data, city_zone_coordinates, on='zone_id')
    travel_data.drop(columns=['zone_id'], inplace=True)
    
    x_t = travel_data['hod'].values
    x_t_ord_1D = x_t
    x_t = np.expand_dims(x_t, axis=1)
    x_s = travel_data[
        [
            'zone_lat_x', 
            'zone_long_x', 
            'zone_lat_y', 
            'zone_long_y'
        ]
    ].values
    x_s = np.expand_dims(x_s, axis=2)
    y = travel_data[
        [
            'mean_travel_time', 
            'standard_deviation_travel_time', 
            'geometric_mean_travel_time', 
            'geometric_standard_deviation_travel_time'
        ]
    ].values
    
    dataset = {
        'x_t_ord_1D': x_t_ord_1D,
        'x_t': x_t,
        'x_s': x_s,
        'y': y,
        'n_datapoints': len(y)
    }
    
    return dataset
    
    
def split_avail_cand(raw_data, dataset):

    """
    """
    
    if not raw_data['silent']:
        # tell us what we are doing
        print('Splitting data into available and candidate sets.')
        
        
    split_array = np.random.choice(
        [0, 1], 
        size=(dataset['n_datapoints'], ), 
        p=[1-raw_data['test_split'], raw_data['test_split']]
    ).astype(bool)
        
    avail_data = {}
    cand_data = {}
    for item in dataset:
        if item != 'n_datapoints':
            cand_data[item] = dataset[item][split_array]
            avail_data[item] = dataset[item][np.invert(split_array)]
        
    return (
        avail_data,
        cand_data
    )
    
def import_geojson(path_to_json_data):

    """ Imports the geojson data from the passed path and maps Uber Movement
    city zone IDs to a flattened list of latitude and longitude coordinates
    in the format of two dictionaries. Uses the recursive function called
    foster_coordinates_recursive to flatten the differently nested data.
    """
    
    data = pd.read_json(path_to_json_data)
    data.pop('type')
    data = data['features']
    
    map_json_entry_to_movement_id = dict()

    for json_id, json_entry in enumerate(data):
        
        map_json_entry_to_movement_id[json_id] = int(
          json_entry['properties']['MOVEMENT_ID']
        )
    
    map_movement_id_to_latitude_coordinates = dict()
    map_movement_id_to_longitude_coordinates = dict()

    for k, v in map_json_entry_to_movement_id.items():
        map_movement_id_to_latitude_coordinates[v] = []
        map_movement_id_to_longitude_coordinates[v] = []


    for json_id, movement_id in map_json_entry_to_movement_id.items():
        coordinates = data[json_id]['geometry']['coordinates']
        
        (
            map_movement_id_to_latitude_coordinates, 
            map_movement_id_to_longitude_coordinates
        ) = foster_coordinates_recursive(
            movement_id,
            map_movement_id_to_latitude_coordinates,
            map_movement_id_to_longitude_coordinates,
            coordinates
        )
        
    
    map_movement_id_to_coordinates = (
        map_movement_id_to_latitude_coordinates,
        map_movement_id_to_longitude_coordinates
    )

    return map_movement_id_to_coordinates
    
    
def foster_coordinates_recursive(
    movement_id,
    map_movement_id_to_latitude_coordinates,
    map_movement_id_to_longitude_coordinates,
    coordinates
):

    """ Flattens the coordinates of a passed city zone id (movement_id)
    and coordiates list recursively and saves their numeric values
    in the dictionaries that map movement ids to a list of latitude and 
    longitude coordinates.
    """

    dummy = 0

    for j in coordinates:

        if type(j) != list and dummy == 0:

            map_movement_id_to_longitude_coordinates[movement_id].append(j)
            dummy = 1
            continue

        elif type(j) != list and dummy == 1:

            map_movement_id_to_latitude_coordinates[movement_id].append(j)
            break

        else:

            dummy = 0
            coordinates = j
            (
                map_movement_id_to_latitude_coordinates,
                map_movement_id_to_longitude_coordinates
            ) = foster_coordinates_recursive(
                movement_id,
                map_movement_id_to_latitude_coordinates,
                map_movement_id_to_longitude_coordinates,
                coordinates
            )

    map_movement_id_to_coordinates = (
        map_movement_id_to_latitude_coordinates,
        map_movement_id_to_longitude_coordinates
    )

    return map_movement_id_to_coordinates


def calc_centroids(
    map_movement_id_to_latitude_coordinates,
    map_movement_id_to_longitude_coordinates
):

    """ Calculates the centroid of passed city zone polygons. Should a city
    zone consist of unregularities or multiple polygons, this is identified
    by centroid coordinates that are not within the bound of minimum and 
    maximum values of all coordinates of that city zone. In this case, the
    centroids are replaced with the mean of lat and long coordinates.
    """
    
    # create empty dictionary for mapping Uber Movement IDs to city zone areas
    map_movement_id_to_cityzone_area = dict()

    # iterate over all movement IDs and latitude coordinates
    for movement_id, lat_coordinates in map_movement_id_to_latitude_coordinates.items():
        
        # get also the longitude coordinates
        long_coordinates = map_movement_id_to_longitude_coordinates[movement_id]
        
        # calculate currently iterated city zone area
        area_cityzone = 0
        for i in range(len(lat_coordinates)-1):

            area_cityzone = (
                area_cityzone
                + long_coordinates[i] * lat_coordinates[i+1]
                - long_coordinates[i+1] * lat_coordinates[i]
            )
      
        area_cityzone = (
            area_cityzone
            + long_coordinates[i+1] * lat_coordinates[0]
            - long_coordinates[0] * lat_coordinates[i+1]
        )
        
        area_cityzone *= 0.5
        #area_cityzone = abs(area_cityzone)
        
        map_movement_id_to_cityzone_area[movement_id] = area_cityzone
        
    # create empty dictionaries for mapping Uber Movement IDs to city zone centroids
    map_movement_id_to_centroid_lat = dict()
    map_movement_id_to_centroid_long = dict()
        
    # iterate over all movement IDs and latitude coordinates
    for movement_id, lat_coordinates in map_movement_id_to_latitude_coordinates.items():
        
        # get also the longitude coordinates
        long_coordinates = map_movement_id_to_longitude_coordinates[movement_id]
        
        
        # calculate currently iterated city zone area
        centroid_lat = 0
        centroid_long = 0
        for i in range(len(lat_coordinates)-1):
            
            centroid_long += (
                long_coordinates[i]
                + long_coordinates[i+1]
            ) * (
                long_coordinates[i] * lat_coordinates[i+1]
                - long_coordinates[i+1] * lat_coordinates[i]
            )

            centroid_lat += (
                lat_coordinates[i]
                + lat_coordinates[i+1]
            ) * (
                long_coordinates[i] * lat_coordinates[i+1]
                - long_coordinates[i+1] * lat_coordinates[i]
            )

        centroid_long += (
            long_coordinates[i+1]
            + long_coordinates[0]
        ) * (
            long_coordinates[i+1] * lat_coordinates[0]
            - long_coordinates[0] * lat_coordinates[i+1]
        )
        
        centroid_lat += (
                lat_coordinates[i+1]
                + lat_coordinates[0]
            ) * (
                long_coordinates[i+1] * lat_coordinates[0]
                - long_coordinates[0] * lat_coordinates[i+1]
            )
        

        centroid_lat /= (
            6 * map_movement_id_to_cityzone_area[movement_id]
        )
        centroid_long /= (
            6 * map_movement_id_to_cityzone_area[movement_id]
        )
     
        # Uber Movement city zones sometimes consist of multiple distinct polygons
        if (
            centroid_lat < min(lat_coordinates)
            or centroid_lat > max(lat_coordinates)
            or centroid_long < min(long_coordinates)
            or centroid_long > max(long_coordinates)
        ):
            # in this case we calculate the mean instead of centroid
            centroid_lat = np.mean(lat_coordinates)
            centroid_long = np.mean(long_coordinates)            
        
        map_movement_id_to_centroid_lat[movement_id] = centroid_lat
        map_movement_id_to_centroid_long[movement_id] = centroid_long
        
    map_movement_id_to_centroid_coordinates = (
        map_movement_id_to_centroid_lat,
        map_movement_id_to_centroid_long
    )
    
    return map_movement_id_to_centroid_coordinates
    
def create_city_zone_coordinates(path_to_json_data):

    """ Calls the functions import_geojson and calc_centroids to get
    a list of city cone coordinates and created a pandas DataFrame in the 
    required format, i.e. wiht columns zone_lat and zone_long, as well as
    zone_id as index column
    """

    # map Uber Movement zone IDs to lists of lats and longs
    (
        map_movement_id_to_latitude_coordinates,
        map_movement_id_to_longitude_coordinates
    ) = import_geojson(path_to_json_data)

    # calculate centroids of city zone polygons
    (
        map_movement_id_to_centroid_lat,
        map_movement_id_to_centroid_long
    ) = calc_centroids(
       map_movement_id_to_latitude_coordinates,
        map_movement_id_to_longitude_coordinates
    )

    # create a pandas Dataframe in required format for bevpo
    city_zone_coordinates = pd.DataFrame(
        list(map_movement_id_to_centroid_lat.items()), 
        columns = ['zone_id','zone_lat']
    )
    
    city_zone_coordinates['zone_long'] = (
        map_movement_id_to_centroid_long.values()
    )
    
    # set the zone_id column as index
    city_zone_coordinates.set_index('zone_id', inplace=True)

    return city_zone_coordinates
    
