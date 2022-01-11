import math
import random

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn import preprocessing 


def prep_load_forecasting_data(
    path_to_data='data/public/electric load forecasting/',
    dataset_name='profiles_100',
    label_type='feature_scaled',
    spatial_features='histogram',
    meteo_types=[
        'air_density',
        'cloud_cover',
        'precipitation',
        'radiation_surface',
        'radiation_toa',
        'snow_mass',
        'snowfall',
        'temperature',
        'wind_speed',
    ],
    timestamp_data=[
        '15min', 
        'hour', 
        'day', 
        'month'
    ],
    time_encoding='ORD',
    histo_bins=100,
    grey_scale=False,
    profiles_per_year=1,
    points_per_profile=0.003,
    history_window_meteo=24,
    prediction_window=96,
    test_split=0.7,
    normalization=True,
    standardization=True,
    silent=True,
    plot=False,
):

    """ Imports data, creates feature-label pairs, normalizes data, splits it into
    available data for training and validation, as well as into candidate data
    consisting of spatial, temporal and spatio-temporal prediction tasks. After
    splitting data, it is standardized.
    """
    
    
    ### Create all required paths to where chosen data is stored
    path_to_data += dataset_name
    profile_years = ['2014']
    path_to_building_year_profile_folder = (
        path_to_data
        + '/building-year profiles/'
        + label_type
        + '/'
    )
    path_to_aerial_imagery_folder = (
        path_to_data 
        + '/building imagery/'
    )
    path_to_meteo_data_folder = (
        path_to_data 
        + '/meteo data/'
    )

    if spatial_features == 'histogram':
          path_to_aerial_imagery_folder += 'histogram/'
    elif spatial_features == 'average':
          path_to_aerial_imagery_folder += 'average/'
      
    if grey_scale:
        path_to_aerial_imagery_folder += 'greyscale/'
        n_channels = 1
    else:
        path_to_aerial_imagery_folder += 'rgb/'
        n_channels = 3
    
    
    ### Save all parameters in dictionary
    raw_data = {
        'path_to_data': path_to_data,
        'profile_years': profile_years,
        'path_to_building_year_profile_folder': path_to_building_year_profile_folder,
        'path_to_aerial_imagery_folder': path_to_aerial_imagery_folder,
        'path_to_meteo_data_folder': path_to_meteo_data_folder,
        'profiles_per_year': profiles_per_year,
        'points_per_profile': points_per_profile,
        'n_subplots': 10,
        'histo_bins': histo_bins,
        'spatial_features': spatial_features,
        'n_channels': n_channels,
        'meteo_types': meteo_types,
        'history_window_meteo': history_window_meteo,
        'prediction_window': prediction_window,
        'timestamp_data': timestamp_data,
        'time_encoding': time_encoding,
        'test_split': test_split,
        'normalization': normalization,
        'standardization': standardization,
        'silent': silent,
        'plot': plot
    }
    
    
    ### Import data
    raw_data = import_consumption_profiles(raw_data)
    raw_data = import_building_images(raw_data)
    raw_data = import_meteo_data(raw_data)
    
    
    ### Pair features and labels
    dataset, raw_data = create_feature_label_pairs(raw_data)
    
    
    ### Encode temporal features
    dataset = encode_time_features(raw_data, dataset)
    
    
    ### Normalize all data
    dataset = normalize_features(raw_data, dataset)
    
    
    ### Split data into train_val and various testing datasets
    (
        train_val_data, 
        cand_data_spatial, 
        cand_data_temporal, 
        cand_data_spatemp
    ) = split_avail_cand(raw_data, dataset)
    
    
    ### Standardize features
    cand_data_spatial = standardize_features(
        raw_data, 
        cand_data_spatial, 
        train_val_data
    )
    cand_data_temporal = standardize_features(
        raw_data, 
        cand_data_temporal, 
        train_val_data
    )
    cand_data_spatemp = standardize_features(
        raw_data, 
        cand_data_spatemp, 
        train_val_data
    )
    train_val_data = standardize_features(
        raw_data, 
        train_val_data, 
        train_val_data
    )
    
    
    ### simplify return datasets by removing x_s and only giving x_s1
    train_val_data['x_s'] = train_val_data['x_s1']
    cand_data_spatemp['x_s'] = cand_data_spatemp['x_s1']
    del train_val_data['x_s1']
    del cand_data_spatemp['x_s1']
    
    
    ### Create return datasets
    datasets = {
        'avail_data': train_val_data, 
        'cand_data': cand_data_spatemp
    }
    
    return datasets


def import_consumption_profiles(raw_data):

    """ Imports building-scale electric consumption ground truth data.
    """
    
    if not raw_data['silent']:
        # tell us what we are doing
        print('Importing consumption profiles')

        # create a progress bar
        progbar = tf.keras.utils.Progbar(len(raw_data['profile_years']))
    
    # save dataframes here instead of under distinct names
    building_year_profiles_list = []
    memory_demand_GB = 0
    
    
    # iterate over the list of years for which we want to import load profiles
    for index_year, year in enumerate(raw_data['profile_years']):
        # get the path to currently iterated building-year profiles file
        path_to_building_year_profile_files = (
            raw_data['path_to_building_year_profile_folder']
            + str(year)
            + ' building-year profiles.csv'
        )
        
        # load currently iterated file
        df = pd.read_csv(path_to_building_year_profile_files)
    
         # get the building IDs of profiles
        building_ids = df.columns.values[1:]

        # get the cluster IDs of profiles and drop the row
        cluster_ids = df.iloc[0, 1:].values.astype(int)

        # get the years of profiles and replace them with the year ID used here
        years = df.iloc[1, 1:].values.astype(int)
        year_ids = years
        year_ids[:] = index_year

        # drop the cluder id and year rows
        df = df.drop([0, 1])

        # rename the 'building ID' column name to 'local_time' so as to match 
        # the meteo files' column name for search later
        df = df.rename(columns={'building ID': 'local_time'})

        # get the time stamp of the imported meters
        time_stamp_profiles = df.pop('local_time')

        # set the new time stamp as index
        df = df.set_index(time_stamp_profiles)

        # create a random array
        randomize = np.arange(len(building_ids))
        np.random.shuffle(randomize)

        # shuffle ID orders with same random array
        building_ids = building_ids[randomize]
        cluster_ids = cluster_ids[randomize]
        year_ids = year_ids[randomize]
        
        # shorten the considered ID lists according to your chosen number of  
        # considerable profiles per year
        n_profiles = math.ceil(raw_data['profiles_per_year'] * len(building_ids))
        building_ids = building_ids[: n_profiles]
        cluster_ids = cluster_ids[: n_profiles]
        year_ids = year_ids[: n_profiles]
        
        # shorten dataframe accordingly
        df = df[building_ids]
        
        # check if first iteration
        if year == raw_data['profile_years'][0]:

            # if yes, set the id lists equal to currently iterated lists
            building_id_list = building_ids
            cluster_id_list = cluster_ids
            year_id_list = year_ids

        else:

            # if not, concatenate previous lists with currently iterated lists
            building_id_list = np.concatenate((building_id_list, building_ids))
            cluster_id_list = np.concatenate((cluster_id_list, cluster_ids))
            year_id_list = np.concatenate((year_id_list, year_ids))
            
        # append dataframe
        building_year_profiles_list.append(df)

        # accumulate the memory demand of building-year profiles we imported
        memory_demand_GB = memory_demand_GB + df.memory_usage().sum() * 1e-9
        
        if not raw_data['silent']:
            # increment the progress bar
            progbar.add(1)
            
    # get the set of building IDs, i.e. drop the duplicate entries
    building_id_set = set(building_id_list)

    # get the set of building IDs, i.e. drop the duplicate entries
    cluster_id_set = set(cluster_id_list)

    # get the set of year IDs. Note: this should be equal to profile_years
    year_id_set = set(year_id_list)

    # get set of cluster-year ID combinations
    cluster_year_set = set(list(zip(cluster_id_list, year_id_list)))

    raw_data['building_year_profiles_list'] = building_year_profiles_list
    raw_data['building_id_list'] = building_id_list
    raw_data['cluster_id_list'] = cluster_id_list
    raw_data['year_id_list'] = year_id_list
    raw_data['building_id_set'] = building_id_set
    raw_data['cluster_id_set'] = cluster_id_set
    raw_data['year_id_set'] = year_id_set
    raw_data['cluster_year_set'] = cluster_year_set

    # Tell us how much RAM we are occupying with the just imported profiles
    if not raw_data['silent']:
        print(
            'The',
            len(building_id_list),
            'imported electric load profiles demand a total amount of',
            memory_demand_GB,
            'GB of RAM',
        )

    if raw_data['plot']:

        # set the number of subplots to the minimum of the desired value and the  
        # actually available profiles for plotting
        n_subplots = min(raw_data['n_subplots'], len(df.columns))

        # Visualize some profiles
        _ = df.iloc[:, :n_subplots].plot(
            title='Exemplar electric load profiles (labels/ground truth data)',
            subplots=True,
            layout=(math.ceil(n_subplots / 2), 2),
            figsize=(16, n_subplots),
        )
        
    return raw_data
    
def import_building_images(raw_data):

    """ Imports histogram values of building image pixels.
    """

    if not raw_data['silent']:

        # tell us what we do
        print('Importing building-scale aerial imagery:')

        # create a progress bar
        progbar = tf.keras.utils.Progbar(len(raw_data['building_id_set']))

        # create a variabl to iteratively add the memory of imported files
        memory_demand_GB = 0

    # create a empty lists for aerial image data and building ids
    building_imagery_data_list = []
    building_imagery_id_list = []

    # create path to imagery data file
    path_to_file = (
        raw_data['path_to_aerial_imagery_folder'] 
        + 'pixel_values.csv'
    )

    # import building imagery data
    df = pd.read_csv(path_to_file)

    # iterate over set of building IDs
    for building_id in raw_data['building_id_set']:
        
        # get the pixel features of currently iterated building image
        imagery_pixel_data = df[building_id].values
        
        ### reshape image pixel values a shape with channels last ###

        # get the number of features per image pixel array channel
        if raw_data['spatial_features'] == 'average':
            n_features = 1
            
        elif raw_data['spatial_features'] == 'histogram':
            n_features = raw_data['histo_bins']

        # reshape image with Fortran method. This is method used to flatten.
        imagery_pixel_data = np.reshape(
            imagery_pixel_data, 
            (n_features, raw_data['n_channels']), 
            order='F'
        )

        # add values to lists
        building_imagery_data_list.append(imagery_pixel_data)
        building_imagery_id_list.append(int(building_id))

        if not raw_data['silent']:

            # Accumulate the memory demand of each image
            memory_demand_GB += imagery_pixel_data.nbytes * 1e-9

            # increment progress bar
            progbar.add(1)


    if not raw_data['silent']:

        # Tell us how much RAM we occupy with the just imported data files
        print(
            'The',
            len(building_imagery_data_list),
            'aerial images demand',
            memory_demand_GB,
            'GB RAM with float32 entries',
        )
 
    # add to raw_data instance
    raw_data['building_imagery_data_list'] = building_imagery_data_list
    raw_data['building_imagery_id_list'] = building_imagery_id_list

    return raw_data
    

def import_meteo_data(raw_data):

    """ Imports space-time variant meteorological data.
    """

    if not raw_data['silent']:

        # tell us what we do
        print('Importing meteorological data')

        # create a variabl to iteratively add the memory demand of each file
        memory_demand_GB = 0

        # create a progress bar
        progbar = tf.keras.utils.Progbar(len(raw_data['cluster_year_set']))

    # create list for saving meteo data
    meteo_data_list = []

    # create array for saving corresponding cluster and year IDs of meteo files
    # that are added to the list
    meteo_data_cluster_year_array = np.zeros(
        (
            len(raw_data['cluster_year_set']), 
            2
        )
    )

    # use counter for meta data array
    counter = 0

    # iterate over each file in the list of all meteo files
    for cluster_id, year_id in raw_data['cluster_year_set']:

        file_name = (
            'meteo_'
            + str(cluster_id)
            + '_'
            + str(int(raw_data['profile_years'][year_id]))
            + '.csv'
        )

        # create the entire path to the currently iterated file
        path_to_file = raw_data['path_to_meteo_data_folder'] + file_name

        # load file
        df = pd.read_csv(path_to_file)

        # set one of the columns 'local_time' as index for later search purposes
        df = df.set_index('local_time')

        # shorten dataframe according to the meteo data types that you chose
        df = df[raw_data['meteo_types']]

        # append to list
        meteo_data_list.append(df)

        # append to list
        meteo_data_cluster_year_array[counter] = (cluster_id, year_id)

        # increment
        counter += 1

        if not raw_data['silent']:
            # Accumulate the memory demand of each file
            memory_demand_GB += df.memory_usage().sum() * 1e-9

            # increment progress bar
            progbar.add(1)

    raw_data['meteo_data_list'] = meteo_data_list
    raw_data['meteo_data_cluster_year_array'] = meteo_data_cluster_year_array

    if not raw_data['silent']:

        # Tell us how much RAM we occupy with the just imported data files
        print(
            'The',
            len(raw_data['cluster_year_set']),
            'meteo data files demand',
            memory_demand_GB,
            'GB RAM',
        )

    if raw_data['plot']:

        # plot the time series data for each metering code
        _ = df.plot(
            title='Exemplar meteorological conditions (spatio-temporal features)',
            use_index=False,
            legend=True,
            figsize=(16, 16),
            fontsize=16,
            subplots=True,
            layout=(3, 3),
        )

    return raw_data    
    
    
def create_feature_label_pairs(raw_data):

    """ Pairs features and labels.
    """

    # determine start and end of iteration over each paired dataframe
    start = raw_data['history_window_meteo'] * 4
    end = (
        len(raw_data['building_year_profiles_list'][0]) 
        - raw_data['prediction_window']
    )
    n_points = math.ceil(
        raw_data['points_per_profile'] 
        * len(raw_data['building_year_profiles_list'][0])
    )
    step = math.ceil((end - start) / n_points)
    points_per_profile = math.ceil((end - start) / step)

    # Calculate how many data points we chose to consider in total
    n_datapoints = len(raw_data['building_id_list']) * points_per_profile

    # Create empty arrays in the right format for saving features and labels
    x_t = np.zeros((n_datapoints, 5))
    x_st = np.zeros(
        (
            n_datapoints, 
            raw_data['history_window_meteo'], 
            len(raw_data['meteo_types'])
        )
    )
    x_s = np.zeros((n_datapoints, 2))
    y = np.zeros((n_datapoints, raw_data['prediction_window']))

    # create a datapoint counter to increment and add to the data entries
    datapoint_counter = 0

    if not raw_data['silent']:

        # tell us what we do
        print('Creating feature label data pairs:')

        # create a progress bar
        progbar = tf.keras.utils.Progbar(n_datapoints)

    # iterate over the set of considered cluser-year ID combinations
    for cluster_id, year_id in raw_data['cluster_year_set']:

        # generate the respective cluster id and building id subsets
        building_id_subset = raw_data['building_id_list'][
            np.nonzero(
                (raw_data['year_id_list'] == year_id)
                & (raw_data['cluster_id_list'] == cluster_id)
            )
        ]

        # get the year in gregorian calendar here
        year = int(raw_data['profile_years'][year_id])

        # get the index of the meteo data list entry that correspondings to 
        # the currently iterated cluster-year ID combination
        index_meteo_data_list = np.where(
            (raw_data['meteo_data_cluster_year_array'][:, 0] == cluster_id)
            & (raw_data['meteo_data_cluster_year_array'][:, 1] == year_id)
        )[0][0]

        # create a new dataframe that merges the meteo values and load profile
        # values by index col 'local_time'
        paired_df = raw_data['building_year_profiles_list'][year_id][
            building_id_subset
        ].merge(raw_data['meteo_data_list'][index_meteo_data_list], on="local_time")

        # iterate over the paired dataframe
        for i in range(start, end, step):

            # get timestamp features
            month = paired_df.index[i][5:7]
            day = paired_df.index[i][8:10]
            hour = paired_df.index[i][11:13]
            minute_15 = paired_df.index[i][14:16]

            # get the meteo features. Note that you need to jump in hourly
            # steps back in time, hence all times 4
            meteo = paired_df.iloc[
                (i - (raw_data['history_window_meteo'] * 4)) : i : 4,
                -(len(raw_data['meteo_types'])) :,
            ]

            # iterate over each building id
            for building_id in building_id_subset:

                # get the label
                label = (
                    paired_df[[building_id]]
                    .iloc[i : (i + raw_data['prediction_window'])]
                    .values[:, 0]
                )

                # Add the features and labels to respective data point entry
                x_t[datapoint_counter, :] = [minute_15, hour, day, month, year]
                x_s[datapoint_counter, :] = [building_id, cluster_id]
                x_st[datapoint_counter, :, :] = meteo
                y[datapoint_counter, :] = label

                # increment datapoint counter
                datapoint_counter += 1

        if not raw_data['silent']:

            # increment progress bar
            progbar.add(points_per_profile * len(building_id_subset))


    ### Shorten x_t according to chosen TIMESTAMP_DATA ###

    # create empty list
    filter_list = []

    # check for all possible entries in correct order and add to filter list if 
    # not in chosen TIMESTAMP_DATA
    if '15min' not in raw_data['timestamp_data']:
        filter_list.append(0)
        
    if 'hour' not in raw_data['timestamp_data']:
        filter_list.append(1)
        
    if 'day' not in raw_data['timestamp_data']:
        filter_list.append(2)
        
    if 'month' not in raw_data['timestamp_data']:
        filter_list.append(3)
        
    if 'year' not in raw_data['timestamp_data']:
        filter_list.append(4)

    # delete the columns according to created filter_list
    x_t = np.delete(x_t, filter_list, 1)

    # get the minimum value for labels
    raw_data['y_min'] = y.min()

    # get the maximum value for labels
    raw_data['y_max'] = y.max()

    # get the full range of possible values
    raw_data['y_range'] = raw_data['y_max'] - raw_data['y_min']

    # bundle data as dataset object and return
    dataset = {
        'x_t': x_t,
        'x_s': x_s,
        'x_st': x_st,
        'y': y,
    }

    ### Process spatial features ###

    df_list = []

    # iterate over number of channels
    for i in range(raw_data['n_channels']):

        # create dataframe with one column 'building id' for iterated channel
        df_list.append(pd.DataFrame(columns=['building id']))

    # iterate over all building scale images and their building IDs
    for index, image in enumerate(raw_data['building_imagery_data_list']):

        building_id = raw_data['building_imagery_id_list'][index]

        for channel, df in enumerate(df_list):
           
            df_list[channel] = df_list[channel].append(
                pd.Series(image[:, channel]), ignore_index=True
            )

            df_list[channel].iloc[index, 0] = building_id

    # create empty x_s1
    dataset['x_s1'] = np.zeros(
        (
            len(dataset['y']), 
            image.shape[0], 
            image.shape[1]
        )
    )

    # iterate over number of channels
    for i in range(raw_data['n_channels']):

        # merge the columns of building ID in x_s and the new dataframe
        paired_df = pd.DataFrame(
            dataset['x_s'], 
            columns=['building id', 'cluster id']
        ).merge(
            df_list[i], 
            on='building id', 
            how='left'
        )

        # pass the paired values to x_s1
        dataset['x_s1'][:, :, i] = paired_df.iloc[:, 2:].values

    return dataset, raw_data
    
    
def encode_time_features(raw_data, dataset):

    """ Gives you one-hot encoded time stamps if chosen so.
    """

    if not raw_data['silent']:

        # tell us what we do
        print('Encoding temporal features')
        print('x_t before:', dataset['x_t'][0])


    ###
    # Ordinally encode all available time stamp dimensions ###
    ###

    # get OrdinalEncoder from sklearn.preprocessing
    enc = preprocessing.OrdinalEncoder()

    # fit the encoder to x_t
    enc.fit(dataset['x_t'])

    # encode x_t
    dataset['x_t'] = enc.transform(dataset['x_t']).astype(int)

    # save the encoded feature categories for x_time
    timestamp_categories = enc.categories_

    ###
    # Create one dimensional ordinal encoding in 1-min steps ###
    ###

    # create an empty array for adding up values
    dataset['x_t_ord_1D'] = np.zeros((len(dataset['y']),))
    x_t_copy = dataset['x_t']

    # check for all possible entries
    if '15min' in raw_data['timestamp_data']:
        dataset['x_t_ord_1D'] += x_t_copy[:, 0] * 15
        x_t_copy = np.delete(x_t_copy, 0, 1)

    if 'hour' in raw_data['timestamp_data']:
        dataset['x_t_ord_1D'] += x_t_copy[:, 0] * 60
        x_t_copy = np.delete(x_t_copy, 0, 1)

    if 'day' in raw_data['timestamp_data']:
        dataset['x_t_ord_1D'] += x_t_copy[:, 0] * 60 * 24
        x_t_copy = np.delete(x_t_copy, 0, 1)

    if 'month' in raw_data['timestamp_data']:
        dataset['x_t_ord_1D'] += x_t_copy[:, 0] * 60 * 24 * 31
        x_t_copy = np.delete(x_t_copy, 0, 1)

    if 'year' in raw_data['timestamp_data']:
        dataset['x_t_ord_1D'] += x_t_copy[:, 0] * 60 * 24 * 31 * 12
        x_t_copy = np.delete(x_t_copy, 0, 1)

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

    elif raw_data['time_encoding'] == 'ORD-1D':

        # copy the 1D ordinal array to x_t
        dataset['x_t'] = dataset['x_t_ord_1D']

        # expand the last dimension for NN input fit
        dataset['x_t'] = np.expand_dims(dataset['x_t'], axis=1)

    if not raw_data['silent']:

        print('x_t after: {} ({})'.format(dataset['x_t'][0], raw_data['time_encoding']))

    return dataset
    
def normalize_features(raw_data, dataset):

    """ Min max scales data.
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

        # normalize x_st
        for i in range(len(raw_data['meteo_types'])):
            dataset['x_st'][:, :, i] = min_max_scaler.fit_transform(
                dataset['x_st'][:, :, i]
            )

        # normalize x_s1
        for channel in range(raw_data['n_channels']):
            dataset['x_s1'][:, :, channel] = min_max_scaler.fit_transform(
                dataset['x_s1'][:, :, channel]
            )

    return dataset
  
def split_avail_cand(raw_data, dataset):

    """ Splits data into available and candidate datasets.
    """

    if not raw_data['silent']:
        # tell us what we are doing
        print('Splitting data into training, validation and testing sets.')

    ###
    # Sort arrays in ascending temporal order ###
    ###

    sort_array = np.argsort(dataset['x_t_ord_1D'])
    for key in dataset:
        dataset[key] = dataset[key][sort_array]

    ###
    # Take away data from both ends of sorted arrays ###
    ###

    # get the number of datapoints to cut out for temporal prediction tests
    split_point = math.ceil(raw_data['test_split'] / 2 * len(dataset['y']))
    
    temporal_test_data = {}
    for key in dataset:
        temporal_test_data[key] = dataset[key][:split_point]
        dataset[key] = dataset[key][split_point:]
        temporal_test_data[key] = np.concatenate(
            (
                temporal_test_data[key], 
                dataset[key][-split_point:]
            )
        )
        dataset[key] = dataset[key][:-split_point]


    ###
    # Set the remaining data as spatial dataset ###
    ###

    # get number of buildings you want to randomly choose from
    n_test_buildings = math.ceil(
        raw_data['test_split'] * len(raw_data['building_id_set'])
    )

    # randomly choose some buildings for testing
    test_building_samples = random.sample(
        raw_data['building_id_set'], 
        k=n_test_buildings
    )

    # transform building ID strings to integers
    test_building_samples = [int(x) for x in test_building_samples]

    spatial_test_data = {}
    for key in dataset:
        spatial_test_data[key] = dataset[key]
        dataset[key] = 0


    ###
    # Extract temporal and spatio-temporal test sets ###
    ###

    ### create the filtering array ###
    boolean_filter_array = np.zeros((len(temporal_test_data['y']),), dtype=bool)

    for building_id in test_building_samples:
        boolean_filter_array = boolean_filter_array | (
            temporal_test_data['x_s'][:, 0] == building_id
        )

    inverted_boolean_filter_array = np.invert(boolean_filter_array)

    ### Spatio-temporal and temporal ###
    spatemp_test_data = {}
    for key in temporal_test_data:
        spatemp_test_data[key] = temporal_test_data[key][boolean_filter_array]
        temporal_test_data[key] = temporal_test_data[key][inverted_boolean_filter_array]


    ###
    # Extract spatial test set ###
    ###

    ### create the filtering array ###
    boolean_filter_array = np.zeros((len(spatial_test_data['y']),), dtype=bool)

    for building_id in test_building_samples:
        boolean_filter_array = (
            boolean_filter_array | (spatial_test_data['x_s'][:, 0] == building_id)
        )

    inverted_boolean_filter_array = np.invert(boolean_filter_array)

    ### Train-validation split ###
    train_val_data = {}
    for key in spatial_test_data:
        train_val_data[key] = spatial_test_data[key][inverted_boolean_filter_array]
        spatial_test_data[key] = spatial_test_data[key][boolean_filter_array]
   

    def f_randomize(dataset):
    
        """ Randomizes all entries of the passed dataset dictionary.
        """
        # create random array
        random_array = np.arange(len(dataset['x_t']))

        # shuffle random array
        np.random.shuffle(random_array)

        for key in dataset:
            dataset[key] = dataset[key][random_array]

        return dataset
        
    
    train_val_data = f_randomize(train_val_data)
    spatial_test_data = f_randomize(spatial_test_data)
    temporal_test_data = f_randomize(temporal_test_data)
    spatemp_test_data = f_randomize(spatemp_test_data)
     
    if not raw_data['silent']:

        n_test_datapoints = (
            len(spatial_test_data['y'])
            + len(temporal_test_data['y'])
            + len(spatemp_test_data['y'])
        )
        n_total_datapoints = (
            len(train_val_data['y'])
            + n_test_datapoints
        )

        print(
            ' With test_split =',
            raw_data['test_split'],
            'the data is split in the following ratio:',
        )
        print('---' * 38)

        print(
            'Available data:   {} ({:.0%})'.format(
                len(train_val_data['y']),
                len(train_val_data['y']) / n_total_datapoints,
            )
        )
        print(
            'Candidate data:    {} ({:.0%})'.format(
                n_test_datapoints, n_test_datapoints / n_total_datapoints
            )
        )
        print('---' * 38)

        print(
            'Spatial testing data:         {} ({:.0%})'.format(
                len(spatial_test_data['y']),
                len(spatial_test_data['y']) / n_test_datapoints,
            )
        )
        print(
            'Temporal testing data:        {} ({:.0%})'.format(
                len(temporal_test_data['y']),
                len(temporal_test_data['y']) / n_test_datapoints,
            )
        )
        print(
            'Spatio-temporal testing data: {} ({:.0%})'.format(
                len(spatemp_test_data['y']),
                len(spatemp_test_data['y']) / n_test_datapoints,
            )
        )

    
    return (
        train_val_data,
        spatial_test_data,
        temporal_test_data,
        spatemp_test_data,
    )

  
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

        # standardize x_st
        for i in range(len(raw_data['meteo_types'])):

            standard_scaler.fit(reference_data['x_st'][:, :, i])
            dataset['x_st'][:, :, i] = standard_scaler.transform(
                dataset['x_st'][:, :, i]
            )

        # standardize x_s1
        for channel in range(raw_data['n_channels']):

            standard_scaler.fit(reference_data['x_s1'][:, :, channel])
            dataset['x_s1'][:, :, channel] = standard_scaler.transform(
                dataset['x_s1'][:, :, channel]
            )

    return dataset
