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

    """
    """
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
        'standardization': standardization
    }
    
    raw_data = import_consumption_profiles(raw_data, silent=silent, plot=plot)
    raw_data = import_building_images(raw_data, silent=silent)
    raw_data = import_meteo_data(raw_data, silent=silent, plot=plot)
    dataset, raw_data = create_feature_label_pairs(raw_data, silent=silent)
    dataset = encode_time_features(raw_data, dataset, silent=silent)
    dataset = normalize_features(raw_data, dataset, silent=silent)
    (
        avail_data, 
        cand_data_spatial, 
        cand_data_temporal, 
        cand_data_spatemp
    ) = split_avail_cand(raw_data, dataset, silent=silent)
    

    cand_data_spatial = standardize_features(
        raw_data, 
        cand_data_spatial, 
        avail_data, 
        silent=silent
    )
    cand_data_temporal = standardize_features(
        raw_data, 
        cand_data_temporal, 
        avail_data, 
        silent=silent
    )
    cand_data_spatemp = standardize_features(
        raw_data, 
        cand_data_spatemp, 
        avail_data, 
        silent=silent
    )
    avail_data = standardize_features(
        raw_data, 
        avail_data, 
        avail_data, 
        silent=silent
    )
    
    datasets = (
        avail_data, 
        cand_data_spatial, 
        cand_data_temporal, 
        cand_data_spatemp
    )
    
    return datasets



def import_consumption_profiles(
    raw_data,
    silent=True,
    plot=False
):

    """
    """
    
    if not silent:
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
        
        if not silent:
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
    if not silent:
        print(
            'The',
            len(building_id_list),
            'imported electric load profiles demand a total amount of',
            memory_demand_GB,
            'GB of RAM',
        )

    if plot:

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
    
def import_building_images(
    raw_data, 
    silent=False
):

    """ 
    """

    if not silent:

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

        if not silent:

            # Accumulate the memory demand of each image
            memory_demand_GB += imagery_pixel_data.nbytes * 1e-9

            # increment progress bar
            progbar.add(1)


    if not silent:

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
    

def import_meteo_data(
    raw_data, 
    silent=False, 
    plot=True
):

    """ 
    """

    if not silent:

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

        if not silent:
            # Accumulate the memory demand of each file
            memory_demand_GB += df.memory_usage().sum() * 1e-9

            # increment progress bar
            progbar.add(1)

    raw_data['meteo_data_list'] = meteo_data_list
    raw_data['meteo_data_cluster_year_array'] = meteo_data_cluster_year_array

    if not silent:

        # Tell us how much RAM we occupy with the just imported data files
        print(
            'The',
            len(raw_data['cluster_year_set']),
            'meteo data files demand',
            memory_demand_GB,
            'GB RAM',
        )

    if plot:

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
    
    
def create_feature_label_pairs(
    raw_data, 
    silent=True
):

    """
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
    X_t = np.zeros((n_datapoints, 5))
    X_st = np.zeros(
        (
            n_datapoints, 
            raw_data['history_window_meteo'], 
            len(raw_data['meteo_types'])
        )
    )
    X_s = np.zeros((n_datapoints, 2))
    Y = np.zeros((n_datapoints, raw_data['prediction_window']))

    # create a datapoint counter to increment and add to the data entries
    datapoint_counter = 0

    if not silent:

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
                X_t[datapoint_counter, :] = [minute_15, hour, day, month, year]
                X_s[datapoint_counter, :] = [building_id, cluster_id]
                X_st[datapoint_counter, :, :] = meteo
                Y[datapoint_counter, :] = label

                # increment datapoint counter
                datapoint_counter += 1

        if not silent:

            # increment progress bar
            progbar.add(points_per_profile * len(building_id_subset))


    ### Shorten X_t according to chosen TIMESTAMP_DATA ###

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
    X_t = np.delete(X_t, filter_list, 1)

    # get the minimum value for labels
    raw_data['Y_min'] = Y.min()

    # get the maximum value for labels
    raw_data['Y_max'] = Y.max()

    # get the full range of possible values
    raw_data['Y_range'] = raw_data['Y_max'] - raw_data['Y_min']

    # bundle data as dataset object and return
    dataset = {
        'X_t': X_t,
        'X_s': X_s,
        'X_st': X_st,
        'Y': Y,
        'n_datapoints':len(X_t)
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

    # create empty X_s1
    dataset['X_s1'] = np.zeros(
        (
            dataset['n_datapoints'], 
            image.shape[0], 
            image.shape[1]
        )
    )

    # iterate over number of channels
    for i in range(raw_data['n_channels']):

        # merge the columns of building ID in X_s and the new dataframe
        paired_df = pd.DataFrame(
            dataset['X_s'], 
            columns=['building id', 'cluster id']
        ).merge(
            df_list[i], 
            on='building id', 
            how='left'
        )

        # pass the paired values to X_s1
        dataset['X_s1'][:, :, i] = paired_df.iloc[:, 2:].values

    return dataset, raw_data
    
def encode_time_features(
    raw_data,
    dataset,
    silent=True
):

    """ 
    """

    if not silent:

        # tell us what we do
        print('Encoding temporal features')
        print('X_t before:', dataset['X_t'][0])


    ###
    # Ordinally encode all available time stamp dimensions ###
    ###

    # get OrdinalEncoder from sklearn.preprocessing
    enc = preprocessing.OrdinalEncoder()

    # fit the encoder to X_t
    enc.fit(dataset['X_t'])

    # encode X_t
    dataset['X_t'] = enc.transform(dataset['X_t']).astype(int)

    # save the encoded feature categories for X_time
    timestamp_categories = enc.categories_

    # create empty matrix for saving number of categories of each feature column
    n_time_categories = np.zeros((len(enc.categories_))).astype(int)

    # iterate over each category array and save number of categories
    for index, category_array in enumerate(enc.categories_):

        # save number of respective category
        n_time_categories[index] = len(category_array)

    ###
    # Create one dimensional ordinal encoding in 1-min steps ###
    ###

    # create an empty array for adding up values
    dataset['X_t_ord_1D'] = np.zeros((dataset['n_datapoints'],))
    X_t_copy = dataset['X_t']

    # check for all possible entries
    if '15min' in raw_data['timestamp_data']:
        dataset['X_t_ord_1D'] += X_t_copy[:, 0] * 15
        X_t_copy = np.delete(X_t_copy, 0, 1)

    if 'hour' in raw_data['timestamp_data']:
        dataset['X_t_ord_1D'] += X_t_copy[:, 0] * 60
        X_t_copy = np.delete(X_t_copy, 0, 1)

    if 'day' in raw_data['timestamp_data']:
        dataset['X_t_ord_1D'] += X_t_copy[:, 0] * 60 * 24
        X_t_copy = np.delete(X_t_copy, 0, 1)

    if 'month' in raw_data['timestamp_data']:
        dataset['X_t_ord_1D'] += X_t_copy[:, 0] * 60 * 24 * 31
        X_t_copy = np.delete(X_t_copy, 0, 1)

    if 'year' in raw_data['timestamp_data']:
        dataset['X_t_ord_1D'] += X_t_copy[:, 0] * 60 * 24 * 31 * 12
        X_t_copy = np.delete(X_t_copy, 0, 1)

    ###
    #  If chosen so, transform encoding here ###
    ###

    if raw_data['time_encoding'] == 'OHE':

        # get OHE encoder
        enc = preprocessing.OneHotEncoder()

        # fit encoder
        enc.fit(dataset['X_t'])

        # encode temporal features
        dataset['X_t'] = enc.transform(dataset['X_t']).toarray().astype(int)

    elif raw_data['time_encoding'] == 'ORD-1D':

        # copy the 1D ordinal array to X_t
        dataset['X_t'] = dataset['X_t_ord_1D']

        # expand the last dimension for NN input fit
        dataset['X_t'] = np.expand_dims(dataset['X_t'], axis=1)

    if not silent:

        print('X_t after: {} ({})'.format(dataset['X_t'][0], raw_data['time_encoding']))

    return dataset
    
def normalize_features(
    raw_data, 
    dataset, 
    silent=True
):

    """
    """

    if raw_data['normalization']:

        if not silent:
        
            # tell us what we do
            print('Normalizing features')

        # get min-max scaler from the sklearn preprocessing package
        min_max_scaler = preprocessing.MinMaxScaler()

        # normalize X_t in the case that it is not OHE
        if raw_data['time_encoding'] != 'OHE':
            dataset['X_t'] = min_max_scaler.fit_transform(dataset['X_t'])

        # normalize X_st
        for i in range(len(raw_data['meteo_types'])):
            dataset['X_st'][:, :, i] = min_max_scaler.fit_transform(
                dataset['X_st'][:, :, i]
            )

        # normalize X_s1
        if raw_data['spatial_features'] != 'image':

            for channel in range(raw_data['n_channels']):
                dataset['X_s1'][:, :, channel] = min_max_scaler.fit_transform(
                    dataset['X_s1'][:, :, channel]
                )

    return dataset
  
def split_avail_cand(
    raw_data, 
    dataset, 
    silent=True
):

    """ 
    """

    if not silent:
        # tell us what we are doing
        print('Splitting data into training, validation and testing sets.')

    ###
    # Reduce memory demand ###
    ###

    dataset['X_t'] = np.float32(dataset['X_t'])
    dataset['X_st'] = np.float32(dataset['X_st'])
    dataset['Y'] = np.float32(dataset['Y'])
    dataset['X_s'] = dataset['X_s'].astype(int)
    dataset['X_s1'] = np.float32(dataset['X_s1'])

    ###
    # Sort arrays in ascending temporal order ###
    ###

    sort_array = np.argsort(dataset['X_t_ord_1D'])
    dataset['X_t'] = dataset['X_t'][sort_array]
    dataset['X_s'] = dataset['X_s'][sort_array]
    dataset['X_st'] = dataset['X_st'][sort_array]
    dataset['Y'] = dataset['Y'][sort_array]
    dataset['X_s1'] = dataset['X_s1'][sort_array]

    ###
    # Take away data from both ends of sorted arrays ###
    ###

    # get the number of datapoints to cut out for temporal prediction tests
    split_point = math.ceil(raw_data['test_split'] / 2 * dataset['n_datapoints'])

    ### extract data from beginning of temporaly sorted dataset ###
    temporal_X_t_ord_1D = dataset['X_t_ord_1D'][:split_point]
    dataset['X_t_ord_1D'] = dataset['X_t_ord_1D'][split_point:]
    
    temporal_X_t = dataset['X_t'][:split_point]
    dataset['X_t'] = dataset['X_t'][split_point:]

    temporal_X_s = dataset['X_s'][:split_point]
    dataset['X_s'] = dataset['X_s'][split_point:]

    temporal_X_st = dataset['X_st'][:split_point]
    dataset['X_st'] = dataset['X_st'][split_point:]

    temporal_Y = dataset['Y'][:split_point]
    dataset['Y'] = dataset['Y'][split_point:]

    temporal_X_s1 = dataset['X_s1'][:split_point]
    dataset['X_s1'] = dataset['X_s1'][split_point:]

    ### extract data from end of temporaly sorted dataset ###
    temporal_X_t_ord_1D = np.concatenate(
        (
            temporal_X_t_ord_1D,
            dataset['X_t_ord_1D'][-split_point:]
        )
    )
    dataset['X_t_ord_1D'] = dataset['X_t_ord_1D'][:-split_point]
    
    temporal_X_t = np.concatenate(
        (
            temporal_X_t, 
            dataset['X_t'][-split_point:]
        )
    )
    dataset['X_t'] = dataset['X_t'][:-split_point]

    temporal_X_s = np.concatenate(
        (
            temporal_X_s, 
            dataset['X_s'][-split_point:]
        )
    )
    dataset['X_s'] = dataset['X_s'][:-split_point]

    temporal_X_st = np.concatenate(
        (
            temporal_X_st, 
            dataset['X_st'][-split_point:]
        )
    )
    dataset['X_st'] = dataset['X_st'][:-split_point]

    temporal_Y = np.concatenate(
        (
            temporal_Y, 
            dataset['Y'][-split_point:]
        )
    )
    dataset['Y'] = dataset['Y'][:-split_point]

    temporal_X_s1 = np.concatenate(
        (
            temporal_X_s1, 
            dataset['X_s1'][-split_point:]
        )
    )
    dataset['X_s1'] = dataset['X_s1'][:-split_point]

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

    spatial_X_t_ord_1D = dataset['X_t_ord_1D']
    dataset['X_t_ord_1D'] = 0
    
    spatial_X_t = dataset['X_t']
    dataset['X_t'] = 0

    spatial_X_s = dataset['X_s']
    dataset['X_s'] = 0

    spatial_X_st = dataset['X_st']
    dataset['X_st'] = 0

    spatial_Y = dataset['Y']
    dataset['Y'] = 0

    spatial_X_s1 = dataset['X_s1']
    dataset['X_s1'] = 0

    ###
    # Extract temporal and spatio-temporal test sets ###
    ###

    ### create the filtering array ###
    boolean_filter_array = np.zeros((len(temporal_X_s),), dtype=bool)

    for building_id in test_building_samples:
        boolean_filter_array = boolean_filter_array | (
            temporal_X_s[:, 0] == building_id
        )

    inverted_boolean_filter_array = np.invert(boolean_filter_array)

    ### Spatio-temporal ###
    spatemp_X_t_ord_1D = temporal_X_t_ord_1D[boolean_filter_array]
    spatemp_X_t = temporal_X_t[boolean_filter_array]
    spatemp_X_s = temporal_X_s[boolean_filter_array]
    spatemp_X_st = temporal_X_st[boolean_filter_array]
    spatemp_Y = temporal_Y[boolean_filter_array]
    spatemp_X_s1 = temporal_X_s1[boolean_filter_array]

    spatemp_test_data = {
        'X_t_ord_1D': spatemp_X_t_ord_1D,
        'X_t': spatemp_X_t, 
        'X_s': spatemp_X_s, 
        'X_s1': spatemp_X_s1, 
        'X_st': spatemp_X_st, 
        'Y': spatemp_Y,
        'n_datapoints': len(spatemp_X_t)
    }
    (
        spatemp_X_t_ord_1D,
        spatemp_X_t, 
        spatemp_X_s, 
        spatemp_X_s1, 
        spatemp_X_st, 
        spatemp_Y
    ) = 0, 0, 0, 0, 0, 0

    ### Temporal ###
    temporal_X_t_ord_1D = temporal_X_t_ord_1D[inverted_boolean_filter_array]
    temporal_X_t = temporal_X_t[inverted_boolean_filter_array]
    temporal_X_s = temporal_X_s[inverted_boolean_filter_array]
    temporal_X_st = temporal_X_st[inverted_boolean_filter_array]
    temporal_Y = temporal_Y[inverted_boolean_filter_array]
    temporal_X_s1 = temporal_X_s1[inverted_boolean_filter_array]

    temporal_test_data = {
        'X_t_ord_1D': temporal_X_t_ord_1D,
        'X_t': temporal_X_t, 
        'X_s': temporal_X_s, 
        'X_s1': temporal_X_s1, 
        'X_st': temporal_X_st, 
        'Y': temporal_Y,
        'n_datapoints': len(temporal_X_t)
    }
    (
        temporal_X_t_ord_1D,
        temporal_X_t, 
        temporal_X_s, 
        temporal_X_s1, 
        temporal_X_st, 
        temporal_Y 
    ) = 0, 0, 0, 0, 0, 0


    ###
    # Extract spatial test set ###
    ###

    ### create the filtering array ###
    boolean_filter_array = np.zeros((len(spatial_X_s),), dtype=bool)

    for building_id in test_building_samples:
        boolean_filter_array = (
            boolean_filter_array | (spatial_X_s[:, 0] == building_id)
        )

    inverted_boolean_filter_array = np.invert(boolean_filter_array)

    ### Train-validation split ###
    train_val_X_t_ord_1D = spatial_X_t_ord_1D[inverted_boolean_filter_array]
    train_val_X_t = spatial_X_t[inverted_boolean_filter_array]
    train_val_X_s = spatial_X_s[inverted_boolean_filter_array]
    train_val_X_st = spatial_X_st[inverted_boolean_filter_array]
    train_val_Y = spatial_Y[inverted_boolean_filter_array]
    train_val_X_s1 = spatial_X_s1[inverted_boolean_filter_array]

    ### Spatial ###
    spatial_X_t_ord_1D = spatial_X_t_ord_1D[boolean_filter_array]
    spatial_X_t = spatial_X_t[boolean_filter_array]
    spatial_X_s = spatial_X_s[boolean_filter_array]
    spatial_X_st = spatial_X_st[boolean_filter_array]
    spatial_Y = spatial_Y[boolean_filter_array]
    spatial_X_s1 = spatial_X_s1[boolean_filter_array]

    spatial_test_data = {
        'X_t_ord_1D': spatial_X_t_ord_1D,
        'X_t': spatial_X_t, 
        'X_s': spatial_X_s, 
        'X_s1': spatial_X_s1, 
        'X_st': spatial_X_st, 
        'Y': spatial_Y,
        'n_datapoints': len(spatial_X_t)
    }
    (
        spatial_X_t_ord_1D,
        spatial_X_t, 
        spatial_X_s, 
        spatial_X_s1, 
        spatial_X_st, 
        spatial_Y 
    ) = 0, 0, 0, 0, 0, 0



    train_val_data = {
        'X_t_ord_1D': train_val_X_t_ord_1D,
        'X_t': train_val_X_t, 
        'X_s': train_val_X_s, 
        'X_s1': train_val_X_s1, 
        'X_st': train_val_X_st, 
        'Y': train_val_Y,
        'n_datapoints': len(train_val_X_t)
    }
    (
        train_val_X_t_ord_1D,
        train_val_X_t, 
        train_val_X_s, 
        train_val_X_s1, 
        train_val_X_st, 
        train_val_Y
    ) = 0, 0, 0, 0, 0, 0


    def f_randomize(dataset):
    
        """
        """
        # create random array
        random_array = np.arange(len(dataset['X_t']))

        # shuffle random array
        np.random.shuffle(random_array)

        dataset['X_t_ord_1D'] = dataset['X_t_ord_1D'][random_array]
        dataset['X_t'] = dataset['X_t'][random_array]
        dataset['X_s'] = dataset['X_s'][random_array]
        dataset['X_s1'] = dataset['X_s1'][random_array]
        dataset['X_st'] = dataset['X_st'][random_array]
        dataset['Y'] = dataset['Y'][random_array]

        return dataset
        
    
    train_val_data = f_randomize(train_val_data)
    spatial_test_data = f_randomize(spatial_test_data)
    temporal_test_data = f_randomize(temporal_test_data)
    spatemp_test_data = f_randomize(spatemp_test_data)
     
    if not silent:

        n_test_datapoints = (
            spatial_test_data['n_datapoints']
            + temporal_test_data['n_datapoints']
            + spatemp_test_data['n_datapoints']
        )
        n_total_datapoints = (
            train_val_data['n_datapoints']
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
                train_val_data['n_datapoints'],
                train_val_data['n_datapoints'] / n_total_datapoints,
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
                spatial_test_data['n_datapoints'],
                spatial_test_data['n_datapoints'] / n_test_datapoints,
            )
        )
        print(
            'Temporal testing data:        {} ({:.0%})'.format(
                temporal_test_data['n_datapoints'],
                temporal_test_data['n_datapoints'] / n_test_datapoints,
            )
        )
        print(
            'Spatio-temporal testing data: {} ({:.0%})'.format(
                spatemp_test_data['n_datapoints'],
                spatemp_test_data['n_datapoints'] / n_test_datapoints,
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
    silent=True
):

    """ Converts the population of each feature into a standard score using mean 
    and std deviations. For X_st, the past time steps of each meteorological 
    condition are transformed separately. For X_s1, the histogram or average 
    values of each channel are transformed separately.
    """

    if raw_data['standardization']:

        if not silent:

            # tell us what we do
            print('Standardizing data')

        # get StandardScaler from the sklearn preprocessing package
        standard_scaler = preprocessing.StandardScaler()

        # standardize X_t in the case that it is not OHE
        if raw_data['time_encoding'] != 'OHE':

            standard_scaler.fit(reference_data['X_t'])
            dataset['X_t'] = standard_scaler.transform(dataset['X_t'])

        # standardize X_st
        for i in range(len(raw_data['meteo_types'])):

            standard_scaler.fit(reference_data['X_st'][:, :, i])
            dataset['X_st'][:, :, i] = standard_scaler.transform(
                dataset['X_st'][:, :, i]
            )

        # standardize X_s1
        for channel in range(raw_data['n_channels']):

            standard_scaler.fit(reference_data['X_s1'][:, :, channel])
            dataset['X_s1'][:, :, channel] = standard_scaler.transform(
                dataset['X_s1'][:, :, channel]
            )

    return dataset
