import sys
sys.path.append('/altility/src')

import numpy as np

import altility.adl_model as adl_model
import altility.datasets.load_forecasting as load_forecasting
import altility.datasets.travel_forecasting as travel_forecasting

test_list = [
    'travel_time',
    'electric_load
]


###
# Full example for forecasting electric consumption of single buildings ###
###

if 'electric_load' in test_list:

    ### Import and prepare load forecasting data
    datasets = load_forecasting.prep_load_forecasting_data(
        silent=False,
        plot=True
    )


    ### Get features and labels for available data
    y = datasets['avail_data']['y']
    x_t = datasets['avail_data']['x_t']
    x_s = datasets['avail_data']['x_s']
    x_st = datasets['avail_data']['x_st']


    ### Get features and labels for candidate data from spatio-temporal test set
    y_cand = datasets['cand_data']['y']
    x_t_cand = datasets['cand_data']['x_t']
    x_s_cand = datasets['cand_data']['x_s']
    x_st_cand = datasets['cand_data']['x_st']


    ### Create a class instance
    ADL_model = adl_model.ADL_model('Electrific f_nn')


    ### Initialize model by creating and training it
    ADL_model.initialize(
        y,
        x_t,
        x_s,
        x_st,
        silent=True,
        plot=True
    )


    ### Collect candidate data
    ADL_model.collect(
        x_t_cand,
        x_s_cand,
        x_st_cand,
        silent=True,
        plot=False
    )


    ### Create one array for picked and one for unpicked data to be predicted
    picked_array = np.zeros([len(y_cand),], dtype=bool)
    picked_array[ADL_model.batch_index_list] = True
    pred_array = np.invert(picked_array)


    ### Extract selected data from candidate data pool for training
    y_picked = y_cand[picked_array]
    x_t_picked = x_t_cand[picked_array]
    x_s_picked = x_s_cand[picked_array]
    x_st_picked = x_st_cand[picked_array]


    ### Train model with picked data
    ADL_model.train(
        y_picked,
        x_t_picked,
        x_s_picked,
        x_st_picked,
        silent=False,
        plot=True
    )


    ### Extract not selected data from candidate data pool for testing/predicting
    y_pred = y_cand[pred_array]
    x_t_pred = x_t_cand[pred_array]
    x_s_pred = x_s_cand[pred_array]
    x_st_pred = x_st_cand[pred_array] 


    ### Predict on remaining data
    ADL_model.predict(
        y_pred,
        x_t_pred,
        x_s_pred,
        x_st_pred,
        silent=False,
        plot=True
    )


###
# Full example for forecasting travel times between single city zones ###
###

if 'travel_time' in test_list:

    ### Import and prepare travel forecasting data
    datasets = travel_forecasting.prep_travel_forecasting_data(
        silent=False,
        plot=True
    )


    ### Get features and labels for available data
    n_points=1000
    y = datasets['avail_data']['y'][:n_points]
    x_t = datasets['avail_data']['x_t'][:n_points]
    x_s = datasets['avail_data']['x_s'][:n_points]


    ### Get features and labels for candidate data from spatio-temporal test set
    y_cand = datasets['cand_data']['y'][:n_points]
    x_t_cand = datasets['cand_data']['x_t'][:n_points]
    x_s_cand = datasets['cand_data']['x_s'][:n_points]


    ### Create a class instance
    ADL_model = adl_model.ADL_model('Spacetimetravelic f_nn')


    ### Initialize model by creating and training it
    ADL_model.initialize(
        y,
        x_t=x_t,
        x_s=x_s,
        silent=True,
        plot=True
    )


    ### Show us if we created all models
    for model_name, model in ADL_model.models.items():
        print(model_name)


    ### Collect candidate data
    ADL_model.collect(
        x_t_cand,
        x_s_cand,
        silent=False,
        plot=True
    )


    ### Create one array for picked and one for unpicked data to be predicted
    picked_array = np.zeros([len(y_cand),], dtype=bool)
    picked_array[ADL_model.batch_index_list] = True
    pred_array = np.invert(picked_array)


    ### Extract selected data from candidate data pool for training
    y_picked = y_cand[picked_array]
    x_t_picked = x_t_cand[picked_array]
    x_s_picked = x_s_cand[picked_array]


    ### Train model with picked data
    ADL_model.train(
        y_picked,
        x_t_picked,
        x_s_picked,
        silent=False,
        plot=True
    )


    ### Extract not selected data from candidate data pool for testing/predicting
    y_pred = y_cand[pred_array]
    x_t_pred = x_t_cand[pred_array]
    x_s_pred = x_s_cand[pred_array] 


    ### Predict on remaining data
    ADL_model.predict(
        y_pred,
        x_t_pred,
        x_s_pred,
        silent=False,
        plot=True
    )
