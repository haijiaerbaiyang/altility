import sys
sys.path.append('/altility/src')
import altility.adl_model as adl_model
import altility.datasets.load_forecasting as load_forecasting
import altility.datasets.travel_forecasting as travel_forecasting


### Import and prepare travel forecasting data
datasets = travel_forecasting.prep_travel_forecasting_data(
    silent=False,
    plot=True
)

### Get features and labels for available data
n_points=1000
y = datasets[0]['y'][:n_points]
x_t = datasets[0]['x_t'][:n_points]
x_s = datasets[0]['x_s'][:n_points]


### Get features and labels for candidate data from spatio-temporal test set
y_cand = datasets[1]['y'][:n_points]
x_t_cand = datasets[1]['x_t'][:n_points]
x_s_cand = datasets[1]['x_s'][:n_points]


### Show us how they look like
print(x_t.shape)
print(x_s.shape)
print(y.shape)

### Show us how they look like
print(x_t_cand.shape)
print(x_s_cand.shape)
print(y_cand.shape)


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
    silent=True,
    plot=False
)


### Show us how many data points are chosen
print(len(ADL_model.batch_index_list))
print(len(ADL_model.inf_score_list))

print(ADL_model.batch_index_list)

"""

###
# Full example for forecasting electric consumption of single buildings ###
###

### Import and prepare load forecasting data
datasets = load_forecasting.prep_load_forecasting_data(
    silent=False,
    plot=True
)


### Get features and labels for available data
y = datasets[0]['y']
x_t = datasets[0]['x_t']
x_s = datasets[0]['x_s1']
x_st = datasets[0]['x_st']


### Get features and labels for candidate data from spatio-temporal test set
y_cand = datasets[3]['y']
x_t_cand = datasets[3]['x_t']
x_s_cand = datasets[3]['x_s1']
x_st_cand = datasets[3]['x_st']


### Show us how they look like
print(x_t.shape)
print(x_s.shape)
print(x_st.shape)
print(y.shape)


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


### Show us if we created all models
for model_name, model in ADL_model.models.items():
    print(model_name)


### Collect candidate data
ADL_model.collect(
    x_t_cand,
    x_s_cand,
    x_st_cand,
    silent=True,
    plot=False
)


### Show us how many data points are chosen
print(len(ADL_model.batch_index_list))
print(len(ADL_model.inf_score_list))

# show us that we ran successfully
print('the file test/integration/main.py is successfully executed through docker')

"""
