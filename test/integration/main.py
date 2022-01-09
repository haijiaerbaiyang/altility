import sys
sys.path.append('/altility/src')
import altility.adl_model as adl_model
import altility.datasets.load_forecasting as load_forecasting

# import and prepare load forecasting data
datasets = load_forecasting.prep_load_forecasting_data(
    silent=False,
    plot=True
)


### Get features and labels for available data
y = datasets[0]['Y']
x_t = datasets[0]['X_t']
x_s = datasets[0]['X_s1']
x_st = datasets[0]['X_st']

# show us how they look like
print(x_t.shape)
print(x_s.shape)
print(x_st.shape)
print(y.shape)

# create a class instance
ADL_model = adl_model.ADL_model('Undisputed f_nn')

# initialize model by creating and training it
ADL_model.initialize(
    y,
    x_t,
    x_s,
    x_st,
    silent=True,
    plot=True
)

# show us if we created all models
for model_name, model in ADL_model.models.items():
    print(model_name)


### Get features and labels for candidate data from spatio-temporal test set
x_t_cand = datasets[3]['X_t']
x_s_cand = datasets[3]['X_s1']
x_st_cand = datasets[3]['X_st']
y_cand = datasets[3]['Y']

# collect candidate data
ADL_model.collect(
    x_t_cand,
    x_s_cand,
    x_st_cand,
    silent=True,
    plot=False
)

print(len(ADL_model.batch_index_list))
print(len(ADL_model.inf_score_list))

# show us that we ran successfully
print('the file test/integration/main.py is successfully executed through docker')
