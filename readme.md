Altility stands for 'actively learning utility'. It is a package that provides 
an active deep learning model for enhancing spatio-temporal predicitions. Originally,
we developed this to help electric utilities in the process of placing new smart 
meters in space and collecting their data at different time. 


### Installation:
```
pip install altility
```


### Docker:
For using altility within an Ubuntu docker container
```
docker run -it aryandoustarsam/altility
```

For using altility with Jupyter notebook inside a docker container
```
docker run -it -p 3333:1111 -v ~/path_to_data/data:/data aryandoustarsam/altility:jupyter
[inside running container]: jupyter notebook --ip 0.0.0.0 --port 1111 --no-browser --allow-root
[in local machine browser]: localhost:3333 
[in local machine browser, type token shown in terminal]
```


### Usage guide:
At the core of altility stands the class **altility.ADL_model**. It bundles properties
and methods of the active deep learning (ADL) model that we want to train. Bellow
is a list of all parameters, attributes, methods and generated results.

<table>

  <tr>
    <th scope='row' colspan='2'> Parameters </th>
  </tr>
  
  <tr> 
    <td>
      <b>name (='adl_model')</b>: <br />  string
    </td>
    <td>
      The name of active deep learning (ADL) model
    </td>
  </tr> 
    
</table>

<table>

  <tr>
    <th scope='row' colspan='2'> Methods </th>
  </tr>
  
  <tr> 
    <td>
      <b>initialize(y, x_t=None, x_s=None, x_st=None, **kwargs):</b>
    </td>
    <td>
      Initializes prediction model.
    </td>
  </tr> 
  
  <tr> 
    <td>
      <b>collect(x_t_cand=None, x_s_cand=None, x_st_cand=None, **kwargs):</b>
    </td>
    <td>
      Performs active learning.
    </td>
  </tr> 
    
</table>

<table>

  <tr>
    <th scope='row' colspan='2'> Results </th>
  </tr>
  
  <tr> 
    <td>
      <b>batch_index_list</b>: <br /> list of integers
    </td>
    <td>
      List of indices for most informative data points suggested to collect.
    </td>
  </tr>
  
  <tr> 
    <td>
      <b>inf_score_list</b>: <br /> list of floats
    </td>
    <td>
      List of information scores for most informative data points suggested to 
      collect.
    </td>
  </tr>
    
</table>


### Datasets:
The package can be tested with a public dataset for making spatio-temporal predictions
of electric load that we provide in our Github repository. To prepare the data
for usage with altility, use the **prep_load_forecasting_data()** function provided
in load_forecasting.py with the following parameter and return values:

<table>

  <tr>
    <th scope='row' colspan='2'> Parameters </th>
  </tr>
  
  <tr> 
    <td>
      <b>path_to_data (='data/public/electric load forecasting/')</b>: <br />  string
    </td>
    <td>
      The path to where data is stored. This is 'data/public/electric load forecasting/'
      in our original repository.
    </td>
  </tr>
  
  <tr>
    <td>
      <b>dataset_name (='profiles_100')</b>: <br /> string
    <td>
      Choose between 'profiles_100' and 'profiles_400'. These are two distinct
      datasets containing load profiles from either 100 or 400 industrial, commercial,
      and residential buildings of different sizes, shapes, consumption and occupancy
      patterns in Switzerland.
    </td>
    </td>
  </tr>
  
  <tr>
    <td>
      <b>label_type (='feature_scaled')</b>: <br /> string
    <td>
      Decide which labels to consider. Choose from 'random_scaled' and 'feature_scaled'.
    </td>
    </td>
  </tr>
  
  <tr>
    <td>
      <b>spatial_features (='histogram')</b>: <br /> string
    <td>
      Decide how to treat aerial imagery. Choose one from 'average' and 'histogram'.
    </td>
    </td>
  </tr>
  
  <tr>
    <td>
      <b>meteo_types </b>: <br /> list
    <td>
      Decide which meteo data types to consider. Choose from 'air_density', 
      'cloud_cover', 'precipitation', 'radiation_surface', 'radiation_toa',
      'snow_mass', 'snowfall', 'temperature' and 'wind_speed'. The default is a 
      list of all meteorological conditions.
    </td>
    </td>
  </tr>
  
  <tr>
    <td>
      <b>timestamp_data </b>: <br /> list
    <td>
      Decide which time stamp information to consider. Choose from: '15min',
      'hour', 'day', 'month' and 'year'.
    </td>
    </td>
  </tr>
  
  <tr>
    <td>
      <b>time_encoding (='ORD')</b>: <br /> string
    <td>
      Decide how to encode time stamp data. Choose one of 'ORD', 'ORD-1D' or 'OHE'
    </td>
    </td>
  </tr>
  
  <tr>
    <td>
      <b>histo_bins (=100)</b>: <br /> int
    <td>
      Set the number of histogram bins that you want to use. Applied if parameter
      spatial_features = 'histogram'.
    </td>
    </td>
  </tr>
  
  <tr>
    <td>
      <b>grey_scale (=False)</b>: <br /> bool
    <td>
      Decide whether you want to consider underlying RGB images in grey-scale.
    </td>
    </td>
  </tr>
  
  <tr>
    <td>
      <b>profiles_per_year (=1)</b>: <br /> float
    <td>
      Decide how many building-year profiles you want to consider for each year. 
      Choose a share between 0 and 1. A value of 1 corresponds to about 100 profiles 
      for the profiles_100 and 400 profiles for the profiles_400 dataset.
    </td>
    </td>
  </tr>
  
  <tr>
    <td>
      <b>points_per_profile (=0.003)</b>: <br /> float
    <td>
      Decide how many data points per building-year profile you want to consider. 
      Choose a share between 0 and 1. A value of 0.01 corresponds to approximately 
      350 points per profile.
    </td>
    </td>
  </tr>
  
  <tr>
    <td>
      <b>history_window_meteo (=24)</b>: <br /> int
    <td>
      Choose past time window for the meteo data. Resolution is hourly.
    </td>
    </td>
  </tr>
  
  <tr>
    <td>
      <b>prediction_window (=96)</b>: <br /> int
    <td>
      Decide how many time steps to predict consumption into the future. Resolution 
      is 15 min. A values of 96 corresponds to 24h.
    </td>
    </td>
  </tr>
  
  <tr>
    <td>
      <b>train_split (=0.3)</b>: <br /> float
    <td>
      Decides on the splitting ratio between training and validation datasets.
    </td>
    </td>
  </tr>
  
  <tr>
    <td>
      <b>test_split (=0.7)</b>: <br /> float
    <td>
      Decides how many buildings and how much of the time period to separate for
      testing.
    </td>
    </td>
  </tr>
  
  <tr>
    <td>
      <b>normalization (=True)</b>: <br /> bool
    <td>
      Decide whether or not to normalize features.
    </td>
    </td>
  </tr>
  
  <tr>
    <td>
      <b>standardization (=True)</b>: <br /> bool
    <td>
      Decide whether to standardize features to zero mean and unit variance.
    </td>
    </td>
  </tr>
  
  <tr>
    <td>
      <b>silent (=True)</b>: <br /> bool
    <td>
      Decide whether or not to print out progress of data processing.
    </td>
    </td>
  </tr>
  
  <tr>
    <td>
      <b>plot (=False)</b>: <br /> bool
    <td>
      Decide whether or not to visualize examples of processed data.
    </td>
    </td>
  </tr>
    
</table>



### Examples:

```
import altility.adl_model as adl_model
import altility.datasets.load_forecasting as load_forecasting

### Import and prepare load forecasting data
datasets = load_forecasting.prep_load_forecasting_data(
    silent=False,
    plot=True
)

### Get features and labels for available data
y = datasets[0]['Y']
x_t = datasets[0]['X_t']
x_s = datasets[0]['X_s1']
x_st = datasets[0]['X_st']

### Create a class instance
ADL_model = adl_model.ADL_model('ADL model f_nn')

### Initialize model by creating and training it
ADL_model.initialize(
    y,
    x_t,
    x_s,
    x_st,
    silent=False,
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
```
