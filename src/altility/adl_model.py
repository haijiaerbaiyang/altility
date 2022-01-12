import sys
import os
import math

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import laplacian_kernel 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OrdinalEncoder


class ADL_model:

    """ Class that boundles all parameters, methods and results for performing
    active learning.
    """
    
    def __init__(
        self,
        name='adl_model',
        path_to_results='results/'
    ):
    
        ### Parameters
        self.name = name
        if path_to_results.endswith('/'):
            self.path_to_results = path_to_results
        else:
            self.path_to_results = path_to_results + '/'
        
        if not os.path.exists(path_to_results):
            os.mkdir(path_to_results)
    
    ### Methods 
        
    def initialize(
        self,
        y,
        x_t=None,
        x_s=None,
        x_st=None,
        encoder_layers=1,
        network_layers=1,
        encoding_nodes_x_t=100,
        encoding_nodes_x_s=100,
        encoding_nodes_x_st=100,
        encoding_nodes_joint=100,
        nodes_per_layer_dense=1000,
        filters_per_layer_cnn=16,
        states_per_layer_rnn=200,
        activation_encoding='relu',
        activation_dense='relu',
        activation_cnn='relu',
        activation_rnn='tanh',
        layer_type_x_st='CNN',
        initialization_method='glorot_normal',
        initialization_method_rnn='orthogonal',
        regularizer='l1_l2',
        batch_normalization=False,
        train_split=0.7,
        split_intervals=0.05,
        random_seed=None,
        epochs=30,
        patience=10,
        batch_size=16,
        monitor='val_loss',
        silent=True,
        plot=False
    ):
    
        """ Initializes parameters passed to model as class attributes and calls
        methods for creating modular deep learning model (embedding network),
        splitting data into training and validation sets, and training embedding
        network prediction model.
        """
        
        ### Set all parameters as attributes of class:
        self.y = y
        self.x_t = x_t
        self.x_s = x_s
        self.x_st = x_st
        self.encoder_layers = encoder_layers
        self.network_layers = network_layers
        self.encoding_nodes_x_t = encoding_nodes_x_t
        self.encoding_nodes_x_s = encoding_nodes_x_s
        self.encoding_nodes_x_st = encoding_nodes_x_st
        self.encoding_nodes_joint = encoding_nodes_joint
        self.nodes_per_layer_dense = nodes_per_layer_dense
        self.filters_per_layer_cnn = filters_per_layer_cnn
        self.states_per_layer_rnn = states_per_layer_rnn
        self.activation_encoding = activation_encoding
        self.activation_dense = activation_dense
        self.activation_cnn = activation_cnn
        self.activation_rnn = activation_rnn
        self.layer_type_x_st = layer_type_x_st
        self.initialization_method = initialization_method
        self.initialization_method_rnn = initialization_method_rnn
        self.regularizer = regularizer
        self.batch_normalization = batch_normalization
        self.train_split = train_split
        self.split_intervals = split_intervals
        self.random_seed = random_seed
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.monitor = monitor
        self.silent = silent
        self.plot = plot
        
        ### Call methods
        self.create_prediction_model()
        self.split_train_val()
        self.train_model(fig_name='train_vs_val_initial')
        
    def train(
        self,
        y_picked,
        x_t_picked=None,
        x_s_picked=None,
        x_st_picked=None,
        silent=True,
        plot=False,
    ):    
          
        """ Trains embedding network prediction model with passed data. Note:
        It does not split the passed data into training and validation sets, but
        rather uses the validation data created when calling ADL_model.initialize().
        """
        
        self.y_train = y_picked
        self.x_t_train = x_t_picked
        self.x_s_train = x_s_picked
        self.x_st_train = x_st_picked
        self.silent = silent
        self.plot = plot
        
        self.train_model(fig_name='train_vs_val_adl')
        
        
    def create_prediction_model(self):
    
        """ Creates the deep learning models that compound our modular computational
        graph (embedding network).
        """
        
        ### Set initialization methods
        
        if self.initialization_method=='glorot_normal':
            self.initialization = tf.keras.initializers.GlorotNormal(
                seed=self.random_seed
            )
            
        if self.initialization_method_rnn=='orthogonal':
            self.initialization_rnn = tf.keras.initializers.Orthogonal(
                gain=1.0, 
                seed=self.random_seed
            )
        
        if not self.silent:
            # tell us what we do
            print('Building deep learning model')
            
        # take example label
        y_example = self.y[0]
        
        # check if features provided and take examples, otherwise, throw out error
        if self.x_t is not None:
            x_t_example = self.x_t[0]
        if self.x_s is not None:
            x_s_example = self.x_s[0]
        if self.x_st is not None:
            x_st_example = self.x_st[0]
        if self.x_t is None and self.x_s is None and self.x_st is None:
            print(
                'At least one feature type x_t, x_s, or x_st must be provided'
                + 'to train prediction model'
            )
            sys.exit()
        


        ### Create the input layers ###
        
        if self.x_t is not None:
            x_t_input = tf.keras.Input(shape=x_t_example.shape, name='X_t')
        if self.x_s is not None:
            x_s_input = tf.keras.Input(shape=x_s_example.shape, name='X_s')
        if self.x_st is not None:
            x_st_input = tf.keras.Input(shape=x_st_example.shape, name='X_st') 
        
        
        ### Create the hidden layers ###
        
        ### Encode x_t ###
        if self.x_t is not None:
        
            if self.encoder_layers == 0:
                x_t = tf.keras.layers.Flatten()(x_t_input)

            else:
                x_t = tf.keras.layers.Dense(
                    self.nodes_per_layer_dense,
                    activation=self.activation_dense,
                    kernel_initializer=self.initialization,
                    kernel_regularizer=self.regularizer,
                )(x_t_input)
                
                if self.batch_normalization:
                    x_t = tf.keras.layers.BatchNormalization()(x_t)
                    
                for i in range(self.encoder_layers - 1):
                    x_t = tf.keras.layers.Dense(
                        self.nodes_per_layer_dense,
                        activation=self.activation_dense,
                        kernel_initializer=self.initialization,
                        kernel_regularizer=self.regularizer,
                    )(x_t)
                    
                    if self.batch_normalization:
                        x_t = tf.keras.layers.BatchNormalization()(x_t)
                        
                x_t = tf.keras.layers.Flatten()(x_t)

            X_t = tf.keras.layers.Dense(
                self.encoding_nodes_x_t,
                activation=self.activation_encoding,
                kernel_initializer=self.initialization,
                kernel_regularizer=self.regularizer,
            )(x_t)
            
            if self.batch_normalization:
                x_t = tf.keras.layers.BatchNormalization()(x_t)
        
            
        ### Encode x_s ###
        
        if self.x_s is not None:
        
            if self.encoder_layers == 0:
                x_s = tf.keras.layers.Flatten()(x_s_input)

            else:
                x_s = tf.keras.layers.Dense(
                    self.nodes_per_layer_dense,
                    activation=self.activation_dense,
                    kernel_initializer=self.initialization,
                    kernel_regularizer=self.regularizer,
                )(x_s_input)
                
                if self.batch_normalization:
                    x_s = tf.keras.layers.BatchNormalization()(X_s1)
                    
                for i in range(self.encoder_layers - 1):
                    x_s = tf.keras.layers.Dense(
                        self.nodes_per_layer_dense,
                        activation=self.activation_dense,
                        kernel_initializer=self.initialization,
                        kernel_regularizer=self.regularizer,
                    )(x_s)
                    
                    if self.batch_normalization:
                        x_s = tf.keras.layers.BatchNormalization()(x_s)
                        
                x_s = tf.keras.layers.Flatten()(x_s)

            x_s = tf.keras.layers.Dense(
                self.encoding_nodes_x_s,
                activation=self.activation_dense,
                kernel_initializer=self.initialization,
                kernel_regularizer=self.regularizer,
            )(x_s)
            
            if self.batch_normalization:
                x_s = tf.keras.layers.BatchNormalization()(x_s)
                
                
        ### Encode x_st ###
        
        if self.x_st is not None:
    
            if self.encoder_layers == 0:
                x_st = tf.keras.layers.Flatten()(x_st_input)

            else:
                if self.layer_type_x_st == 'ANN':
                    x_st = tf.keras.layers.Dense(
                        self.nodes_per_layer_dense,
                        activation=self.activation_dense,
                        kernel_initializer=self.initialization,
                        kernel_regularizer=self.regularizer,
                    )(x_st_input)
                    
                    if self.batch_normalization:
                        X_st = tf.keras.layers.BatchNormalization()(X_st)
                        
                    for i in range(self.encoder_layers - 1):
                        x_st = tf.keras.layers.Dense(
                            self.nodes_per_layer_dense,
                            activation=self.activation_dense,
                            kernel_initializer=self.initialization,
                            kernel_regularizer=self.regularizer,
                        )(x_st)
                        
                        if self.batch_normalization:
                            X_st = tf.keras.layers.BatchNormalization()(X_st)

                elif self.layer_type_x_st == 'CNN':
                    x_st = tf.keras.layers.Conv1D(
                        self.filters_per_layer_cnn,
                        2,
                        activation=self.activation_cnn,
                        kernel_initializer=self.initialization,
                        kernel_regularizer=self.regularizer,
                    )(x_st_input)
                    
                    if self.batch_normalization:
                        x_st = tf.keras.layers.BatchNormalization()(x_st)
                        
                    for i in range(self.encoder_layers - 1):
                        x_st = tf.keras.layers.Conv1D(
                            self.filters_per_layer_cnn,
                            2,
                            activation=self.activation_cnn,
                            kernel_initializer=self.initialization,
                            kernel_regularizer=self.regularizer,
                        )(x_st)
                        
                        if self.batch_normalization:
                            x_st = tf.keras.layers.BatchNormalization()(x_st)

                elif self.layer_type_x_st == 'LSTM':
                    if self.encoder_layers == 1:
                        x_st = tf.keras.layers.LSTM(
                            self.states_per_layer_rnn,
                            activation=self.activation_rnn,
                            kernel_initializer=self.initialization,
                            kernel_regularizer=self.regularizer,
                        )(x_st_input)
                        
                        if self.batch_normalization:
                            x_st = tf.keras.layers.BatchNormalization()(x_st)
                            
                    else:
                        x_st = tf.keras.layers.LSTM(
                            self.states_per_layer_rnn,
                            return_sequences=True,
                            activation=self.activation_rnn,
                            kernel_initializer=self.initialization,
                            kernel_regularizer=self.regularizer,
                        )(x_st_input)
                        
                        if self.batch_normalization:
                            x_st = tf.keras.layers.BatchNormalization()(x_st)
                            
                        for i in range(self.encoder_layers - 2):
                            x_st = tf.keras.layers.LSTM(
                                self.states_per_layer_rnn,
                                return_sequences=True,
                                activation=self.activation_rnn,
                                kernel_initializer=self.initialization,
                                kernel_regularizer=self.regularizer,
                            )(x_st)
                            
                            if self.batch_normalization:
                                X_st = tf.keras.layers.BatchNormalization()(X_st)
                                
                        x_st = tf.keras.layers.LSTM(
                            self.states_per_layer_rnn,
                            activation=self.activation_rnn,
                            kernel_initializer=self.initialization,
                            kernel_regularizer=self.regularizer,
                        )(x_st)
                        
                        if self.batch_normalization:
                            x_st = tf.keras.layers.BatchNormalization()(x_st)

                x_st = tf.keras.layers.Flatten()(x_st)

            x_st = tf.keras.layers.Dense(
                self.encoding_nodes_x_st,
                activation=self.activation_encoding,
                kernel_initializer=self.initialization,
                kernel_regularizer=self.regularizer,
            )(x_st)
            
            if self.batch_normalization:
                x_st = tf.keras.layers.BatchNormalization()(x_st)
            
            
        ### Create and join the encoders ###

        # create empty lists for joing layers and inputs of total prediction model
        input_list = []
        joining_list = []

        ### create x_t encoder ###
        if self.x_t is not None:
            x_t_encoder = tf.keras.Model(inputs=x_t_input, outputs=x_t)
            input_list.append(x_t_input)
            joining_list.append(x_t)

        ### create x_s encoder ###
        if self.x_s is not None:
            x_s_encoder = tf.keras.Model(inputs=x_s_input, outputs=x_s)
            input_list.append(x_s_input)
            joining_list.append(x_s)

        ### create x_st encoder ###
        if self.x_st is not None:
            x_st_encoder = tf.keras.Model(inputs=x_st_input, outputs=x_st)
            input_list.append(x_st_input)
            joining_list.append(x_st)
        
        
        ### create joint encoder ###
        
        if len(joining_list) > 1:
            joining_layer = tf.keras.layers.concatenate(joining_list)
        else:
            joining_layer = joining_list[0]
        
        for i in range(self.encoder_layers):
            joining_layer = tf.keras.layers.Dense(
                self.nodes_per_layer_dense,
                activation=self.activation_dense,
                kernel_initializer=self.initialization,
                kernel_regularizer=self.regularizer,
            )(joining_layer)
            
            if self.batch_normalization:
                joining_layer = tf.keras.layers.BatchNormalization()(joining_layer)

        joining_layer = tf.keras.layers.Dense(
            self.encoding_nodes_joint,
            activation=self.activation_encoding,
            kernel_initializer=self.initialization,
            kernel_regularizer=self.regularizer,
        )(joining_layer)
        
        if self.batch_normalization:
            joining_layer = tf.keras.layers.BatchNormalization()(joining_layer)
            
        x_joint_encoder = tf.keras.Model(inputs=input_list, outputs=joining_layer)
        
        
        ### Create total prediction model ###

        for i in range(self.network_layers):
            joining_layer = tf.keras.layers.Dense(
                self.nodes_per_layer_dense,
                activation=self.activation_dense,
                kernel_initializer=self.initialization,
                kernel_regularizer=self.regularizer,
            )(joining_layer)
            
            if self.batch_normalization:
                joining_layer = tf.keras.layers.BatchNormalization()(joining_layer)

        model_output = tf.keras.layers.Dense(
            len(y_example),
            activation='softplus',
            kernel_initializer=self.initialization,
        )(joining_layer)
  

        # create the tf model and define its inputs and outputs
        f_nn = tf.keras.Model(
            inputs=input_list, 
            outputs=model_output
        )

        # create class instance for encoding and prediction models
        models = {}
        if self.x_t is not None:
            models['x_t_encoder'] = x_t_encoder
        if self.x_s is not None:
            models['x_s_encoder'] = x_s_encoder
        if self.x_st is not None:
            models['x_st_encoder'] = x_st_encoder
        models['x_joint_encoder'] = x_joint_encoder
        models['f_nn'] = f_nn
        
        self.models = models
        
        # give us the summary of the total prediction model that we define
        if not self.silent:
            f_nn.summary()
            
            
            
            
    def split_train_val(self):
    
        """ Splits available data into training and validation sets.
        """     
        
        ###
        # Split remaining into training and validation datasets using intervals ###
        ###

        split_bins = math.ceil(len(self.y) * self.split_intervals)

        random_array = np.arange(len(self.y))
        random_array = np.array_split(random_array, split_bins)
        np.random.shuffle(random_array)
        random_array = np.concatenate(random_array).ravel()
        
        
        self.y = self.y[random_array]
        if self.x_t is not None:
            self.x_t = self.x_t[random_array]
        if self.x_s is not None:
            self.x_s = self.x_s[random_array]
        if self.x_st is not None:
            self.x_st = self.x_st[random_array]    

        # get splitting point for training validation split
        split_point = math.ceil(self.train_split * len(self.y))

        # split train and delete unused entries immediately
        self.y_train, self.y_val = np.split(self.y, [split_point])
        self.y = 0
        
        if self.x_t is not None:
            self.x_t_train, self.x_t_val = np.split(self.x_t, [split_point])
            self.x_t = 0

        if self.x_s is not None:
            self.x_s_train, self.x_s_val = np.split(self.x_s, [split_point])
            self.x_s = 0
        
        if self.x_st is not None:
            self.x_st_train, self.x_st_val = np.split(self.x_st, [split_point])
            self.x_st = 0
        
    def f_randomize(self, data):
    
        """ Randomizes data order for training and validation.
        """
        
        # randomize training data
        if data == 'train':
            
            # create random array
            random_array = np.arange(len(self.y_train))
            self.y_train = self.y_train[random_array]

            if self.x_t is not None:
                self.x_t_train = self.x_t_train[random_array]

            if self.x_s is not None:
                self.x_s_train = self.x_s_train[random_array]
                
            if self.x_st is not None:
                self.x_st_train = self.x_st_train[random_array]
                
        elif data == 'val':
        
            # create random array
            random_array = np.arange(len(self.y_val))
            self.y_val = self.y_val[random_array]

            if self.x_t is not None:
                self.x_t_val = self.x_t_val[random_array]

            if self.x_s is not None:
                self.x_s_val = self.x_s_val[random_array]
                
            if self.x_st is not None:
                self.x_st_val = self.x_st_val[random_array]
        
        
    def create_batched_data(self, data, i):

        """ Creates data batches for training and validation.
        """

        if data == 'train':
        
            data_y = self.y_train
            if self.x_t is not None:
                data_x_t = self.x_t_train
            if self.x_st is not None:
                data_x_st = self.x_st_train
            if self.x_s is not None:
                data_x_s = self.x_s_train
                
        elif data == 'val':
        
            data_y = self.y_val
            if self.x_t is not None:
                data_x_t = self.x_t_val
            if self.x_s is not None:
                data_x_s = self.x_s_val
            if self.x_st is not None:
                data_x_st = self.x_st_val

        # build batches of data
        for j in range(self.batch_size):

            # Get training data of currently iterated batch
            
            y = data_y[i + j]
            if self.x_t is not None:
                x_t = data_x_t[i + j]
            if self.x_s is not None:
                x_s = data_x_s[i + j]
            if self.x_st is not None:
                x_st = data_x_st[i + j]


            # Expand dimensions for batching
            y = np.expand_dims(y, axis=0)
            if self.x_t is not None:
                x_t = np.expand_dims(x_t, axis=0)
            if self.x_s is not None:
                x_s = np.expand_dims(x_s, axis=0)
            if self.x_st is not None:
                x_st = np.expand_dims(x_st, axis=0)

            # Create batches
            if j == 0:

                y_batched = y
                if self.x_t is not None:
                    x_t_batched = x_t
                if self.x_s is not None:
                    x_s_batched = x_s
                if self.x_st is not None:
                    x_st_batched = x_st

            else:
            
                y_batched = np.concatenate((y_batched, y), axis=0)
                if self.x_t is not None:
                    x_t_batched = np.concatenate((x_t_batched, x_t), axis=0)
                if self.x_s is not None:
                    x_s_batched = np.concatenate((x_s_batched, x_s), axis=0)
                if self.x_st is not None:
                    x_st_batched = np.concatenate((x_st_batched, x_st), axis=0)

        # Create model input list
        model_input_list = []
        if self.x_t is not None:
            model_input_list.append(x_t_batched)
        if self.x_s is not None:
            model_input_list.append(x_s_batched)
        if self.x_st is not None:
            model_input_list.append(x_st_batched)

        return model_input_list, y_batched
        
        
    def train_model(
        self, 
        fig_name='training'
    ):
        """ Functions for training and validating model.
        """
        
        if self.silent:
            verbose = 0
        else:
            verbose = 1
        
        # initialize loss functions and optimizer
        loss_object = tf.keras.losses.mean_squared_error
        optimizer = tf.keras.optimizers.RMSprop()
        mean_loss = tf.keras.metrics.Mean(name='mean_loss_train_test')
        
        model = self.models['f_nn']
        
        # define the training step to execute in each iteration with magic
        @tf.function
        def train_step(model_input_list, y_data):
        
            with tf.GradientTape() as tape:
                predictions = model(model_input_list, training=True)
                loss = loss_object(predictions, y_data)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables)
            )
            mean_loss(loss)

            return loss

        # define the test step to execute in each iteration with a magic handle
        @tf.function
        def test_step(model_input_list, y_data):
        
            predictions = model(model_input_list, training=False)
            t_loss = loss_object(predictions, y_data)
            mean_loss(t_loss)
        
        
        ###
        # Perform epochs of training and validation ###
        ###

        # create lists to save train and validation loss results for each epoch
        val_loss_history = []
        train_loss_history = []

        for epoch in range(self.epochs):

            if not self.silent:
            
                # tell which epoch we are at
                print('Epoch {}/{}'.format(epoch + 1, self.epochs))


            ###
            # Training ###
            ###

            # Reset the metrics at the start of the next epoch
            mean_loss.reset_states()
            
            # Shuffle training data
            self.f_randomize('train')
            
            # get number of datapoints
            n_datapoints = len(self.y_train)

            if not self.silent:
            
                # tell us that we start training now
                print('Training:')

                # create a progress bar for training
                progbar = tf.keras.utils.Progbar(
                    math.floor(n_datapoints - self.batch_size) / self.batch_size,
                    stateful_metrics=['loss'],
                )

            # iterate over training data in BATCH_SIZE steps
            for i in range(
                0, 
                n_datapoints - self.batch_size, 
                self.batch_size
            ):

                # call function to create batched model inputs and labels
                model_input_list, y_batched = self.create_batched_data('train', i)

                # Execute the training step for this batch
                train_step(model_input_list, y_batched)

                # update the progress bar
                if not self.silent:
                
                    values = [('loss', mean_loss.result().numpy())]
                    progbar.add(1, values=values)

            # add training loss to history
            train_loss_history.append(mean_loss.result().numpy())


            ###
            # Validation ###
            ###

            # Reset the metrics at the start of the next epoch
            mean_loss.reset_states()

            # Shuffle validation data
            self.f_randomize('val')
            
            # get number of datapoints
            n_datapoints = len(self.y_val)

            if not self.silent:
            
                # Tell us that we start validating now
                print('Validation:')

                # create a progress bar for validation
                progbar = tf.keras.utils.Progbar(
                    math.floor(n_datapoints - self.batch_size) / self.batch_size,
                    stateful_metrics=['loss'],
                )

            # iterate over validation data in BATCH_SIZE steps
            for i in range(
                0, 
                n_datapoints - self.batch_size, 
                self.batch_size
            ):

                # call function to create batched model inputs and labels
                model_input_list, y_batched = self.create_batched_data('val', i)

                # Execute the testing step for this batch
                test_step(model_input_list, y_batched)

                # update the progress bar
                if not self.silent:
                
                    values = [('loss', mean_loss.result().numpy())]
                    progbar.add(1, values=values)

            # add validation loss to history
            val_loss_history.append(mean_loss.result().numpy())


            ###
            # implement early break here ###
            ###

            if self.monitor == 'val_loss':
                current_minimum = min(val_loss_history[-self.patience :])
                
            elif self.monitor == 'loss':
                current_minimum = min(train_loss_history[-self.patience :])
                
            elif self.monitor is None:
                continue

            if epoch == 0:
                total_minimum = current_minimum
                
            if current_minimum > total_minimum:
                break
                
            else:
                total_minimum = min(total_minimum, current_minimum)
        
        
        
        # Plot training and validation history
        if self.plot:

            plt.figure(figsize=(16, 8))
            plt.plot(train_loss_history)
            plt.plot(val_loss_history)
            plt.title('Training and validation loss history of neural network')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['training', 'validation'], loc='upper left')
            plt.show()
            
            # save figure
            file_name = fig_name +'.pdf'
            saving_path = self.path_to_results + file_name
            plt.savefig(saving_path)
            
            
        self.train_loss_history = train_loss_history
        self.val_loss_history = val_loss_history
        
    def encode_features(self):

        """ Encodes features using modules of embedding network.
        """

        if not self.silent:
            # tell us what we are doing
            print('Encoding features into embedded vector spaces for')


        ### Create random subsample before encoding if wanted ###

        # create an index array in the length of the passed dataset
        n_datapoints = self.n_datapoints_cand
        index_array = list(np.arange(n_datapoints))

        # if we chose a subset of these data points, create a random sub-sample
        if (
            self.subsample is not None and 
            self.subsample > 0 and
            self.subsample < 1
        ):

            n_datapoints = math.ceil(self.subsample * n_datapoints) 
            index_array = random.sample(index_array, n_datapoints)

        # create copy of dataset
        if self.x_t_cand is not None:
            x_t = self.x_t_cand[index_array]
        if self.x_s_cand is not None:
            x_s = self.x_s_cand[index_array]
        if self.x_st_cand is not None:
            x_st = self.x_st_cand[index_array]

        ### Encode features here ###
        if (
            self.x_t_cand is not None and 
            self.x_s_cand is not None and
            self.x_st_cand is not None
        ):
            model = self.models['x_joint_encoder']
            self.encoding = model.predict([x_t, x_s, x_st])
        elif (
            self.x_t_cand is not None and 
            self.x_s_cand is not None
        ):
            model = self.models['x_joint_encoder']
            self.encoding = model.predict([x_t, x_s])
        elif (
            self.x_t_cand is not None and 
            self.x_st_cand is not None
        ):
            model = self.models['x_joint_encoder']
            self.encoding = model.predict([x_t, x_st])
        elif (
            self.x_s_cand is not None and 
            self.x_st_cand is not None
        ):
            model = self.models['x_joint_encoder']
            self.encoding = model.predict([x_s, x_st])
        elif self.x_st_cand is not None:
            model = self.models['x_st_encoder']
            self.encoding = model.predict(x_st)
            
        elif self.x_t_cand is not None:
            model = self.models['x_t_encoder']    
            self.encoding = model.predict(x_t)
        
        elif self.x_s_cand is not None:
            model = self.models['x_s_encoder']
            self.encoding = model.predict(x_s)

        # save potentially subsampled index array
        self.index_array = index_array
        
    def compute_clusters(self):

        """ Computes clusters in feature embedded vector space.
        """

        if not self.silent:
            # tell us what we are doing
            print(
                'Creating clusters in encodings with n_clusters=', self.adl_batch_size
            )

        # set number of clusters equal to passed or corrected value
        clustering_method = self.method_cluster(n_clusters=self.adl_batch_size)

        # cluster encodings
        clustering_method.fit(self.encoding)
        cluster_labels = clustering_method.labels_
        cluster_centers = clustering_method.cluster_centers_
        
        # get ordinal encoder from Sklearn
        enc = OrdinalEncoder()
        
        # encode labels. NOTE: ordinally encoding clusters ensures that cluster
        # labels start at 0 and end at number of clusters, which is not the case
        # for X_t and X_s1 when not ordinally encoding.
        cluster_labels = enc.fit_transform(
            np.expand_dims(cluster_labels, 1)
        ).astype(int)
        
        # save cluster centers
        self.cluster_centers = cluster_centers[enc.categories_[0]]
        
        # delete expanded dimension again as it is redundant
        self.cluster_labels = cluster_labels[:, 0]
        
        # calculate number of clusters created in data
        self.n_clusters = max(cluster_labels) + 1
        
    def compute_similarity(self):

        """ Compute similarity score of candidates to cluster centers.
        """

        # set the number of encoded data points
        n_enc_datapoints = len(self.encoding)
        
        if not self.silent:
            # tell us what we are doing
            print("Calculating distances" )

            # create a progress bar for training
            progbar_distance = tf.keras.utils.Progbar(n_enc_datapoints)

        
        # CAUTION: create shape (n_enc_datapoints,) instead of (n_enc_datapoints, 1)
        similarity_array = np.zeros((n_enc_datapoints,))

        # iterate over all encodings
        for i in range(n_enc_datapoints):
            
            # get encoding's cluster label
            label = self.cluster_labels[i]
            
            # get cluster's center
            center = self.cluster_centers[label]
            
            # calculate similarity/closeness of encoding to its cluster center
            similarity_array[i] = self.method_distance(
                np.expand_dims(center, axis=0), np.expand_dims(self.encoding[i], axis=0)
            )
            
            if not self.silent:
                # increment progress bar
                progbar_distance.add(1)

        # save results
        self.similarity_array = similarity_array
        
    def collect(
        self,
        x_t_cand=None,
        x_s_cand=None,
        x_st_cand=None,
        budget=0.5,
        method='embedding_uncertainty',
        method_variant='max_uncertainty',
        method_distance='laplacian_kernel',
        method_cluster='KMeans',
        subsample=None,
        silent=True,
        plot=False
    ):
    
        """ Perform pool-based active learning using embedding uncertainty.
        """
        
        self.x_t_cand = x_t_cand
        self.x_s_cand = x_s_cand
        self.x_st_cand = x_st_cand
        self.budget = 0.5
        self.method = method
        self.method_variant = method_variant 
        exec('self.method_distance = {}'.format(method_distance))
        exec('self.method_cluster = {}'.format(method_cluster))
        self.subsample = subsample
        self.silent = silent
        self.plot = plot
        
        # set number of datapoints
        if x_t_cand is not None:
            self.n_datapoints_cand = len(x_t_cand)
        elif x_s_cand is not None:
            self.n_datapoints_cand = len(x_s_cand)
        elif x_st_cand is not None:
            self.n_datapoints_cand = len(x_st_cand)
            
        self.adl_batch_size = math.ceil(self.n_datapoints_cand * budget)
        
        ### Encode data points
        self.encode_features()
        
        ### Calculate clusters
        self.compute_clusters()
        
        ### Compute similarity values for each candidate
        if method_variant != 'rnd_uncertainty':
            self.compute_similarity()
        
            if method_variant == 'max_uncertainty':
                self.similarity_array = -1 * self.similarity_array
        
        ### Choose data from clusters 
        
        # create zero array that is filled with cluster IDs for this batch
        batch_index_list = []
        inf_score_list = []

        # iterates over the batch_index_array up to cand_batch_size
        cluster_batch_counter = 0
        
        # iterates over clusters until n_clusters, then resets to 0
        # if cluster_batch_counter does not reached cand_batch_size
        cluster_index = 0
        
        # iterate over all clusters until cluster_batch_counter reaches 
        # cand_batch_size
        while cluster_batch_counter < self.adl_batch_size:

            # get an array of indices matching to currently iterated cluster 
            # ID
            index_array = np.where(self.cluster_labels == cluster_index)[0]

            # if the set is not empty
            if len(index_array) != 0:
            
                if method == 'rnd_uncertainty':
                    # choose one element at random from this index array
                    index_choice = np.random.choice(index_array)
                    
                else:
                    # get similarity values for matching index array
                    similarity_array = self.similarity_array[index_array]

                    if method == 'avg_ucertainty':
                        # turn into absolute difference to average similarity
                        similarity_array = abs(
                            similarity_array - np.mean(similarity_array)
                        )

                    # calculate largest similarity
                    max_similarity = similarity_array.max()
                    
                    # choose first/largest value from similarity_array
                    index_choice = index_array[
                        np.where(
                            similarity_array == max_similarity
                        )[0][0]
                    ]
                    
                    # add information content score of data point to inf_score_list
                    inf_score_list.append(max_similarity)

                # add chosen index to zero array
                batch_index_list.append(self.index_array[index_choice])

                # setting the cluster ID to -1 excludes data point from  
                # considerations in next iterations of this loop
                self.cluster_labels[index_choice] = -1

                # increment the counter for already added data points to
                # zero array
                cluster_batch_counter += 1

            # increment the cluster ID index for the next iteration
            cluster_index += 1

            # set cluster ID index to zero for next iteration if an entire 
            # round of iterations did not fill zero array
            if cluster_index >= self.n_clusters:
                cluster_index = 0
        
        # Save results
        self.batch_index_list = batch_index_list
        self.inf_score_list = inf_score_list
        
        
    def predict(
        self,
        y_pred=None,
        x_t_pred=None,
        x_s_pred=None,
        x_st_pred=None,
        silent=True,
        plot=False,
    ):

        """ Predict labels for unqueried candidate data.
        """
        
        def plot_true_vs_prediction(test_data_Y, predictions):

            """ Visualizes predictions vs. true values. """

            plot_rows = 3
            plot_clos = 3

            # create a matplotlib.pyplot.subplots figure
            fig, ax = plt.subplots(plot_rows, plot_clos, figsize=(16, 16))

            # set the figtitle
            fig.suptitle('True vs. predicted', fontsize=16)

            # pick at random a set of integers for visualization
            data_indices = np.random.randint(
                0, 
                len(test_data_Y), 
                plot_rows * plot_clos
            )

            # create a variable for iteratively adding number of subplots
            subplot_counter = 0

            # iterate over number of rows
            for i in range(plot_rows):

                # iterate over number of columns
                for j in range(plot_clos):

                    # choose currently iterated random index
                    data_index = data_indices[subplot_counter]

                    # plot the true values
                    plot1 = ax[i, j].plot(test_data_Y[data_index])

                    # plot the predicted values
                    plot2 = ax[i, j].plot(predictions[data_index])

                    # increment the subplot_counter
                    subplot_counter += 1

            # add a figure legend
            fig.legend(
                [plot1, plot2],
                labels=['true', 'predicted'],
                fontsize=16,
            )
            
            # save figure
            file_name = 'true_vs_pred_cand.pdf'
            saving_path = self.path_to_results + file_name
            fig.savefig(saving_path)

        ### Reset the state of the test loss metric
        loss_function = tf.keras.losses.mean_squared_error
        mean_loss = tf.keras.metrics.Mean(name='mean_loss_train_test')
        mean_loss.reset_states()


        ### Make predictions
        model = self.models['f_nn']
        model_input_list = []
        if self.x_t is not None:
            model_input_list.append(x_t_pred)
        if self.x_s is not None:
            model_input_list.append(x_s_pred)
        if self.x_st is not None:
            model_input_list.append(x_st_pred)
        predictions = model.predict(model_input_list)
        
        self.silent = silent
        self.plot = plot

        ###
        # Calculate the testing loss ###
        ###
        
        if y_pred is not None:
            # calculate the testing losses
            t_loss = loss_function(y_pred, predictions)

            # take the mean of single losses
            testing_loss = mean_loss(t_loss).numpy()

            # tell us how much testing loss we have
            if not self.silent:
            
                print('loss:', testing_loss)
                
            # Plot exemplar predictions
            if self.plot:
                plot_true_vs_prediction(y_pred, predictions)

        else:
            testing_loss = None
            
        self.predictions = predictions
        self.testing_loss = testing_loss
        
