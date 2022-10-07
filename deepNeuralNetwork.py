

#Disable TensorFlow Information printouts
import warnings
warnings.filterwarnings('ignore')

#Keras modules
from keras.models import Sequential
from keras.layers import Dense
import keras.optimizers as opt
from keras.models import load_model

'''
Constants
'''
C_VERBOSE_NONE   = 0  # Printing is disabled
C_VERBOSE_INFO   = 1  # Only information printouts (constructor)
C_VERBOSE_DEBUG  = 2  # Debugging printing level (all printouts)
    

class DeepNeuralNetwork(object):
    

    def __init__(self, file_name = None, inputs = None, outputs = None, hidden_layers = None, hidden_layers_size = None,
                 optimizer_learning_rate = 0.001, seed = None, verbose = C_VERBOSE_NONE):
               
        self.__verbose = verbose
        
        #If file_name is present, just loads the model from the file
        if file_name is not None:
            if self.__verbose > C_VERBOSE_NONE:
                print('\nDeep Neural Network object initialization (file_name = ', file_name, ')', sep = '')
        
            self.__model = load_model(file_name)
            return None
            
        if self.__verbose > C_VERBOSE_NONE:
            print('\nDeep Neural Network object initialization (inputs = ', inputs, ', outputs = ', outputs,
                  ', hidden_layers = ', hidden_layers, ', hidden_layers_size = ', hidden_layers_size, 
                  ', optimizer_learning_rate = ', optimizer_learning_rate, ', seed = ', seed, ')', sep = '')
        
        #Applies the given seed to the Keras (with Tensor Flow backend)
        if seed is not None:
            self.__applySeed(seed)
        
        #Create a sequential model 
        self.__model = Sequential()

        #Create first layer (use 'relu' as activation function, hardcoded)
        self.__model.add(Dense(units = hidden_layers_size, activation = 'relu', input_dim = inputs))
        
        #Create hidden layers (use 'relu' as activation function, hardcoded)
        for i in range(hidden_layers):
            self.__model.add(Dense(units = hidden_layers_size, activation = 'relu'))
        
        #Create last layer (use 'linear' as activation function, hardcoded)
        self.__model.add(Dense(units = outputs, activation = 'linear'))

        #Compile model, optimizer used is Adam with its defaults values, only learning rate is passed
        #for experimenting during the model complexity analysis.
        self.__model.compile(loss = 'mse', optimizer = opt.Adam(lr = optimizer_learning_rate))
                   
    
    def __applySeed(self, seed):
        '''
            see: 
            https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development 
        '''
        
        if self.__verbose > C_VERBOSE_INFO:
            print('Apply Seed to the \'Keras\' with the \'Tensor Flow\' Backend environment (seed = ', seed, ')', sep = '')
        
        import tensorflow
        from keras import backend

        #Sets random seed for the tensor flow and limits the parallel threds to one.
        tensorflow.set_random_seed(seed)     
        backend.set_session(tensorflow.Session(graph = tensorflow.get_default_graph(), 
            config = tensorflow.ConfigProto(intra_op_parallelism_threads = 1,
            inter_op_parallelism_threads = 1)))
            
    
    def train(self, X, Y):
        

        if self.__verbose > C_VERBOSE_INFO:
            print('Deep Neural Network Train (training_instances = ', X.shape[0], ')', sep = '')
            
        #Train the model using all the default values, presented here for experimental reasons
        #and future use. Verbose is disabled.
        self.__model.fit(x = X, y = Y, batch_size = None, epochs = 1, verbose = False, callbacks = None, 
            validation_split = 0.0, validation_data = None, shuffle = True, class_weight = None, sample_weight = None, 
            initial_epoch = 0, steps_per_epoch = None, validation_steps = None)
            
        
    def predict(self, X):
        
        if self.__verbose > C_VERBOSE_INFO:
            print('Deep Neural Network Predict (prediction_instances = ', X.shape, ')', sep = '')
        
        #If there is only one instance reshape the array so each column holds each one of the feature values
        #See keras predict function.
        if len(X.shape) == 1:
            return self.__model.predict(x = X.reshape(1, X.shape[0]), batch_size = None, verbose = 0, steps = None)
        else:
            return self.__model.predict(x = X, batch_size = None, verbose = 0, steps = None)
    
    
    def saveModel(self, file_name):
        
        
        if self.__verbose > C_VERBOSE_INFO:
            print('Deep Neural Network Model Saved (file_name = ', file, ')', sep = '')
            
        self.__model.save(file_name)