

import memory as mem
import numpy as np

'''
Constants
'''
C_VERBOSE_NONE   = 0  # Printing is disabled
C_VERBOSE_INFO   = 1  # Only information printouts (constructor)
C_VERBOSE_DEBUG  = 2  # Debugging printing level (all printouts)


class DeepQNetwork(object):
    

    
    def __init__(self, emulator, dnn, states_size, actions_number, memory_size, minibatch_size, gamma, epsilon, 
                 epsilon_decay_factor, seed = None, verbose = C_VERBOSE_NONE):
        
        self.__verbose = verbose  
            
        if self.__verbose > C_VERBOSE_NONE:
            print('\nDeep Q Network object created (states_size = ', states_size, ', actions_number = ', actions_number,
                  ', memory_size = ', memory_size, ', minibatch_size = ', minibatch_size, ', gamma = ', gamma,
                  ', epsilon = ', epsilon, ', epsilon_decay_factor = ', epsilon_decay_factor, ', seed = ', seed, 
                  ')', sep = '')
        
        #Seed the numpy random number generator
        if seed is not None:
            np.random.seed(seed)
            
        self.__dnn = dnn
        self.__actions_number = actions_number
        self.__states_size = states_size
        
        #Create a memory object instance
        self.__memory = mem.Memory(size = memory_size, type = mem.MemoryType.ROTATE, seed = seed, verbose = self.__verbose)
        
        self.__minibatch_size = minibatch_size
        self.__gamma = gamma
        self.__epsilon = epsilon 
        self.__epsilon_decay_factor = epsilon_decay_factor
        
        
    def decideAction(self, state):
        
        #With probability epsilon select a random action
        if np.random.random() < self.__epsilon:
            action = np.random.randint(0, self.__actions_number)
        #Otherwise select the best action from the DNN model
        else:
            action = np.argmax(self.__dnn.predict(state))
            
        if self.__verbose > C_VERBOSE_INFO:
            print('DQN Decide Action (state = ', state, ', action = ', action, ')', sep = '')
            
        #Reduce epsilon based on the decay factor
        self.__epsilon *= self.__epsilon_decay_factor
        
        return action
        
        
    def storeTransition(self, experience):
       
        if self.__verbose > C_VERBOSE_INFO:
            print('DQN Store Transition (experience = ', experience, ')', sep = '')
                    
        self.__memory.add(experience)
        
        
    def sampleRandomMinibatch(self):
        '''
        Summary: 
            Samples a random minibatch from the memory and trains with it the DNN.
    
        Args: 
            -
                
        Raises:
            -
            
        Returns:
            -
            
        notes:
            -
     
        '''
      
        minibatch = self.__memory.get(self.__minibatch_size)
        
        if self.__verbose > C_VERBOSE_INFO:
            print('DQN Sample Random Minibatch (minibatch_length = ', minibatch.shape[0], 
                  ', minibatch = ', minibatch, ')', sep = '')
        
        #End state s' was stored in memory as None (see class rlEmulator)
        #Replace it with a zero array for passing it to the model
        for i in range(minibatch.shape[0]):
            if minibatch[i, 3] is None:
                minibatch[i, 3] = np.zeros((self.__states_size), dtype = np.float64)
        
        #Predictions for starting state s and end state s' (see Algorithm 1 in [1])
        Q_start_state = self.__dnn.predict(np.stack(minibatch[:, 0], axis = 0))  #Q(s,a)   
        Q_end_state   = self.__dnn.predict(np.stack(minibatch[:, 3], axis = 0))  #Q(s',a)
        
        #Update the Q(s,a) according the Algorithm 1 in [1]
        for i in range(minibatch.shape[0]):
            #End state is a terminal state
            if np.array_equal(minibatch[i, 3], np.zeros((8))):
                Q_start_state[i, minibatch[i, 1]] = minibatch[i, 2]
            else:
                Q_start_state[i, minibatch[i, 1]] = minibatch[i, 2] + self.__gamma*np.amax(Q_end_state[i,:])
                
        #Train the dnn with the updated values of the Q(s,a)
        self.__dnn.train(np.stack(minibatch[:, 0], axis = 0), Q_start_state)
        
        