

from collections import deque
import numpy as np 
import pandas as pd
import gym

'''
Constants
'''
C_VERBOSE_NONE   = 0  # Printing is disabled
C_VERBOSE_INFO   = 1  # Only information printouts (constructor)
C_VERBOSE_DEBUG  = 2  # Debugging printing level (all printouts)

    
class Emulator():


    def __init__(self, scenario, average_reward_episodes = 1, statistics = False, rendering = False, 
                 seed = None, verbose = C_VERBOSE_NONE):
        

        self.__verbose = verbose  
        
        if self.__verbose > C_VERBOSE_NONE:
            print('\nRL Emulator initialization (scenario = ', scenario, ', average_reward_episodes = ', average_reward_episodes,
                  ', statistics = ', statistics, ', rendering = ', rendering, ', seed = ', seed, ')', sep = '')
        
        self.__rendering = rendering
        self.__statistics = statistics
        self.__current_state = None
        self.__average_reward_episodes = average_reward_episodes
        
        #Create Open AI gym environment
        try:
            self.__environment = gym.make(scenario)
        except:
            print('ERROR: class Emulator, \'', scenario, '\' is not a valid Open AI Gym scenario, script exits.', sep = '')
            exit(-1) 
            
        #Seed the environment
        if seed is not None:
            self.__environment.seed(seed)

        #Keep track of the last x consecutive total rewards
        self.__last_x_rewards = deque(maxlen = self.__average_reward_episodes)
             
        #Public Attributes
        self.emulator_started = False  #If False the Emulator not started, or episode finished
        self.average_reward = 0
        self.episode_number = 0
        self.episode_total_reward = 0
        self.episode_total_steps  = 0
        
        #Keeps statistics for all the episodes
        if self.__statistics:
            self.execution_statistics = pd.DataFrame(data = None, index = None, columns = ['episode', 'steps', 'total_reward', 
                'last_X_average_reward'], dtype = None, copy = False)
        
        #Get the observation space size based on its type
        if isinstance(self.__environment.observation_space, gym.spaces.box.Box):       
            self.state_size = 1
            for i in range(len(self.__environment.observation_space.shape)):
                self.state_size *= self.__environment.observation_space.shape[i]      
        
        elif isinstance(self.__environment.observation_space, gym.spaces.discrete.Discrete):
            self.state_size = self.__environment.observation_space.n           
        
        else:
            print('ERROR: class RLEmulator, \'', type(self.__environment.observation_space), '\' Observation Space type is not supported, script exits.')
            exit(-1)
            
        #Get the action space size based on its type
        if isinstance(self.__environment.action_space, gym.spaces.box.Box):          
            self.actions_number = 1
            for i in range(len(self.__environment.action_space.shape)):
                self.actions_number *= self.__environment.action_space.shape[i]
       
        elif isinstance(self.__environment.action_space, gym.spaces.discrete.Discrete):
            self.actions_number = self.__environment.action_space.n
            
        else:
            print('ERROR: class RLEmulator, \'', type(self.__environment.observation_space), '\' Actions Space type is not supported, script exits.')
            exit(-1)
            
        if self.__verbose > C_VERBOSE_NONE:
            print('- RL Emulator created (observations_size = ', self.state_size, ', actions_size = ', self.actions_number, ')', sep = '')
        
 
    def start(self):
        '''
        summary: 
            Starts or restarts the Emulator (starts an episode).
    
        Args:
            -
            
        Raises:
            -

        Returns:
            current_state: Environment's state object
                The current (initial) state of the Open AI gym environment. Returned
                after the OpenAI gym reset() function called.
            
        notes:
            -
     
        '''
        
        #Initialze attributes at the begining of the episode.
        self.emulator_started = True       
        self.episode_number += 1
        self.episode_total_reward = 0
        self.episode_total_steps  = 0
        
        #Reset the environment
        self.__current_state = self.__environment.reset()

        if self.__rendering:
            self.__environment.render()
            
        if self.__verbose > C_VERBOSE_INFO:
            print('RL Emulator Started')
  
        return self.__current_state
        
            
    def applyAction(self, action = None):
        '''
        summary: 
            Applies a specific or random action to the environment.
    
        Args:
            action: integer or None
                Action to be applied in the envoronment. If None, then a random action
                should be used.
        
        Raises:
            -
            
        Returns:
            experience_tuple: list
                An experience tuple for this step in the form of [s, a, r, s']
            
        notes:
            -
     
        '''
        
        if not self.emulator_started:
            print('ERROR: Emulator is not started yet, script exits.')
            exit(-1)
            
        #If no action given, select a random one
        if action is None:
            action = self.__environment.action_space.sample()

        new_state, reward, done, info = self.__environment.step(action)
        
        self.episode_total_reward += reward
        self.episode_total_steps  += 1
        
        if self.__rendering:
            self.__environment.render()
        
        if self.__verbose > C_VERBOSE_INFO:
            print('RL Emulator Apply Action (action = ', action, ', reward = ', reward, 
                  ', episode_is_done = ', done, ')', sep = '')
                  
        current_state = self.__current_state #Keep the current state for the return statememt
        
        #Episode finished
        if done:
            new_state = None
        
            #Update Emulator status, the caller has to restart the Emulator for another episode (start())
            self.emulator_started = False
            
            #Calculate average reward
            self.__last_x_rewards.append(self.episode_total_reward)
            self.average_reward = np.mean(self.__last_x_rewards)
            
            if self.__statistics:
                print('Statistics: episode = %05d' % self.episode_number, ', steps = %4d' % self.episode_total_steps, 
                      ', total_reward = %9.3f' % round(self.episode_total_reward, 3), ', ', self.__average_reward_episodes, 
                      '_episodes_average_reward = %9.3f' % round(self.average_reward, 3), sep = '')
                
                #Store the statistics for this episode
                self.execution_statistics.loc[len(self.execution_statistics.index)] = ([self.episode_number, self.episode_total_steps, 
                    round(self.episode_total_reward, 3), round(self.average_reward, 3)])
        
        #Episode is not done yet
        else:
            self.__current_state = new_state     #Update the current state for the next step
        
        #Return an experience tuple (list)
        return [current_state, action, reward, new_state]
        