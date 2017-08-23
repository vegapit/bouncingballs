import os
import pickle
import numpy as np

class ExperienceReplay( object ):

    @classmethod
    def load(cls, filename):
        if os.path.isfile(filename):
            return pickle.load( open(filename, "rb") )
        else:
            return None

    def __init__(self, max_size):
        self.max_size = max_size
        self.states = []
        self.actions = []
        self.rewards = []
        self.terminals = []
        self.next_states = []

    def size(self):
        return len(self.rewards)

    def add_experience(self, state, action, reward, terminal, next_state):
        if ( len(self.rewards) < self.max_size ):
            self.states.append( state )
            self.actions.append( action )
            self.rewards.append( reward )
            self.terminals.append( terminal )
            self.next_states.append( next_state )
        else:
            index = np.random.randint(self.max_size)
            self.states[index] = state
            self.actions[index] = action
            self.rewards[index] = reward
            self.terminals[index] = terminal
            self.next_states[index] =  next_state

    def get_batch(self, batch_size):
        indices = np.random.choice( len(self.rewards), batch_size, replace=False)
        return [self.states[i] for i in indices], [self.actions[i] for i in indices], [self.rewards[i] for i in indices], [self.terminals[i] for i in indices], [self.next_states[i] for i in indices]

    def save(self, filename):
        pickle.dump(self, open(filename, "wb"))