# 環境
import numpy as np
import gym
from gym import spaces

class SortEnv(gym.Env):
    def __init__(self, s_0):
        # gymの初期化
        super(SortEnv, self).__init__()
        N = len(s_0)
        self.N = N
        # 離散のInt
        self.action_space = spaces.Discrete(N-1)
        self.observation_space = spaces.Box(low=0, high=9, shape=(N,), dtype=np.int32)
        # 状態を初期化
        self.state = s_0
        self.state_0 = s_0
        self.sorted_list = sorted(s_0)

    def reset(self):
        self.state = self.state_0
        return self.state
    
    def step(self, action):
        state = self.state.copy()
        state[action], state[action+1] = state[action+1], state[action]
        self.state = state
        if state == self.sorted_list:
            reward = 1
            done = True
        else:
            reward = -2/(self.N*(self.N-1))
            done = False
        
        return state, reward, done, {}
    
    def render(self):
        print(self.state)