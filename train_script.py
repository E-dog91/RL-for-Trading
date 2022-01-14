import random
import json
import sys
import gym
from gym import spaces
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Markets_env import MultiAssetTradingEnv
from stable_baselines3 import PPO,DDPG,SAC,TD3,A2C
from AdvisorClass import SimpleAdvisor, RNNAdvisor
#import sys
#sys.path.insert(0,'/content/drive/My Drive/ColabNotebooks')

np.random.seed(0)


data = pd.read_csv('DATA/Gemini 1hr timeframe/BTCUSD_1h.csv')
data = data.sort_values('Date')
d1 = data[-2000:]
data = data[-5000:-2000]
#print(data)
#TODO: take only data from 2016
#print(data)

data = data.drop(columns={'Unix Timestamp','Date','Symbol'})
d1 = d1.drop(columns={'Unix Timestamp','Date','Symbol'})

data2 = pd.read_csv('DATA/Gemini 1hr timeframe/ETHUSD_1h.csv')
data2 = data2.sort_values('Date')
d2 = data2[-2000:]
data2 = data2[-5000:-2000]
#print(data2)
data2 = data2.drop(columns={'Unix Timestamp','Date','Symbol'})
d2 = d2.drop(columns={'Unix Timestamp','Date','Symbol'})

data = data.reset_index(drop=True)
data2 = data2.reset_index(drop=True)

d1 = d1.reset_index(drop=True)
d2 = d2.reset_index(drop=True)

data3 = pd.read_csv('DATA/Gemini 1hr timeframe/ZECUSD_1h.csv')
data3 = data3.sort_values('Date')
d3 = data3[-2000:]
data3 = data3[-5000:-2000]

d3 = d3.drop(columns={'Unix Timestamp','Date','Symbol'})
data3 = data3.drop(columns={'Unix Timestamp','Date','Symbol'})

data4 = pd.read_csv('DATA/Gemini 1hr timeframe/LTCUSD_1h.csv')
data4 = data4.sort_values('Date')
d4 = data4[-2000:]
data4 = data4[-5000:-2000]


d4 = d4.drop(columns={'Unix Timestamp','Date','Symbol'})
data4 = data4.drop(columns={'Unix Timestamp','Date','Symbol'})

data3 = data3.reset_index(drop=True)
data4 = data4.reset_index(drop=True)

d3 = d3.reset_index(drop=True)
d4 = d4.reset_index(drop=True)

# train set
train_data = [data, data2, data3, data4]

#test set
test_data = [d1, d2, d3, d4]

timesteps = 200000
fortune = 10000

simple_advisor = SimpleAdvisor()
recurrent_advisor = RNNAdvisor(num_features=128,num_layers=1,num_assets=4,nonlinearity='tanh')

# creating the env

env = MultiAssetTradingEnv(train_data, reward_type='return', reward_scaling='identity', window_size=120,
                           max_steps=900, initial_fortune=fortune, advisors=[simple_advisor,recurrent_advisor])

md = DDPG('MlpPolicy', env=env,verbose=1)
md.learn(total_timesteps=200000)
md.save('DDPGenv1')
#md = A2C.load('A2Creturnarctanwindow_60steps_900num_feat64num_layer2.zip')
# 680 steps not that bad
test_env = MultiAssetTradingEnv(test_data, reward_type='return', reward_scaling='identity', window_size=120, max_steps=900,
                                initial_fortune=fortune, advisors=[simple_advisor,recurrent_advisor])
del env
#md = DDPG.load('DDPGreturntanhwindow_120steps_500num_feat64num_layer1.zip')
#md = DDPG.load('DDPG-PC-win100-4layer-64feat-800steps.zip')
obs = test_env.reset()
d = True
steps = []
profit = []
while d:
    action, _states = md.predict(obs)
    obs, rewards, dones, info = test_env.step(action)
    s,p = test_env.render()
    steps.append(s)
    profit.append(p)
    d = not dones

plt.plot(range(len(profit[:-1])), np.array(profit[:-1]))
print(f'Return: {100*profit[-2]/fortune} %')
plt.show()

