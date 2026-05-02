import sys
from pathlib import Path
from time import sleep
# Add project root to path
root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))
config_path = root / 'configs' / 'env.yaml'

from hems_core.env.components.battery import battery
from hems_core.env.components.EV import electric_vehicle as EV
from hems_core.utils.data_functions import pv_profile, price_profile, outdoor_temperature, load_profile, get_day_details, normalize_load, normalize_pv, normalize_price, normalize_temperature
import numpy as np
import yaml


class EnergyEnv:
    def __init__(self):
        with open(config_path,'r') as file:
            config = yaml.safe_load(file)
        self.dt = 0.25
        self.bat = battery(config["battery"],self.dt)
        self.ev = EV(config["ev"],self.dt)
        self.t_arr = 72  # EV arrives at 6 PM (72*15min = 8 hours)
        self.t_dep = 32  # EV departs at 8 AM (32*15min = 8 hours)
        #starting date and hour 01/03/2021 00:00
        self.idx = 0
        self.reset()



    def reset(self):
            return self.get_state()

    def get_state(self):
        h = self.idx % 96
        day = get_day_details(self.idx)

        
        return np.array([normalize_temperature(outdoor_temperature(self.idx)),
                        normalize_load(load_profile(self.idx)),
                        self.bat.SOC,
                        self.ev.SOC,
                        normalize_pv(pv_profile(self.idx)),
                        normalize_price(price_profile(self.idx)),
                        day['season']/4.0,
                        day["day_type"]/7.0,
                        h/96.0]
                        , dtype=np.float32)
    def step(self,action):

            # battery model
            P_bat, bat_reward =self.bat.update(action[0])

            # EV charging
            h = self.idx % 96
            plugged = (h<=self.t_dep or h>=self.t_arr)
            departure = (h == self.t_dep)
            P_ev, ev_reward = self.ev.update(action[1], plugged=plugged,departure=departure)



        # Power balance
            P_pv = pv_profile(self.idx)
            load = load_profile(self.idx)
            P_grid = max(0.0, P_bat + load+ P_ev - P_pv)
            price = price_profile(self.idx)
            cost = P_grid * price *self.dt


            reward = -cost + bat_reward + ev_reward

            self.idx += 1
            done = (self.idx % 96 == 0)
            info= {'cost':cost , 'ev_reward':ev_reward, 'bat_reward':bat_reward,'P_grid':P_grid}
            return self.get_state(), reward, done, info


    def sample_random_action(self):
            return np.random.uniform(-1.0,1.0,2)

if __name__ == "__main__":
    env = EnergyEnv()
    state = env.reset()
    for _ in range(10):
        action = env.sample_random_action()
        next_state, reward, done, info = env.step(action)
        if done:
            print("Episode finished, resetting environment.")
            state = env.reset()
 