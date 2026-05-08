import numpy as np
from .battery import battery


class electric_vehicle(battery):
    def __init__(self,config,step):
        super().__init__(config,step)

    def update(self,action,plugged=False,departure=False):
        bat_rew=0
        if  not plugged:
           action=0  # No charging/discharging if not plugged in
        # Penalty for departing with low SOC (below 80%)
        if departure and self.SOC < 0.8:
            bat_rew -= 10.0*(1-self.SOC)  # Penalty for departing with low SOC
        
        if plugged:
            E, rew = super().update(action)
            bat_rew += rew
        else:
            E=0
            self.SOC =0
        return E, bat_rew