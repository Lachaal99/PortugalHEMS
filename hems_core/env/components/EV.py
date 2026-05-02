import numpy as np
from .battery import battery


class electric_vehicle(battery):
    def __init__(self,config,step):
        super().__init__(config,step)

    def update(self,action,plugged=False,departure=False):
        bat_rew=0
        if action < 0:
            bat_rew -= 5.0  # Penalty for trying to discharge the EV
        # Penalty for trying to charge when EV is not plugged in
        if action > 0 and not plugged:
            bat_rew -= 5.0
        # Penalty for departing with low SOC (below 80%)
        if departure and self.SOC < 0.8:
            bat_rew -= 10.0*(1-self.SOC)  # Penalty for departing with low SOC
        action = np.clip(action, 0.0, 1.0)
        if plugged:
            E, rew = super().update(action)
            bat_rew += rew
        else:
            E=0
            self.SOC =0
        return E, bat_rew