import numpy as np

class battery():
    def __init__(self,config,step):
        self.E_bat = config["capacity"] #battery total capacity
        self.charge_rate_max=config["max_charge_rate"] #maximum charging discharging rate
        self.SOC=0.0 # state of charge of the battery
        self.eta=config["efficiency"] #charging discharging efficiency
        self.dt = step # time step
    
    def update(self,action):
        E = action*self.charge_rate_max*self.dt # energy absorbed or supplied
        self.SOC+= (E*self.eta)/self.E_bat
        # Penalty for SOC outside [0, 1] - penalize both discharging below 0 and charging above 1
        bat_rew = -np.max([0, -self.SOC, self.SOC-1.0])
        self.SOC= np.clip(self.SOC,0,1)
        return E, bat_rew
    
    def reset(self):
        """Reset battery to initial state (SOC = 0)."""
        self.SOC = 0.0
