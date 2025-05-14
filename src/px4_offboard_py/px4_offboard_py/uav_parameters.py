# uav_parameters.py
import math

class UAVParameters:
    def __init__(self):

        self.mass = 2.0  # kg，from x500_base/model.sdf 
        self.g = 9.81    
        self.kf = 8.54858e-6  # motorConstant
        self.km = 0.016
        self.omega_max = 1000.0  # maxRotVelocity rad/s
        self.num_motors = 4
        x = 0.174
        y = 0.174
        self.arm_length = math.sqrt(x**2 + y**2)

        self.fz_max = self.num_motors * self.kf * self.omega_max ** 2
        self.tauxy_max = self.kf * self.arm_length * self.omega_max **2
        self.tauz_max = self.km * self.kf * self.omega_max **2

        self.print_summary()

    def print_summary(self):
        print(f"  Max Torque xy         {self.tauxy_max}")
        print(f"  Max Torque z         {self.tauz_max}")
        print(f"  Max Total Thrust:  {self.fz_max:.2f} N\n")


    def get_fz_max(self):
        return self.fz_max

    def get_tau_max(self):
        return self.tauxy_max, self.tauz_max
