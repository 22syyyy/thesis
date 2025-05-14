# ppc_controller.py
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

class PPCController:
    def __init__(self):

            self.rho_para = {
            'position': {
                'px': (12.0, 0.2, 0.4),
                'py': (12.0, 0.2, 0.4),
                'pz': (12.0, 0.2, 0.4)
            },
            'velocity': {
                'vx': (3.0, 0.5, 0.5),
                'vy': (3.0, 0.5, 0.5),
                'vz': (5.0, 0.2, 1.5)
            },
            'angular': {
                'phitheta1': (0.5, 0.25, 0.5),
                'phitheta2': (0.5, 0.25, 0.5),
                'psi': (0.4, 0.05, 0.1)
            },
            'omega': {
                'omega_phi': (0.3, 0.1, 0.5),
                'omega_theta': (0.3, 0.1, 0.5),
                'omega_psi': (0.3, 0.1, 0.5)
            }
        }



        # Gains from the simulation section in the paper
            self.kp = np.diag([1.25, 1.25, 12.5])
            self.kvz = 10.0
            self.kvxy = np.diag([1.0, 2.0])
            self.kphitheta = np.diag([3.0, 1.5])
            self.kpsi = 1.0
            self.komega = np.eye(3) * 10.0

    
    #PPC CONTROLLER
    #############
    def rho_calculation(self, layer, axis, t): 
        rho0, rho_inf, l = self.rho_para[layer][axis]
        return (rho0 - rho_inf) * math.exp( -l * t) + rho_inf

    def transformation(self, e, rho):
        xi = e / rho
        xi = np.clip(xi, -0.999, 0.999)  # avoiding devide by zero
        return np.log((1 + xi) / (1 - xi)) /2 
    
    def transformation_derivative(self, e, rho):
        xi = e / rho
        xi = np.clip(xi, -0.999, 0.999)
        return np.diag(1.0 / (1.0 - xi**2)) 


#################
    #FRAME TRANSFORM
    def body_to_inertial(self, body, quat): 
        r = R.from_quat(quat)
        R_IB = r.as_matrix()
        inertial = R_IB @ body
        return inertial


    def inertial_to_body(self, inertial, quat):

        r = R.from_quat(quat)
        R_IB = r.as_matrix()
        R_BI = R_IB.T
        body = R_BI @ inertial
        return body


################

    def PPC(self, state, target_pos, target_yaw, t):

        #get the current values from 'state'
        current_pos = np.array(state['pos'])
        current_vel = np.array(state['vel'])
        current_euler = np.array(state['euler'])
        
        current_omega_body = np.array(state['omega'])#Angular velocity in body-fixed
        quat = np.array(state['quat'])
        print("current_euler", current_euler)
    
        #position layer (# Velocity in NED frame)
        self.e_pos = current_pos - target_pos 
        print("e_pos =",self.e_pos)

        self.rho_pos = np.array ([
            self.rho_calculation('position', axis, t) for axis in ['px', 'py', 'pz']
             ])
    
        self.xi_pos = self.e_pos / self.rho_pos
        self.epsilon_pos = self.transformation(self.e_pos, self.rho_pos)
        self.r_pos = self.transformation_derivative(self.e_pos, self.rho_pos)

        
        rho_pos_inv = np.diag(1.0 / self.rho_pos)          # shape: (3,3)

        self.v_ref = -self.kp @ rho_pos_inv @ self.r_pos @ self.epsilon_pos  # shape: (3,)
        

        #velocity layer (# Velocity in NED frame)
        self.e_vel = current_vel - self.v_ref 
        print("e_vel =",self.e_vel)
        print("v_ref =",self.v_ref)
 
   
        self.rho_vel = np.array ([
            self.rho_calculation('velocity', axis, t) for axis in ['vx', 'vy', 'vz']
             ])

        self.xi_vel = self.e_vel / self.rho_vel
        self.epsilon_vel = self.transformation(self.e_vel, self.rho_vel)
        self.r_vel = self.transformation_derivative(self.e_vel, self.rho_vel)

        rho_vz_inv = 1.0 / self.rho_vel[2]
        r_vz = self.r_vel[2, 2]  
        eps_vz = self.epsilon_vel[2]

        self.Fz_ref = -self.kvz * rho_vz_inv * r_vz * eps_vz


        #subscriber from phi, theta, psi
        phi, theta, psi = current_euler
        Cphi, Sphi = np.cos(phi), np.sin(phi) 
        Ctheta, Stheta = np.cos(theta), np.sin(theta)
        Cpsi, Spsi = np.cos(psi), np.sin(psi)
        Ttheta = Stheta / Ctheta
        

        self.R_psi = np.array([
            [Cpsi, -Spsi],
            [Spsi, Cpsi]
            ])


        rho_vel_xy_inv = np.diag(1 / self.rho_vel[0:2])
        epsilon_vxy = self.epsilon_vel[0:2]
        r_vxy = self.r_vel[:2, :2]        

        
        self.T_phitheta_r = -self.kvxy @ self.R_psi.T @ rho_vel_xy_inv @ r_vxy @ epsilon_vxy / self.Fz_ref
        print("TTREF",self.T_phitheta_r )

        #angular layer 
        
        self.current_T_phitheta = np.array([Stheta * Cphi, -Sphi])
        self.e_T_phitheta = self.current_T_phitheta - self.T_phitheta_r 
        
        
        self.e_yaw = psi - target_yaw
        
 
        self.e_angular = np.concatenate((self.e_T_phitheta, [self.e_yaw]))
        print("e_angular =",self.e_angular)
       
        
        self.rho_angular = np.array ([
            self.rho_calculation('angular', axis, t) for axis in ['phitheta1', 'phitheta2', 'psi']
             ])
    
    
        self.xi_angular = self.e_angular / self.rho_angular
        self.r_angular = self.transformation_derivative(self.e_angular, self.rho_angular)
        self.epsilon_angular = self.transformation(self.e_angular, self.rho_angular)
      
        self.R_phitheta = np.array([[Cpsi/Ctheta, Spsi/Ctheta],
                                   [-Spsi, Cpsi]
                                   ])
        self.J_phitheta = np.array([[-Stheta*Sphi, Ctheta*Cphi],
                                    [-Cphi, 0]
                                    ])

       
        rho_inv_phitheta = np.diag(1.0 / self.rho_angular[0:2])
        
        #r_phitheta = np.diag(self.r_angular[0:2])
        r_phitheta = self.r_angular[:2,:2]

        eps_phitheta = self.epsilon_angular[0:2]
        current_omega_inertial = np.array(self.body_to_inertial(current_omega_body, quat))


        A = self.kphitheta @ np.linalg.inv(self.R_phitheta) @ np.linalg.inv(self.J_phitheta) @ rho_inv_phitheta @ r_phitheta @ eps_phitheta
        B = self.kpsi * (1 / self.rho_angular[2]) * self.r_angular[2, 2]  * self.epsilon_angular[2] + current_omega_inertial[0] * Cpsi * Ttheta + current_omega_inertial[1]* Spsi * Ttheta

        self.omega_ref_inertial = -np.concatenate((A,[B]))
        self.omega_ref_body = self.inertial_to_body(self.omega_ref_inertial, quat)

        #angular velocities
        self.e_omega = current_omega_inertial - self.omega_ref_inertial
        print("e_omega INERTIAL =",self.e_omega)
        print("REF OMEGA BODY",self.omega_ref_body)
        
        self.rho_omega = np.array ([
          self.rho_calculation('omega', axis, t) for axis in ['omega_phi', 'omega_theta', 'omega_psi']
             ])
        
        self.xi_omega = self.e_omega / self.rho_omega
        self.r_omega = self.transformation_derivative(self.e_omega, self.rho_omega)
        self.epsilon_omega = self.transformation(self.e_omega, self.rho_omega)
        
        rho_omega_inv = np.diag(1.0 / self.rho_omega)          # shape: (3,3)
        self.tau_inertial = - self.komega @ rho_omega_inv @ self.r_omega @ self.epsilon_omega
        self.tau_body = self.inertial_to_body(self.tau_inertial, quat)

        return self.tau_body, self.Fz_ref, self.v_ref, self.omega_ref_body
        #All the values are in the drone body FRD frame and normalized in [-1, 1]
    


    #################