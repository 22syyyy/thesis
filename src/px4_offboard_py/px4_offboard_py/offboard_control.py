#!/usr/bin/env python3

import rclpy
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus, VehicleOdometry, VehicleThrustSetpoint, VehicleTorqueSetpoint, VehicleRatesSetpoint
from px4_offboard_py.trajectory import TrajectoryGenerator
from px4_offboard_py.PPC_controller import PPCController
from px4_offboard_py.uav_parameters import UAVParameters


class OffboardControl(Node):
    """Node for controlling a vehicle in offboard mode."""

    def __init__(self) -> None:
        super().__init__('offboard_control_takeoff_and_land')

        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)

        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        
        self.thrust_setpoint_publisher = self.create_publisher(
            VehicleThrustSetpoint, '/fmu/in/vehicle_thrust_setpoint', qos_profile)
        self.torque_setpoint_publisher = self.create_publisher(
            VehicleTorqueSetpoint, '/fmu/in/vehicle_torque_setpoint', qos_profile)
        self.omega_setpoint_publisher = self.create_publisher(
            VehicleRatesSetpoint, '/fmu/in/vehicle_rates_setpoint', qos_profile)
    
        
        # Create subscribers
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v1', self.vehicle_status_callback, qos_profile)
        self.vehicle_odometry_subscriber = self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry', self.vehicle_odometry_callback, qos_profile)

        # Initialize variables
        
        self.offboard_setpoint_counter = 0
        
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()        
        self.vehicle_odemetry = VehicleOdometry()
        self.takeoff_height = -5.0

        self.target_pos = [0.0, 0.0, 0.0] 
        self.target_yaw = 0.0
        #self.traj_gen = TrajectoryGenerator()
        self.uav_para = UAVParameters()#get the uav parameters
        self.controller = PPCController()

        self.state = {} 

        # Create a timer to publish control commands

        self.timer = self.create_timer(0.001, self.timer_callback)

    def normalize_torque_and_thrust(self, Fz, tau):

        Fz_max = self.uav_para.get_fz_max()
        tau_xy_max, tau_z_max = self.uav_para.get_tau_max()
        normalized_torque_and_thrust = np.zeros(4)

        normalized_torque_and_thrust[0] = Fz / Fz_max
        self.Fz_ref = np.clip(self.Fz_ref, -1.0, 0.0)
        normalized_torque_and_thrust[1] = tau[0] / tau_xy_max
        normalized_torque_and_thrust[2] = tau[1] / tau_xy_max
        normalized_torque_and_thrust[3] = tau[2] / tau_z_max
        
        normalized_torque_and_thrust = np.clip(normalized_torque_and_thrust, -1.0, 1.0)
       

        return normalized_torque_and_thrust



    def vehicle_local_position_callback(self, vehicle_local_position):
        #self.get_logger().info("vehicle_local_position_callback triggered") 
        """Callback function for vehicle_local_position topic subscriber."""
        self.vehicle_local_position = vehicle_local_position
        #self.get_logger().info(f'Current Z:{self.vehicle_local_position.z}')

    def vehicle_status_callback(self, vehicle_status):
        """Callback function for vehicle_status topic subscriber."""
        #self.get_logger().info("vehicle_status_callback triggered") 
        self.vehicle_status = vehicle_status

    def vehicle_odometry_callback(self, vehicle_odometry):

        """Callback function for vehicle_odemetry topic subscriber."""
        self.vehicle_odometry = vehicle_odometry
        self.pos = np.array(self.vehicle_odometry.position)
        self.vel = np.array(self.vehicle_odometry.velocity)
        self.omega = np.array(self.vehicle_odometry.angular_velocity) # Angular velocity in body-fixed frame (rad/s)

        self.q = self.vehicle_odometry.q
        
        euler = np.zeros(3)
        if not any(map(lambda v: v != v, self.q)):  # NaN check
            q_xyzw = [self.q[1], self.q[2], self.q[3], self.q[0]]
            euler = R.from_quat(q_xyzw).as_euler('xyz', degrees=False)
            self.euler = np.array(euler)

        else:
            self.get_logger().warn("Received NaN quaternion — skipping orientation conversion.")

        #self.get_logger().info(f'Current position: x={self.pos[0]:.2f}, y={self.pos[1]:.2f}, z={self.pos[2]:.2f}')
        #self.get_logger().info(f'Current velocity: vx={self.vel[0]:.2f}, vy={self.vel[1]:.2f}, vz={self.vel[2]:.2f}')

        self.state = {
            'pos': self.pos,
            'vel': self.vel,
            'euler': self.euler,
            'omega': self.omega,
            'yaw': euler[2],
            'quat':q_xyzw
                }
    

    def arm(self):
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def disarm(self):
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self):
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")

    def land(self):
        """Switch to land mode."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")

    def publish_offboard_control_heartbeat_signal(self):
        """Publish the offboard control mode."""
        msg = OffboardControlMode()
        msg.position = False
        msg.velocity =  False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = True
        msg.thrust_and_torque = False
        msg.direct_actuator = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    # def publish_position_setpoint(self, x: float, y: float, z: float):
    #     """Publish the trajectory setpoint."""
    #     msg = TrajectorySetpoint()
    #     msg.velocity = [x, y, z]
    #     msg.yaw = 1.57079   # (90 degree)
    #     msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
    #     self.trajectory_setpoint_publisher.publish(msg)
    #     #self.get_logger().info(f"Publishing position setpoints {[x, y, z]}")
    
    def publish_thrust_setpoint(self, Fx: float, Fy: float, Fz: float):
        msg = VehicleThrustSetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.xyz = [Fx, Fy, Fz] 
        self.thrust_setpoint_publisher.publish(msg)

    def publish_torque_setpoint(self, Tx: float, Ty: float, Tz: float):
        msg = VehicleTorqueSetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.xyz = [Tx, Ty, Tz] 
        self.torque_setpoint_publisher.publish(msg)
        
    def publish_rates_setpoint(self, roll: float, pitch: float, yaw: float, Fz: float, quat):
        msg = VehicleRatesSetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.roll = roll
        msg.pitch = pitch
        msg.yaw = yaw
        
        F_ned = np.array([0.0, 0.0, Fz],dtype=np.float32)
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        R_BI = r.as_matrix().T
        msg.thrust_body = (R_BI @ F_ned).astype(np.float32)
        self.omega_setpoint_publisher.publish(msg)

    def publish_vehicle_command(self, command, **params) -> None:
        """Publish a vehicle command."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        
        self.vehicle_command_publisher.publish(msg)

    def timer_callback(self) -> None:
        """Callback function for the timer."""
        
        self.publish_offboard_control_heartbeat_signal()
       
        #print(self.offboard_setpoint_counter)

        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            self.arm()
            self.start_time = self.get_clock().now().nanoseconds / 1e9 

        if 'pos' not in self.state:
            self.get_logger().warn('No odometry data received yet, skipping this cycle.')
            return
        

        #if self.vehicle_local_position.z > self.takeoff_height and self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
        if  self.offboard_setpoint_counter >= 10 and self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            
            #self.get_logger().info('under PPC controlling') self.offboard_setpoint_counter == 10
           
            #self.target_pos, self.target_yaw = self.trajectory_generator.get_position(self.state['pos'])
            #self.target_pos, self.target_yaw = self.traj_gen.get_position(self.state['pos'])

            now = self.get_clock().now().nanoseconds / 1e9
            self.time = now - self.start_time
          

            self.get_logger().info(f'TIMETIMETIME:{self.time}')
            x = np.cos(self.time)/(1 + np.cos(self.time )**2)
            y = np.sin(self.time) * x
            z = - 1 - self.time/5

            self.target_pos = [0, 0, -5]
            self.target_yaw = 0.0
           
        
            self.tau, self.Fz_ref, self.v_ref, self.omega_ref = self.controller.PPC(state=self.state, target_pos = self.target_pos, target_yaw = self.target_yaw, t = self.time)
            [Tx, Ty, Tz] = self.tau
            [vx, vy, vz] = self.v_ref
            [roll, pitch, yaw] = self.omega_ref
            #[Fz, Tx, Ty, Tz]= self.normalize_torque_and_thrust(self.Fz_ref, self.tau)
            #self.get_logger().info('under PPC controlling')
            #self.get_logger().info(f"Fz_ref: {self.Fz_ref}")
            
            #self.get_logger().info(f"TAU: {self.tau}")


            # self.publish_thrust_setpoint(0.0, 0.0, self.Fz_ref)
            # self.publish_torque_setpoint(0.0, 0.0, 0.0)
            self.publish_rates_setpoint(roll, pitch, yaw, self.Fz_ref, self.q)
            print("THRUST", self.Fz_ref)
            #self.publish_torque_setpoint(Tx, Ty, Tz)
            #self.get_logger().info(f"Fz: {Fz:.6f}, Tx: {Tx:.3f}, Ty: {Ty:.3f}, Tz: {Tz:.3f}")

            # elif self.vehicle_local_position.z <= self.takeoff_height:
            #     self.land()
            #     exit(0)

        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1
        
        
        



def main(args=None) -> None:
    print('Starting offboard control node...')
    rclpy.init(args=args)
    offboard_control = OffboardControl()
    rclpy.spin(offboard_control)
    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)

