"""
PID Controllers for Drone Control

Implements industry-standard cascaded PID control loops
based on gym-pybullet-drones and PX4 autopilot architecture.

Control hierarchy:
Position → Velocity → Attitude → Angular Rate → Motors
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class PIDGains:
    """PID controller gains."""
    kp: float  # Proportional gain
    ki: float  # Integral gain
    kd: float  # Derivative gain

    # Anti-windup
    integral_limit: float = 10.0

    # Output limits
    output_min: float = -np.inf
    output_max: float = np.inf


class PIDController:
    """
    Basic PID controller with anti-windup.

    Implements:
    - Proportional control
    - Integral control with anti-windup
    - Derivative control with filtering
    """

    def __init__(self, gains: PIDGains):
        """
        Initialize PID controller.

        Args:
            gains: PID gains configuration
        """
        self.gains = gains

        # State
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0

        # Derivative filtering (low-pass filter)
        self.derivative_filter_alpha = 0.1  # Higher = less filtering

    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0

    def update(self, error: float, dt: float) -> float:
        """
        Update PID controller.

        Args:
            error: Current error (setpoint - measured)
            dt: Time step (seconds)

        Returns:
            Control output
        """
        # Proportional term
        p_term = self.gains.kp * error

        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = np.clip(
            self.integral,
            -self.gains.integral_limit,
            self.gains.integral_limit
        )
        i_term = self.gains.ki * self.integral

        # Derivative term with filtering
        if dt > 0:
            raw_derivative = (error - self.prev_error) / dt
            # Low-pass filter
            derivative = (
                self.derivative_filter_alpha * raw_derivative +
                (1 - self.derivative_filter_alpha) * self.prev_derivative
            )
            self.prev_derivative = derivative
        else:
            derivative = 0.0

        d_term = self.gains.kd * derivative

        # Total output
        output = p_term + i_term + d_term

        # Clamp output
        output = np.clip(output, self.gains.output_min, self.gains.output_max)

        # Store error for next iteration
        self.prev_error = error

        return output


class CascadedDroneController:
    """
    Cascaded PID controller for drone.

    Implements standard control architecture:
    Position PID → Velocity PID → Attitude PID → Motor mixing

    Based on:
    - gym-pybullet-drones DSLPIDControl
    - PX4 autopilot control architecture
    """

    def __init__(
        self,
        mass: float = 0.027,  # kg (Crazyflie mass)
        arm_length: float = 0.0397,  # m
        ixx: float = 1.4e-5,  # kg*m^2
        iyy: float = 1.4e-5,
        izz: float = 2.17e-5,
        kf: float = 3.16e-10,  # Thrust coefficient
        km: float = 7.94e-12,  # Torque coefficient
    ):
        """
        Initialize cascaded controller.

        Args:
            mass: Drone mass (kg)
            arm_length: Distance from center to motor (m)
            ixx, iyy, izz: Moments of inertia
            kf: Thrust coefficient
            km: Torque coefficient
        """
        self.mass = mass
        self.arm_length = arm_length
        self.inertia = np.array([ixx, iyy, izz])
        self.kf = kf
        self.km = km
        self.g = 9.81  # m/s^2

        # Position PID controllers (x, y, z)
        self.pos_pid_x = PIDController(PIDGains(kp=0.4, ki=0.0, kd=0.2))
        self.pos_pid_y = PIDController(PIDGains(kp=0.4, ki=0.0, kd=0.2))
        self.pos_pid_z = PIDController(PIDGains(kp=1.0, ki=0.1, kd=0.5))

        # Velocity PID controllers
        self.vel_pid_x = PIDController(PIDGains(
            kp=0.4, ki=0.05, kd=0.2,
            output_min=-np.pi/6, output_max=np.pi/6  # ±30 degrees
        ))
        self.vel_pid_y = PIDController(PIDGains(
            kp=0.4, ki=0.05, kd=0.2,
            output_min=-np.pi/6, output_max=np.pi/6
        ))
        self.vel_pid_z = PIDController(PIDGains(
            kp=1.0, ki=0.5, kd=0.0,
            output_min=0.0, output_max=2*mass*self.g  # Thrust limits
        ))

        # Attitude PID controllers (roll, pitch, yaw)
        self.att_pid_roll = PIDController(PIDGains(
            kp=70000.0, ki=0.0, kd=20000.0
        ))
        self.att_pid_pitch = PIDController(PIDGains(
            kp=70000.0, ki=0.0, kd=20000.0
        ))
        self.att_pid_yaw = PIDController(PIDGains(
            kp=70000.0, ki=500.0, kd=20000.0
        ))

        # Angular rate PID controllers (for finer control)
        self.rate_pid_roll = PIDController(PIDGains(kp=100.0, ki=0.0, kd=0.0))
        self.rate_pid_pitch = PIDController(PIDGains(kp=100.0, ki=0.0, kd=0.0))
        self.rate_pid_yaw = PIDController(PIDGains(kp=100.0, ki=0.0, kd=0.0))

    def reset(self):
        """Reset all controllers."""
        for pid in [
            self.pos_pid_x, self.pos_pid_y, self.pos_pid_z,
            self.vel_pid_x, self.vel_pid_y, self.vel_pid_z,
            self.att_pid_roll, self.att_pid_pitch, self.att_pid_yaw,
            self.rate_pid_roll, self.rate_pid_pitch, self.rate_pid_yaw,
        ]:
            pid.reset()

    def compute_control(
        self,
        state: np.ndarray,
        target_pos: np.ndarray,
        target_yaw: float = 0.0,
        dt: float = 1/48,  # 48 Hz default
    ) -> np.ndarray:
        """
        Compute motor commands to reach target position.

        Args:
            state: [pos(3), vel(3), rpy(3), angular_vel(3)] - 12D state
            target_pos: Target position [x, y, z]
            target_yaw: Target yaw angle (radians)
            dt: Time step (seconds)

        Returns:
            motor_rpm: [rpm1, rpm2, rpm3, rpm4]
        """
        # Extract state
        pos = state[0:3]
        vel = state[3:6]
        rpy = state[6:9]  # Roll, pitch, yaw
        ang_vel = state[9:12]

        # === Position Control → Target Velocity ===
        pos_error = target_pos - pos

        target_vel_x = self.pos_pid_x.update(pos_error[0], dt)
        target_vel_y = self.pos_pid_y.update(pos_error[1], dt)
        target_vel_z = self.pos_pid_z.update(pos_error[2], dt)

        target_vel = np.array([target_vel_x, target_vel_y, target_vel_z])

        # === Velocity Control → Target Attitude ===
        vel_error = target_vel - vel

        # Velocity control outputs desired tilts (in world frame)
        target_pitch = -self.vel_pid_x.update(vel_error[0], dt)  # Pitch for x
        target_roll = self.vel_pid_y.update(vel_error[1], dt)     # Roll for y
        target_thrust = self.vel_pid_z.update(vel_error[2], dt)   # Thrust for z

        # Add feedforward thrust to compensate gravity
        target_thrust += self.mass * self.g

        # === Attitude Control → Target Torques ===
        att_error_roll = target_roll - rpy[0]
        att_error_pitch = target_pitch - rpy[1]
        att_error_yaw = target_yaw - rpy[2]

        # Wrap yaw error to [-pi, pi]
        att_error_yaw = np.arctan2(np.sin(att_error_yaw), np.cos(att_error_yaw))

        target_torque_roll = self.att_pid_roll.update(att_error_roll, dt)
        target_torque_pitch = self.att_pid_pitch.update(att_error_pitch, dt)
        target_torque_yaw = self.att_pid_yaw.update(att_error_yaw, dt)

        target_torques = np.array([
            target_torque_roll,
            target_torque_pitch,
            target_torque_yaw
        ])

        # === Motor Mixing: Thrust + Torques → Motor RPMs ===
        motor_rpm = self.motor_mixing(target_thrust, target_torques)

        return motor_rpm

    def motor_mixing(
        self,
        total_thrust: float,
        torques: np.ndarray
    ) -> np.ndarray:
        """
        Convert thrust and torques to motor RPMs.

        Solves the mixing matrix equation:
        [T, τ_roll, τ_pitch, τ_yaw]^T = M * [rpm1^2, rpm2^2, rpm3^2, rpm4^2]^T

        Args:
            total_thrust: Total thrust force (N)
            torques: [τ_roll, τ_pitch, τ_yaw]

        Returns:
            motor_rpm: [rpm1, rpm2, rpm3, rpm4]
        """
        # Motor layout (X configuration):
        #     0 (CW)
        #       ^
        # 3 <-   -> 1 (CCW)
        #       v
        #     2 (CW)

        L = self.arm_length
        kf = self.kf
        km = self.km

        # Mixing matrix (from dynamics)
        # Thrust contribution: all motors positive
        # Roll: motors 0,3 positive, 1,2 negative
        # Pitch: motors 0,1 positive, 2,3 negative
        # Yaw: motors 0,2 negative (CW), 1,3 positive (CCW)

        # Build system: [T, τx, τy, τz] = A * [F1, F2, F3, F4]
        # Where Fi = kf * omega_i^2

        A = np.array([
            [1, 1, 1, 1],  # Total thrust
            [L/np.sqrt(2), -L/np.sqrt(2), -L/np.sqrt(2), L/np.sqrt(2)],  # Roll
            [L/np.sqrt(2), L/np.sqrt(2), -L/np.sqrt(2), -L/np.sqrt(2)],  # Pitch
            [-km/kf, km/kf, -km/kf, km/kf],  # Yaw (torque)
        ])

        # Desired forces/torques
        desired = np.array([
            total_thrust,
            torques[0],
            torques[1],
            torques[2]
        ])

        # Solve for motor thrusts: F = A^-1 * desired
        try:
            motor_thrusts = np.linalg.solve(A, desired)
        except np.linalg.LinAlgError:
            # Singular matrix - use default hover
            motor_thrusts = np.array([total_thrust/4] * 4)

        # Clamp to positive thrust
        motor_thrusts = np.clip(motor_thrusts, 0, None)

        # Convert thrust to RPM: T = kf * omega^2 → omega = sqrt(T / kf)
        motor_omega = np.sqrt(motor_thrusts / kf)
        motor_rpm = motor_omega * 60 / (2 * np.pi)  # rad/s to RPM

        # Clamp RPM to motor limits (e.g., 21702 for Crazyflie)
        max_rpm = 21702
        motor_rpm = np.clip(motor_rpm, 0, max_rpm)

        return motor_rpm

    def compute_velocity_control(
        self,
        state: np.ndarray,
        target_vel: np.ndarray,
        target_yaw: float = 0.0,
        dt: float = 1/48,
    ) -> np.ndarray:
        """
        Compute motor commands for velocity tracking.

        Bypasses position control loop.

        Args:
            state: [pos(3), vel(3), rpy(3), angular_vel(3)]
            target_vel: Target velocity [vx, vy, vz]
            target_yaw: Target yaw angle
            dt: Time step

        Returns:
            motor_rpm: [rpm1, rpm2, rpm3, rpm4]
        """
        vel = state[3:6]
        rpy = state[6:9]

        vel_error = target_vel - vel

        # Velocity control → attitude targets
        target_pitch = -self.vel_pid_x.update(vel_error[0], dt)
        target_roll = self.vel_pid_y.update(vel_error[1], dt)
        target_thrust = self.vel_pid_z.update(vel_error[2], dt) + self.mass * self.g

        # Attitude control
        att_error_roll = target_roll - rpy[0]
        att_error_pitch = target_pitch - rpy[1]
        att_error_yaw = np.arctan2(np.sin(target_yaw - rpy[2]), np.cos(target_yaw - rpy[2]))

        torques = np.array([
            self.att_pid_roll.update(att_error_roll, dt),
            self.att_pid_pitch.update(att_error_pitch, dt),
            self.att_pid_yaw.update(att_error_yaw, dt),
        ])

        return self.motor_mixing(target_thrust, torques)

    def compute_attitude_control(
        self,
        state: np.ndarray,
        target_thrust: float,
        target_rpy: np.ndarray,
        dt: float = 1/48,
    ) -> np.ndarray:
        """
        Compute motor commands for attitude control only.

        Direct attitude control with specified thrust.

        Args:
            state: [pos(3), vel(3), rpy(3), angular_vel(3)]
            target_thrust: Desired thrust force (N)
            target_rpy: Target roll, pitch, yaw
            dt: Time step

        Returns:
            motor_rpm: [rpm1, rpm2, rpm3, rpm4]
        """
        rpy = state[6:9]

        # Attitude errors
        att_error = target_rpy - rpy
        att_error[2] = np.arctan2(np.sin(att_error[2]), np.cos(att_error[2]))  # Wrap yaw

        torques = np.array([
            self.att_pid_roll.update(att_error[0], dt),
            self.att_pid_pitch.update(att_error[1], dt),
            self.att_pid_yaw.update(att_error[2], dt),
        ])

        return self.motor_mixing(target_thrust, torques)

    def compute_rate_control(
        self,
        state: np.ndarray,
        target_rates: np.ndarray,
        target_thrust: float,
        dt: float = 1/48,
    ) -> np.ndarray:
        """
        Compute motor commands for direct angular-rate tracking.

        Args:
            state: [pos(3), vel(3), rpy(3), ang_vel(3)]
            target_rates: Desired body rates [p, q, r] (rad/s)
            target_thrust: Desired total thrust (N)
            dt: Time step

        Returns:
            motor_rpm: [rpm1, rpm2, rpm3, rpm4]
        """
        ang_vel = state[9:12]
        rate_error = target_rates - ang_vel

        torques = np.array([
            self.rate_pid_roll.update(rate_error[0], dt),
            self.rate_pid_pitch.update(rate_error[1], dt),
            self.rate_pid_yaw.update(rate_error[2], dt),
        ])

        return self.motor_mixing(target_thrust, torques)


class SimplePIDController:
    """
    Simple single-axis PID for quick testing.
    """

    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, setpoint: float, measurement: float, dt: float) -> float:
        error = setpoint - measurement
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


# Example usage and testing
if __name__ == "__main__":
    print("Cascaded Drone Controller Test")
    print("=" * 50)

    # Create controller for Crazyflie 2.x
    controller = CascadedDroneController(
        mass=0.027,
        arm_length=0.0397,
    )

    # Simulate hovering at z=1.0m
    dt = 1/48  # 48 Hz control

    # Current state: [pos, vel, rpy, ang_vel]
    state = np.array([
        0.0, 0.0, 0.5,  # Position: slightly below target
        0.0, 0.0, 0.0,  # Velocity: stationary
        0.0, 0.0, 0.0,  # Attitude: level
        0.0, 0.0, 0.0   # Angular velocity: not rotating
    ])

    # Target: hover at (0, 0, 1.0)
    target_pos = np.array([0.0, 0.0, 1.0])
    target_yaw = 0.0

    # Compute control
    motor_rpm = controller.compute_control(state, target_pos, target_yaw, dt)

    print(f"\nCurrent altitude: {state[2]:.2f} m")
    print(f"Target altitude: {target_pos[2]:.2f} m")
    print(f"\nMotor RPMs:")
    for i, rpm in enumerate(motor_rpm):
        print(f"  Motor {i}: {rpm:.0f} RPM")

    print(f"\nAverage RPM: {np.mean(motor_rpm):.0f}")

    # Estimate thrust
    kf = 3.16e-10
    omega = motor_rpm * 2 * np.pi / 60
    thrust_per_motor = kf * omega**2
    total_thrust = np.sum(thrust_per_motor)

    print(f"Total thrust: {total_thrust:.4f} N")
    print(f"Weight: {0.027 * 9.81:.4f} N")
    print(f"Thrust-to-weight: {total_thrust / (0.027 * 9.81):.2f}")
