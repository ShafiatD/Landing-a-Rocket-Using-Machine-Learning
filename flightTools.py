import numpy as np
from numpy import array as A
from numpy.linalg import norm


def angle_wrap(t):
    """
    Wraps continuous angles to between -pi and pi
    """
    return np.arctan2(np.sin(t), np.cos(t))


class Rocket:

    def __init__(self):
        """
        Holds all the constant attributes of the rocket
        Units for each quantity are in the square brackets
        """
        self.hull_mass = 1000                            # Mass of the rocket w/o fuel [kg]
        self.fuel_mass = 5000                            # Initial mass of the fuel only [kg]
        self.width = 2                                   # Width [m]
        self.height = 20                                 # Height [m]
        self.impact_velocity = 5.0                       # Speed above which rocket crashes on impact [m/s]
        self.exhaust_velocity = 200 * 9.81               # Specific impulse of all rocket's engines [s]
        self.main_thrust = 100000                        # Maximum thrust of main engine [N]
        self.side_thrust = 50000                         # Maximum thrust of side engines [N]


class Flight:

    def __init__(self, flight_controller=None, reward_function=None, verbose=True, mode=0):
        """
        Keeps track of the data and runs the simulation for a single flight.
        Requires a flight controller (see starting notebook)
        and a reward function (see starting notebook).

        Verbose=True/False turns on/off printing the status as it runs.

        Can be run in one of three modes:

        0   Vertical motion only, all horizontal motion set to zero.
            In this mode the throttle can only have two values:
            0   Main booster off
            1   Main booster on

        1   As above plus left/right motion. The R/L boosters move the
            rocket L/R respectively. Throttle has six possible values:

                    Throttle value
                    0   1   2   3   4   5
            Booster
            M       0   0   0   1   1   1
            L       0   0   1   0   0   1
            R       0   1   0   0   1   0

        2   As in mode 0 plus ACW/CW rotation. Throttle takes the
            same values as mode 1 but now the R/L boosters rotate
            the rocket ACW/CW respectively.

        """

        # Initialise the rocket
        # Units for quantities are in [square brackets]
        self.rocket = Rocket()

        #  Simulation constants
        self.simulation_resolution = 0.1            # dt [s]
        self.max_runtime = 30.0                     # Max time before sim ends [s]
        self.gravitational_field = A([0.0, -9.81])  # Field strength [m/s^2]
        self.verbose = verbose
        self.sim_scale = 10.0 * self.rocket.height  # Height of sim 'roof' [m]
        self.base_size = 20.0                       # Size of the safe landing zone [m]
        self.mode = mode

        # Initial conditions
        # (all of these arrays will be appended to as the simulation runs)

        self.status = A([0]).astype('int')
        self.time = A([0.0])
        self.score = A([0.0])

        # Random starting position near the top of the screen
        initial_x = np.random.uniform(- self.sim_scale / 2 + self.rocket.height,
                                      self.sim_scale / 2 - self.rocket.height)
        initial_y = np.random.uniform(self.sim_scale / 2 + self.rocket.height,
                                      self.sim_scale - self.rocket.height)

        self.position = A([[initial_x * int(self.mode > 0), initial_y]])

        # Initial angle pointing (more or less) towards the base
        initial_angle = np.arctan2(initial_x, initial_y)
        self.angle = A([initial_angle]) * int(self.mode == 2)

        # Initial angular velocity of zero
        self.angular_v = A([0.0])

        # Random starting velocity in the direction of the base for mode==2,
        # completely random for mode==1
        initial_v = np.random.uniform(-25.0, -10.0)
        initial_v_x = (np.sin(initial_angle) * initial_v * int(self.mode == 2)) + (int(self.mode == 1) * np.random.uniform(-10.0, 10.0))
        initial_v_y = np.cos(initial_angle) * initial_v
        self.velocity = A([[initial_v_x * int(self.mode > 0), initial_v_y]])

        # Initial a = g
        self.acceleration = A([self.gravitational_field])

        # Initial mass = fuel + hull mass
        self.mass = A([self.rocket.hull_mass + self.rocket.fuel_mass])

        # Throttle begins in the off position
        self.throttle = A([0])

        # Bounding rectangle of the rocket for collision checks
        # (this has to rotate as the rocket rotates in mode 2)
        self.b_h = A(
            [self.rocket.height * np.abs(np.cos(self.angle[0])) + self.rocket.width * np.abs(np.sin(self.angle[0]))])
        self.b_w = A(
            [self.rocket.height * np.abs(np.sin(self.angle[0])) + self.rocket.width * np.abs(np.cos(self.angle[0]))])

        # Assign the passed functions
        self.flight_controller = flight_controller
        self.reward_function = reward_function

    def state_vector(self, mode):
        """
        Returns the state vector for the current state. The state
        vector is different in each mode...
        
        0   [y pos, y vel]
        1   [x pos, y pos, x vel, y vel]
        2   [x pos, y pox, x vel, y vel, angle, angular vel]
        
        All positions and velocities are normalised by the size of the simulation
        so that they're roughly around 0-1.
        """
        if mode == 0:
            # 
            state = A([
                self.position[-1][1] / self.sim_scale,
                self.velocity[-1][1] / self.sim_scale
            ])

        elif mode == 1:
            state = A([
                self.position[-1][0] / self.sim_scale,
                self.position[-1][1] / self.sim_scale,
                self.velocity[-1][0] / self.sim_scale,
                self.velocity[-1][1] / self.sim_scale
            ])

        elif mode == 2:
            state = A([
                self.position[-1][0] / self.sim_scale,
                self.position[-1][1] / self.sim_scale,
                self.velocity[-1][0] / self.sim_scale,
                self.velocity[-1][1] / self.sim_scale,
                self.angle[-1] / np.pi,
                self.angular_v[-1] / np.pi
            ])
        else:
            state = None

        return state

    def run(self):
        """
        Runs the simulation given this flight's initial conditions
        and flight controller
        """

        # Get the initital state vector
        state = self.state_vector(self.mode)

        i, done = 1, False
        # Start at time step 1 and run until max_runtime or the rocket lands/crashes
        while (not done) and (self.time[i - 1] < self.max_runtime):

            # Get the throttle position
            throttle = self.flight_controller(A([state]))

            # If rocket is out of fuel, cut the throttle
            if self.mass[i - 1] <= self.rocket.hull_mass:
                throttle = 0

            # Update the flight based on the throttle chosen by the controller
            state, reward, done = self.update(throttle)

            # Print the current status
            if self.verbose:
                update_text = 'T: {:05.2f} | {:<6}'.format(self.time[i - 1] + self.simulation_resolution,
                                                           self.status_string())
                print('\r', update_text, end='')

            # Iterate
            i += 1

    def update(self, throttle):
        """
        Updates the position, velocity and mass of the rocket at each
        timestep, given the previous state and the current throttle setting
        """

        # Set some numbers for convenience
        dt = self.simulation_resolution
        ve = self.rocket.exhaust_velocity

        # PHYSICS UPDATES -------------------------------------

        # Convert throttle selection to vector: ([M, L, R])
        M, L, R = thrust_parse(throttle, self.mode)
        delta_m_M = (M * self.rocket.main_thrust * dt) / ve
        delta_m_L = (L * self.rocket.side_thrust * dt) / ve
        delta_m_R = (R * self.rocket.side_thrust * dt) / ve

        # Update the total mass
        self.mass = np.append(self.mass, [self.mass[-1] - (delta_m_M + delta_m_L + delta_m_R)])

        # Update the throttle
        self.throttle = np.append(self.throttle, [throttle])

        # Update the acceleration based on the mass expulsion above
        # Note this calculation always uses the initial mass of the rocket
        # so the engine achieves the same acceleration regardless of how much fuel is in the rocket

        # Left/Right delta-v
        if self.mode < 2:
            delta_v_L = A([ve, 0]) * np.log((self.mass[0] + delta_m_L) / self.mass[0])
            delta_v_R = A([-ve, 0]) * np.log((self.mass[0] + delta_m_R) / self.mass[0])
            self.angular_v = np.append(self.angular_v, [0.0])
            self.angle = np.append(self.angle, [0.0])
            delta_v_M = A([0, ve]) * np.log((self.mass[0] + delta_m_M) / self.mass[0])
            total_a = ((delta_v_L + delta_v_R + delta_v_M) / dt) + self.gravitational_field

        # Angular delta-v
        else:
            delta_v_L = A([-ve]) * np.log((self.mass[0] + delta_m_L) / (self.mass[0]))
            delta_v_R = A([ve]) * np.log((self.mass[0] + delta_m_R) / (self.mass[0]))
            angular_a = (delta_v_L + delta_v_R) / (dt * self.rocket.height / 2)
            self.angular_v = np.append(self.angular_v, self.angular_v[-1] + angular_a * dt)
            self.angle = np.append(self.angle, angle_wrap(self.angle[-1] + self.angular_v[-1] * dt))
            delta_v_M = A([- ve * np.sin(self.angle[-1]), ve * np.cos(self.angle[-1])]) * np.log(
                (self.mass[0] + delta_m_M) / (self.mass[0]))
            total_a = (delta_v_M / dt) + self.gravitational_field

        # Update the acceleration
        self.acceleration = np.append(self.acceleration, [total_a], axis=0)

        # Update the velocity, position and time
        self.velocity = np.append(self.velocity, [self.velocity[-1] + total_a * dt], axis=0)
        self.position = np.append(self.position, [self.position[-1] + self.velocity[-1] * dt], axis=0)
        self.time = np.append(self.time, [self.time[-1] + dt])

        # Update the bounding rectangle
        self.b_h = np.append(self.b_h, A(
            [self.rocket.height * np.abs(np.cos(self.angle[-1])) + self.rocket.width * np.abs(np.sin(self.angle[-1]))]))
        self.b_w = np.append(self.b_w, A(
            [self.rocket.height * np.abs(np.sin(self.angle[-1])) + self.rocket.width * np.abs(np.cos(self.angle[-1]))]))

        # Collision check
        status_, done = self.status_check()
        self.status = np.append(self.status, [status_])

        # Calculate the reward
        reward_ = self.reward_function(self)
        self.score = np.append(self.score, [reward_])

        return self.state_vector(self.mode), reward_, done

    def status_check(self):
        """
        Checks the status of the rocket. Status codes:
        -1 Crashed into side/ocean
         0 Still flying
         1 Crashed into ship
         2 Successful landing
        """
        
        # Account for crashing before the next timestep
        vertical_dx = self.simulation_resolution * self.velocity[-1][1]
        
        # Check if rocket has landed safely
        if (((self.position[-1][1] - (self.b_h[-1] / 2) + vertical_dx) <= 0.0 and
            np.abs(self.position[-1][0]) <= self.base_size and
            norm(self.velocity[-1]) <= self.rocket.impact_velocity and
            norm(self.angular_v[-1]) <= 0.2) and
                norm(self.angle[-1]) <= np.pi / 8):
            done = True
            status = 2
            
        # Check if rocket hits the ship
        elif ((self.position[-1][1] - (self.b_h[-1] / 2) + vertical_dx) <= 0.0 and
                np.abs(self.position[-1][0]) <= self.base_size):
            done = True
            status = 1
            
        # Check if rocket hits the sea
        elif (self.position[-1][1] - (self.b_h[-1] / 2) + vertical_dx) <= 0.0:
            done = True
            status = -1

        # Check if rocket hits the top wall
        elif (self.position[-1][1] + (self.b_h[-1] / 2) + vertical_dx) >= self.sim_scale:
            done = True
            status = -1

        # Check if rocket hits the side
        elif (np.abs(self.position[-1][0]) + (self.b_w[-1] / 2)) >= (self.sim_scale / 2):
            done = True
            status = -1

        # Check if rocket has gone over time
        elif self.time[-1] > self.max_runtime:
            done = True
            status = 0

        else:
            done = False
            status = 0

        return status, done

    def status_string(self, k=-1):
        """
        Returns the current status of the rocket, given a code 0, 1 or 2
        """
        j = self.status[k]
        if abs(j) == 1:
            ss = 'Crashed'
        elif j == 0:
            ss = 'Flying'
        else:
            ss = 'Landed'
        return ss


def thrust_parse(j, mode=0):
    """
    j in binary gives the appropriate thrust selection:
    Translation:
    Input    0 1 2 4 5 6
    Output...
    Main     0 0 0 1 1 1 2^2
    Left     0 0 1 0 0 1 2^1
    Right    0 1 0 0 1 0 2^0
    """
    if mode == 0:
        thrust = A([j, 0, 0])
    else:
        if j > 2:
            k = j + 1
        else:
            k = j
        thrust = A([x for x in '{0:03b}'.format(k)]).astype(int)
    return thrust



