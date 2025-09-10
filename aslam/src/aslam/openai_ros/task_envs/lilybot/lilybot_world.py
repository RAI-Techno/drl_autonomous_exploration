from collections import deque

import numpy as np
import gym
import cv2
import tf.transformations

import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from rtabmap_msgs.srv import ResetPose
from std_srvs.srv import Empty

from aslam.openai_ros.robot_envs import lilybot_env


class LilyBotWorldEnv(lilybot_env.LilyBotEnv):
    """
    LilyBotWorldEnv: Custom OpenAI-Gym environment for LilyBot in simulation.

    This class handles:
        - Initialization (respawn robot and reset mapping)
        - Reward calculation
        - Episode termination
        - Acquiring states and setting actions for RL.
    """

    def __init__(self):
        """
        Initializes ROS parameters, Gym spaces, and internal state variables.
        """
        super().__init__()

        # ROS parameters
        self.n_actions = rospy.get_param('n_actions')           
        self.history_length = rospy.get_param('history_length') 
        self.episode_length = rospy.get_param('max_episode_length') 
        self.robot_name = rospy.get_param('robot_name')      
        self.init_poses = rospy.get_param('init_poses')       
        self.speeds = rospy.get_param('speeds')              
        self.map_area_threshold = rospy.get_param('map_area_threshold') 
        self.total_map_area = rospy.get_param('total_map_area') 


        # Define Gym state and action spaces
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.observation_space = gym.spaces.Dict({
            "local_occupancy_grid": gym.spaces.Box(
                low=0.0, high=1.0, shape=(self.history_length, 160, 160), dtype=np.float32
            ),
            "laser_scan": gym.spaces.Box(
                low=0.0, high=1.0, shape=(self.history_length, 580), dtype=np.float32
            )
        })

        # Buffers and internal states
        self.state_buffer = deque(maxlen=self.history_length)
        self.reward_range = (-30, 1.0)                      

        # Map-related variables
        self.last_map_area = 0.0         
        self.map_area = 0.0              
        self.last_smoothed_map_gain = 0.0 
        self.smoothed_map_gain = 0.0     

        # Exploration metrics
        self.trajectory_length = 0
        self.exploration_time  = 0
        self.termination_reason = None
        
        self.last_odom  = None                        # Used in trajectory length calculation
        self.start_time = rospy.Time.now().to_sec()   # Used in exploration time calculation
        

        # Action tracking
        self.last_action = None  

    #############################################################
    # Initialization
    #############################################################
    def _set_init_pose(self) -> bool:
        """
        Randomizes initial robot pose and resets RTAB-Map at the beginning of each episode.
        
        Returns:
            bool: True if initial pose successfully set
        """
        def random_yaw() -> float:
            """Generates a random yaw angle in radians between 0 and 2π."""
            return 2.0 * np.pi * np.random.rand()

        # Respawn until odometry readings are valid
        while True:
            # Pause RTAB-Map mapping process
            rospy.wait_for_service('/rtabmap/pause')
            rospy.ServiceProxy('/rtabmap/pause', Empty)()
            rospy.sleep(0.1)

            # Set robot pose in Gazebo simulation
            rospy.wait_for_service('/gazebo/set_model_state')
            set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            model_state = ModelState()
            model_state.model_name = self.robot_name
            ind = np.random.randint(len(self.init_poses))
            x, y, z = self.init_poses[ind][0], self.init_poses[ind][1], 0.0
            roll, pitch, yaw = 0.0, 0.0, random_yaw()
            q = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
            model_state.pose.position.x, model_state.pose.position.y, model_state.pose.position.z = x, y, z
            model_state.pose.orientation.x, model_state.pose.orientation.y, model_state.pose.orientation.z, model_state.pose.orientation.w = q
            set_model_state(model_state)
            rospy.sleep(0.2)

            # Reset RTAB-Map
            rospy.wait_for_service('/rtabmap/reset')
            rospy.ServiceProxy('/rtabmap/reset', Empty)()
            rospy.sleep(0.2)
            
            # Reset odometry in RTAB-Map
            rospy.wait_for_service('/rtabmap/reset_odom_to_pose')
            rospy.ServiceProxy('/rtabmap/reset_odom_to_pose', ResetPose)(x, y, z, roll, pitch, yaw)
            rospy.sleep(0.2)
            
            # Resume RTAB-Map
            rospy.wait_for_service('/rtabmap/resume')
            rospy.ServiceProxy('/rtabmap/resume', Empty)()
            rospy.sleep(0.2)

            # Validate odometry: retry if invalid
            odom = self.get_odom()
            if odom.pose.covariance[0] != 9999.0:
                break

        rospy.logdebug('Initial pose set successfully.')
        return True

    def _init_env_variables(self):
        """
        Resets internal variables at the start of a new episode.
        This includes collision flag, step counter, exploration metrics,
        map area tracking, and clearing the observation buffer.
        """
        self.collision = False
        self.step_num = 0
        self.last_map_area = 0.0
        self.map_area = 0.0
        self.last_smoothed_map_gain = 0.0
        self.smoothed_map_gain = 0.0
        self.state_buffer.clear()
        
        # Exploration metrics
        self.trajectory_length = 0
        self.exploration_time  = 0
        self.termination_reason = None
        
        self.last_odom  = self.get_odom()             # Used in trajectory length calculation
        self.start_time = rospy.Time.now().to_sec()   # Used in exploration time calculation

    #############################################################
    # Actions
    #############################################################
    def _set_action(self, action: int) -> None:
        """
        Executes a discrete action by mapping it to linear and angular velocities.

        Args:
            action (int): Action index corresponding to a velocity pair (v, w)
        """
        rospy.logdebug(f"Set Action ==> {action}")
        v, w = self.speeds[action]
        self.last_action = action
        self.move_base(v, w)
        rospy.logdebug(f"End Action ==> {action}")

    #############################################################
    # Observations
    #############################################################
    def _create_empty_observation(self) -> dict:
        """
        Creates a default empty observation frame.

        Returns:
            dict: Contains zeroed 'local_occupancy_grid' and 'laser_scan'
        """
        return {
            "local_occupancy_grid": np.zeros((160, 160), dtype=np.float32),
            "laser_scan": np.zeros(580, dtype=np.float32)
        }

    def _get_obs(self) -> dict:
        """
        Fetches current sensor observation (single-frame) without stacking.

        Returns:
            dict: Observation containing current occupancy grid and laser scan
        """
        self.process_sensors()
        self.calculate_map_area()
        return {
            "local_occupancy_grid": self.get_local_OG_map(),
            "laser_scan": self.get_laser_scan()
        }

    def _stack_observation(self, observation: dict) -> dict:
        """
        Stacks recent observations into a history buffer for RL state representation.
        Pads with oldest observation if history buffer not yet full (at the beginning of an episode).

        Args:
            observation (dict): Single-frame observation from _get_obs()

        Returns:
            dict: Stacked observation with shape (history_length, ...)
        """
        self.state_buffer.append(observation)
        stacked = {}
        for key in observation.keys():
            frames = list(self.state_buffer)
            if len(frames) < self.history_length:
                pad_len = self.history_length - len(frames)
                pad = [frames[0]] * pad_len
                frames = pad + frames
            stacked[key] = np.stack([f[key] for f in frames], axis=0)
        return stacked

    #############################################################
    # Episode termination
    #############################################################
    def _is_done(self) -> bool:
        """
        Checks whether the current episode should terminate.

        Conditions:
            1. Maximum episode steps reached
            2. Collision detected
            3. Environment exploration complete
            4. Odometry lost

        Returns:
            bool: True if any termination condition is met
        """
        if self.step_num >= self.episode_length:
            self.termination_reason = 'Time_Out'
            return True
        if self.check_collision():
            self.termination_reason = 'Collision_Occured'
            return True
        if self.check_completeness():
            self.termination_reason = 'Exploration_Done'
            return True
        odom = self.get_odom()
        if (odom.pose.covariance[0] == 9999.0):
            self.termination_reason = 'Odom_Lost'
            return True
        return False

    def check_collision(self) -> bool:
        """
        Checks whether the robot collided with an obstacle.

        Returns:
            bool: True if collision occurred
        """
        return self.collision

    def check_completeness(self) -> bool:
        """
        Computes explored area and determines if coverage threshold is met.

        Returns:
            bool: True if explored area exceeds map_area_threshold
        """
        global_OG_map = self.get_global_OG_map()
        occupied = np.sum(global_OG_map > 0.9)
        free = np.sum(global_OG_map < 0.1)
        resolution = self.get_map().info.resolution
        area = (occupied + free) * (resolution ** 2)
        return area >= self.map_area_threshold * self.total_map_area

    #############################################################
    # Reward
    #############################################################
    def _compute_reward(self) -> float:
        """
        Computes total reward for the current step based on:
            - Smoothed map gain
            - Smoothness of control (penalizes angular velocity)
            - Collision penalty

        Returns:
            float: Step reward
        """
        map_reward = self.smoothed_map_gain
        control_reward = self.calculate_control_reward()
        reward = (3.2 * map_reward +
                  0.2 * control_reward)
        if self.check_collision():
            reward = -30.0

        return reward

    def calculate_map_area(self) -> float:
        """
        Computes both the current explored area of the environment and the smoothed map area gain.

        Returns:
            float: Current map area
        """
        global_OG_map = self.get_global_OG_map()
        resolution = self.get_map().info.resolution
        self.last_map_area = self.map_area
        occupied = np.sum(global_OG_map > 0.9)
        free     = np.sum(global_OG_map < 0.1)
        self.map_area = (occupied + free) * (resolution ** 2)
        self.smoothed_map_gain = (0.6 * (self.map_area - self.last_map_area) +
                                  0.4 * self.last_smoothed_map_gain)
        self.last_smoothed_map_gain = self.smoothed_map_gain
        return self.map_area

    def calculate_control_reward(self) -> float:
        """
        Provides a small penalty for angular rotation to encourage smooth movement.

        Returns:
            float: Control reward
        """
        if self.last_action is None:
            return 0.0
        _, w = self.speeds[self.last_action]
        return -0.5 if w > 0 else 0.01

    #############################################################
    # Exploration metrics
    #############################################################
    
    def update_trajectory_length(self):
        """
        Updates the length of the robot trajectory after performing a step
        """
        odom, last_odom = self.get_odom(), self.last_odom
        self.last_odom = odom
        x_old, x_new = last_odom.pose.pose.position.x, odom.pose.pose.position.x
        y_old, y_new = last_odom.pose.pose.position.y, odom.pose.pose.position.y
        self.trajectory_length += np.sqrt(np.square(x_new - x_old) + np.square(y_new - y_old))
        
        
    def calculate_exploration_time(self):
        """
        Computes the time spent from the beginning of the episode until its end
        """
        self.exploration_time = rospy.Time.now().to_sec() - self.start_time
        return self.exploration_time
        
        
