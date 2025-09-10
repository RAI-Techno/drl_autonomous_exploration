import numpy as np
import rospy
import cv2
import random
from aslam.openai_ros import robot_gazebo_env
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ContactsState


class LilyBotEnv(robot_gazebo_env.RobotGazeboEnv):
    """
    Custom environment class for LilyBot in Gazebo using ROS.

        - Initializes subscribers and publishers.
        - Ensures sensors are ready to use.
        - Processes observations from robot's sensors (occupancy grid maps and laser scans).
        - Provides access to robot state and sensors' data.
        - Publishes motion commands
    """

    def __init__(self):
        rospy.logdebug("Initializing LilyBotEnv...")

        self.controllers_list = []
        self.robot_name_space = ""

        super(LilyBotEnv, self).__init__(
            controllers_list=self.controllers_list,
            robot_name_space=self.robot_name_space,
            reset_controls=False,
            start_init_physics_parameters=False
        )
        # Unpause simulation to initialize sensors
        self.gazebo.unpauseSim()
        self._check_all_sensors_ready()

        # Subscribers
        rospy.Subscriber("/rtabmap/rtab_odom", Odometry, self._odom_callback)     # Odometry's subscriber
        rospy.Subscriber("/rtabmap/grid_map", OccupancyGrid, self._map_callback)  # Occupancy grid's subscriber
        rospy.Subscriber("/scan", LaserScan, self._laser_scan_callback)     # LaserScan's subscriber


        for topic in ["/bumper_back", "/bumper_front", "/bumper_left", "/bumper_right"]:
            rospy.Subscriber(topic, ContactsState, self._bumper_callback)	     # Touch sensors' subscribers (to check collision)

        # Publisher
        self._cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)      # Motion commands' publisher
        self._check_publishers_connection()

        self.gazebo.pauseSim()
        rospy.logdebug("LilyBotEnv initialized.")
   
    ####### Check whether sensors are ready. #######
    def _check_all_systems_ready(self):
        self._check_all_sensors_ready()
        return True

    def _check_all_sensors_ready(self):
        """Checks if all required sensors are ready before starting the environment."""
        rospy.logdebug("Checking sensors...")
        self._wait_for_message("/rtabmap/rtab_odom", Odometry)
        self._wait_for_message("/rtabmap/grid_map", OccupancyGrid)
        self._wait_for_message("/scan", LaserScan)
        rospy.logdebug("All sensors ready.")

    def _wait_for_message(self, topic, msg_type, timeout=5.0):
        """Waits until data is available from the topic."""
        msg = None
        rospy.logdebug(f"Waiting for {topic}...")
        while msg is None and not rospy.is_shutdown():
            try:
                msg = rospy.wait_for_message(topic, msg_type, timeout=timeout)
                rospy.logdebug(f"{topic} ready.")
            except rospy.ROSException:
                rospy.logwarn(f"Timeout while waiting for {topic}, retrying...")
        return msg

    ####### Callbacks. #######
    def _odom_callback(self, data: Odometry):
        """Callback for odometry data."""
        self.odom = data

    def _bumper_callback(self, data: ContactsState):
        """Sets collision flag if any bumper is triggered."""
        if len(data.states) > 0:
            self.collision = True

    def _map_callback(self, data: OccupancyGrid):
        """Callback for occupancy grid data."""        
        self.map = data

    def _laser_scan_callback(self, data: LaserScan):
        """Callback for LaserScan data."""
        self.scan = data

    ####### Observations' processing. #######
    def process_sensors(self):
        """ Processes raw sensor's data before passing the current robot's state to RL."""
        self._process_map()
        self._process_scan()

    def _process_map(self):
        """ Extracts a local occupancy grid centered around the robot and rotated around its heading.  """
        data = self.get_map()
        pad = rospy.get_param("/OG_pad") 
        local_w = rospy.get_param("/local_OG_width")
        robot_pose = self.get_odom().pose.pose
	
        # Extrating information from raw data msg.
        xr, yr = robot_pose.position.x, robot_pose.position.y
        xo, yo = data.info.origin.position.x, data.info.origin.position.y
        w, h = data.info.width, data.info.height
        res = data.info.resolution
        theta = np.rad2deg(np.arctan2(robot_pose.orientation.z, robot_pose.orientation.w)) * 2
	
        og_map = np.array(data.data, dtype=np.int8).reshape(h, w)
        og_map[og_map == -1] = 50
        og_map = og_map.astype(np.float32) * (1 / 100.0)
        self.global_OG_map = og_map

        # Creating Local patch around the robot
        cx, cy = int(np.round((xr - xo) / res)), int(np.round((yr - yo) / res))
        radius = int(np.sqrt(2)*local_w) + 1
        xmin, xmax = max(0, cx - radius), min(w, cx + radius)
        ymin, ymax = max(0, cy - radius), min(h, cy + radius)

        patch = cv2.transpose(og_map[ymin:ymax, xmin:xmax])
        patch = cv2.copyMakeBorder(patch, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

        pcx, pcy = cx - xmin + pad, cy - ymin + pad
        rotation_matrix = cv2.getRotationMatrix2D((pcy, pcx), -theta, 1.0)
        rotated = cv2.warpAffine(
            patch, rotation_matrix, dsize=(patch.shape[1], patch.shape[0]),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE
        )
        
        # Changing resolution from 0.05 to 0.1 for faster processing
        half_w = local_w // 2
        big_roi = rotated[pcx - 2 * half_w:pcx + 2 * half_w, pcy - 2 * half_w:pcy + 2 * half_w]
        kernel = np.ones((2, 2), dtype=big_roi.dtype)
        dilated = cv2.dilate(big_roi, kernel)
        self.local_OG_map = dilated[::2, ::2]

    def _process_scan(self):
        data = self.scan
        laser_scan = np.array(data.ranges, dtype=np.float32)
        laser_scan = np.nan_to_num(laser_scan, nan=data.range_max)           #Removing nan values
        laser_scan = np.clip(laser_scan, data.range_min, data.range_max)     #clipping ranges
        self.laser_scan = laser_scan / data.range_max

    ####### Motion commands Publisher #######
    def _check_publishers_connection(self):
        rate = rospy.Rate(10)
        while self._cmd_vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("Waiting for /cmd_vel publisher connection...")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass
        rospy.logdebug("/cmd_vel publisher connected.")

    def move_base(self, linear_speed: float, angular_speed: float):
        """
        Publishes linear/angular velocity commands
        """
        controller_freq = rospy.get_param("/controller_freq")
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed
        rospy.logdebug(f"Publishing cmd_vel: {cmd_vel_value}")
        self._check_publishers_connection()
        self._cmd_vel_pub.publish(cmd_vel_value)
        rospy.sleep(1.0 / controller_freq)


    ####### Accessors #######
    def get_odom(self) -> Odometry:
        return self.odom

    def get_global_OG_map(self) -> np.ndarray:
        return self.global_OG_map

    def get_local_OG_map(self) -> np.ndarray:
        return self.local_OG_map

    def get_laser_scan(self) -> np.ndarray:
        return self.laser_scan

    def get_map(self) -> OccupancyGrid:
        return self.map


    ####### Abstract methods #######
    def _set_init_pose(self):
        raise NotImplementedError()

    def _init_env_variables(self):
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        raise NotImplementedError()

    def _set_action(self, action):
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        raise NotImplementedError()
