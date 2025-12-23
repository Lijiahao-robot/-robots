# 安装ROS2（Humble）+ RLlib
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-ackermann-msgs
pip install ray[rllib]==2.6.0 torch==2.0.1 gymnasium
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import PettingZooEnv
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

# ROS2节点：连接RL算法与阿克曼机器人
class AckermannRLNode(Node):
    def __init__(self):
        super().__init__("ackermann_rl_node")
        # 1. ROS话题订阅/发布
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_cb, 10)
        self.path_sub = self.create_subscription(PoseStamped, "/target_path", self.path_cb, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        
        # 2. 状态缓存
        self.current_odom = None
        self.target_path = None
        self.obs_dim = 9  # 位姿+速度+偏航+路径偏差+航向偏差+前视距离
        self.act_dim = 2  # 线速度+转向角

    def odom_cb(self, msg):
        self.current_odom = msg

    def path_cb(self, msg):
        self.target_path = msg

    def get_obs(self):
        # 构建观测：机器人状态 + 路径偏差
        if self.current_odom is None or self.target_path is None:
            return np.zeros(self.obs_dim)
        # 机器人状态
        x = self.current_odom.pose.pose.position.x
        y = self.current_odom.pose.pose.position.y
        yaw = 2 * np.arctan2(self.current_odom.pose.pose.orientation.z, self.current_odom.pose.pose.orientation.w)
        vx = self.current_odom.twist.twist.linear.x
        vy = self.current_odom.twist.twist.linear.y
        # 路径偏差
        path_x = self.target_path.pose.position.x
        path_y = self.target_path.pose.position.y
        path_yaw = 2 * np.arctan2(self.target_path.pose.orientation.z, self.target_path.pose.orientation.w)
        path_error = np.linalg.norm([x-path_x, y-path_y])
        heading_error = yaw - path_yaw
        # 前视距离
        lookahead_dist = 1.0  # 阿克曼前视距离
        return np.array([x, y, yaw, vx, vy, path_error, heading_error, lookahead_dist, 0.0])

    def send_action(self, action):
        # 发布阿克曼控制指令
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = np.clip(action[0], 0.0, 2.0)  # 线速度0~2m/s
        drive_msg.drive.steering_angle = np.clip(action[1], -np.pi/4, np.pi/4)  # 转向角±45度
        self.drive_pub.publish(drive_msg)

# 自定义阿克曼环境（对接ROS）
class AckermannEnv:
    def __init__(self):
        rclpy.init()
        self.node = AckermannRLNode()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.node.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.node.act_dim,), dtype=np.float32)
        self.max_episode_steps = 1000
        self.current_step = 0

    def reset(self, seed=None, options=None):
        self.current_step = 0
        return self.node.get_obs(), {}

    def step(self, action):
        # 动作缩放：线速度0~2m/s，转向角±45度
        scaled_action = [
            (action[0] + 1) / 2 * 2.0,  # -1→0m/s，1→2m/s
            action[1] * np.pi/4  # -1→-45度，1→45度
        ]
        # 发布控制指令
        self.node.send_action(scaled_action)
        # 等待ROS回调
        rclpy.spin_once(self.node, timeout_sec=0.01)
        # 获取新观测
        obs = self.node.get_obs()
        # 计算奖励
        reward = self._compute_reward(obs)
        # 终止条件
        self.current_step += 1
        terminated = (obs[5] < 0.1) or (self.current_step >= self.max_episode_steps)  # 路径偏差<0.1m
        truncated = False
        info = {"path_error": obs[5]}
        return obs, reward, terminated, truncated, info

    def _compute_reward(self, obs):
        reward = 0.0
        path_error = obs[5]
        heading_error = obs[6]
        speed = obs[3]
        # 1. 路径跟踪奖励：偏差越小，奖励越高
        reward += 8 * np.exp(-path_error / 0.2)
        # 2. 航向奖励：航向偏差小加分
        reward += 4 * np.exp(-abs(heading_error) / 0.1)
        # 3. 速度奖励：保持稳定速度
        reward += 1 * np.exp(-(speed - 1.0)**2 / 0.5)
        # 4. 惩罚：转向角过大
        reward -= 2 * abs(obs[7])
        return reward

    def close(self):
        rclpy.shutdown()

# 初始化Ray + 训练
ray.init(ignore_reinit_error=True)
config = (
    PPOConfig()
    .environment(AckermannEnv)
    .framework("torch")
    .rollouts(num_rollout_workers=1)
    .training(
        lr=3e-4,
        gamma=0.98,
        train_batch_size=4096,
        clip_range=0.2
    )
)

algo = config.build()
# 训练50轮
for epoch in range(50):
    result = algo.train()
    print(f"Epoch {epoch}: 平均奖励={result['episode_reward_mean']:.2f}，路径偏差={result['custom_metrics']['path_error_mean']:.2f}m")
    if epoch % 10 == 0:
        algo.save(f"./ackermann_ppo_epoch_{epoch}")

algo.stop()
ray.shutdown()
