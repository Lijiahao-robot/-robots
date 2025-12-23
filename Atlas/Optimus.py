# ROS2 + 实时控制接口
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

class AtlasRLDeployNode(Node):
    def __init__(self, model_path):
        super().__init__("atlas_rl_deploy")
        # 1. 加载训练好的RL模型
        self.model = PPO.load(model_path)
        # 2. ROS订阅/发布
        self.joint_sub = self.create_subscription(JointState, "/atlas/joint_states", self.joint_cb, 10)
        self.ctrl_pub = self.create_publisher(Float32MultiArray, "/atlas/joint_commands", 10)
        self.current_obs = None

    def joint_cb(self, msg):
        """接收真实Atlas关节状态，构建观测"""
        # 拼接与仿真一致的观测向量
        joint_pos = np.array(msg.position)
        joint_vel = np.array(msg.velocity)
        # 补充IMU/接触力/质心状态（从真实传感器获取）
        imu_data = np.array([0.0]*6)  # 替换为真实IMU数据
        contact_force = np.array([0.0]*12)  # 替换为真实接触力数据
        self.current_obs = np.concatenate([joint_pos, joint_vel, imu_data, contact_force, [2.0,0.0,1.0,1.0,0.0]])

    def run(self):
        """实时预测并发布关节指令"""
        rate = self.create_rate(100)  # 100Hz控制频率
        while rclpy.ok():
            if self.current_obs is not None:
                action, _ = self.model.predict(self.current_obs, deterministic=True)
                cmd = Float32MultiArray()
                cmd.data = action.tolist()
                self.ctrl_pub.publish(cmd)
            rate.sleep()

# 启动部署节点
rclpy.init()
node = AtlasRLDeployNode("./atlas_ppo_epoch_500")
node.run()
#人形机器人必备
def safety_check(self, action, obs):
    """安全校验，避免危险动作"""
    # 1. 关节限位检查
    joint_pos = obs[:, :28]
    joint_limit_low = -np.pi/2
    joint_limit_high = np.pi/2
    unsafe_joint = (joint_pos < joint_limit_low) | (joint_pos > joint_limit_high)
    action[unsafe_joint] = 0.0  # 超限关节力矩置0
    
    # 2. 摔倒预判（质心偏移>0.2m）
    com_pos = obs[:, 62:65]
    com_offset = np.abs(com_pos[:, 0] - com_pos[:, 1])
    if com_offset > 0.2:
        action = np.zeros_like(action)  # 紧急制动
    
    # 3. 力矩超限检查（>150N·m）
    action = np.clip(action, -150.0, 150.0)
    return action
