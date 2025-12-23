# 在SpotTask的reset_idx中添加随机化
def reset_idx(self, env_ids):
    for env_id in env_ids:
        # 随机化地面摩擦（Spot真实场景摩擦0.4~1.2）
        ground_mat = self.gym.create_material(self.sim, np.random.uniform(0.4, 1.2), np.random.uniform(0.4, 1.2), 0.1)
        self.gym.set_actor_material(self.envs[env_id], self.ground_id, ground_mat)
        
        # 随机化液压阻尼（±30%）
        dof_props = self.gym.get_actor_dof_properties(self.envs[env_id], self.spot_handles[env_id])
        dof_props["damping"][:] *= np.random.uniform(0.7, 1.3)
        self.gym.set_actor_dof_properties(self.envs[env_id], self.spot_handles[env_id], dof_props)
        
        # 随机化负载（Spot可载重14kg）
        payload_mass = np.random.uniform(0, 14)
        self.gym.set_actor_mass(self.envs[env_id], self.spot_handles[env_id], self.gym.get_actor_mass(self.envs[env_id], self.spot_handles[env_id]) + payload_mass)
     
  2.ROS2 + A1/Spot
# ROS2发布控制指令（A1为例）
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

class A1RLNode(Node):
    def __init__(self):
        super().__init__("a1_rl_controller")
        self.pub = self.create_publisher(Float32MultiArray, "/a1/joint_command", 10)
        self.sub = self.create_subscription(JointState, "/a1/joint_states", self.joint_state_cb, 10)
        self.model = PPO.load("a1_walk_ppo_final")  # 加载训练好的模型
        self.current_obs = None

    def joint_state_cb(self, msg):
        # 从ROS话题获取关节状态，构建观测
        joint_angles = msg.position
        joint_vels = msg.velocity
        base_orn = msg.header.frame_id  # 替换为实际基座姿态话题
        base_vel = msg.header.stamp  # 替换为实际基座速度话题
        self.current_obs = np.concatenate([joint_angles, joint_vels, base_orn, base_vel])
        
        # 预测动作并发布
        action, _ = self.model.predict(self.current_obs, deterministic=True)
        cmd = Float32MultiArray()
        cmd.data = action.tolist()
        self.pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = A1RLNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
