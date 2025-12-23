# 克隆A1官方模型仓库
git clone https://github.com/unitreerobotics/unitree_ros.git
# 模型路径：unitree_ros/unitree_description/urdf/a1/
import pybullet as p
import pybullet_data
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

# 自定义A1环境（适配官方URDF）
class A1Env:
    def __init__(self, render=False):
        self.render = render
        self.physics_client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1/240)  # A1官方控制频率240Hz

        # 1. 加载地面和A1机器人
        self.ground_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(
            "unitree_ros/unitree_description/urdf/a1/a1.urdf",  # 替换为你的A1 URDF路径
            [0, 0, 0.4],  # A1初始高度
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=False,
            flags=p.URDF_USE_INERTIA_FROM_FILE  # 加载官方惯性参数
        )

        # 2. A1关节配置（12个可控关节，对应4条腿×3关节）
        self.joint_ids = []
        self.joint_limits = []
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            if joint_info[2] != p.JOINT_FIXED:  # 跳过固定关节
                self.joint_ids.append(i)
                self.joint_limits.append([joint_info[8], joint_info[9]])  # 关节限位
        self.num_joints = len(self.joint_ids)

        # 3. 状态/动作空间定义
        self.obs_dim = self.num_joints*2 + 6 + 3  # 关节角度+速度 + 身体姿态 + 基座速度
        self.act_dim = self.num_joints  # 关节力矩控制
        self.max_episode_steps = 500
        self.current_step = 0

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(
            "unitree_ros/unitree_description/urdf/a1/a1.urdf",
            [0, 0, 0.4],
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=False,
            flags=p.URDF_USE_INERTIA_FROM_FILE
        )
        self.current_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        # 获取状态：关节角度+速度 + 基座姿态+速度
        joint_states = p.getJointStates(self.robot_id, self.joint_ids)
        joint_angles = [state[0] for state in joint_states]
        joint_vels = [state[1] for state in joint_states]
        
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        base_euler = p.getEulerFromQuaternion(base_orn)
        base_vel, base_ang_vel = p.getBaseVelocity(self.robot_id)

        obs = np.concatenate([joint_angles, joint_vels, base_euler, base_vel[:3]])
        return obs.astype(np.float32)

    def _compute_reward(self):
        # A1专属奖励函数
        reward = 0.0
        base_vel, _ = p.getBaseVelocity(self.robot_id)
        forward_vel = base_vel[0]  # x轴前进速度（A1目标：1.5m/s）
        base_euler = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.robot_id)[1])
        roll, pitch, _ = base_euler

        # 1. 前进奖励（核心）：速度越接近1.5m/s，奖励越高
        reward += 10 * np.exp(-(forward_vel - 1.5)**2 / 0.5)  # 高斯奖励
        
        # 2. 平衡奖励：roll/pitch越小越好
        reward -= 5 * (abs(roll) + abs(pitch))
        
        # 3. 关节限位奖励：避免关节超出安全范围
        joint_states = p.getJointStates(self.robot_id, self.joint_ids)
        for i, (angle, limit) in enumerate(zip([s[0] for s in joint_states], self.joint_limits)):
            if limit[0] < angle < limit[1]:
                reward += 0.1
            else:
                reward -= 1.0
        
        # 4. 能耗惩罚：动作幅度越小越好
        current_force = [abs(p.getJointState(self.robot_id, i)[3]) for i in self.joint_ids]
        reward -= 0.01 * sum(current_force)
        
        # 5. 摔倒惩罚：pitch/roll超过30度
        if abs(roll) > np.pi/6 or abs(pitch) > np.pi/6:
            reward -= 50
        
        return reward

    def step(self, action):
        # 动作裁剪：限制力矩范围（A1关节最大力矩：33N·m）
        action = np.clip(action, -33, 33)
        
        # 执行力矩控制
        for i, joint_id in enumerate(self.joint_ids):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_id,
                controlMode=p.TORQUE_CONTROL,
                force=action[i]
            )
        
        p.stepSimulation()
        self.current_step += 1
        
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = (self.current_step >= self.max_episode_steps) or (abs(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.robot_id)[1])[1]) > np.pi/3)
        truncated = False
        info = {"forward_vel": p.getBaseVelocity(self.robot_id)[0][0]}
        
        return obs, reward, terminated, truncated, info

    def close(self):
        p.disconnect(self.physics_client)

# 注册环境
def make_a1_env(render=False):
    return A1Env(render=render)

# 1. 创建向量环境
env = make_vec_env(make_a1_env, n_envs=4, env_kwargs={"render": False})

# 2. 配置PPO（适配A1的高动态特性）
checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path="./a1_checkpoints/",
    name_prefix="a1_ppo"
)

model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=4096,
    batch_size=256,
    gamma=0.98,
    ent_coeff=0.02,  # 更高的熵系数，适配A1高速探索
    clip_range=0.2,
    verbose=1,
    tensorboard_log="./a1_tensorboard/"
)

# 3. 训练（建议至少200万步）
model.learn(
    total_timesteps=2_000_000,
    callback=checkpoint_callback
)
model.save("a1_walk_ppo_final")

# 4. 测试（可视化A1行走）
test_env = A1Env(render=True)
obs, info = test_env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    print(f"前进速度：{info['forward_vel']:.2f} m/s，奖励：{reward:.2f}")
    if terminated or truncated:
        obs, info = test_env.reset()
test_env.close()
