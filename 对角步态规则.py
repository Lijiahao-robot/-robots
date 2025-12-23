import pybullet as p
import pybullet_data
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# 自定义A1环境（继承PyBullet内置环境，增加步态约束）
class A1CustomEnv(pybullet_envs.bullet.ymal_bullet_env.AnymalBulletEnv):
    def __init__(self, render=False):
        super().__init__(
            urdfRoot=pybullet_data.getDataPath(),
            robot="a1",  # 指定A1机器人模型
            render=render,
            maxEpisodeSteps=500
        )
        # 对角步态关节索引（前左+后右为一组，前右+后左为一组）
        self.gait_joints = [[0,1,2, 6,7,8], [3,4,5, 9,10,11]]
        self.step_phase = 0  # 步态相位（0/1切换）

    def step(self, action):
        # 步态约束：按相位调整动作幅度，避免关节乱摆
        phase_action = np.zeros_like(action)
        for i, joint_group in enumerate(self.gait_joints):
            if (self.step_phase % 2) == i:
                phase_action[joint_group] = action[joint_group] * 0.8  # 发力组
            else:
                phase_action[joint_group] = action[joint_group] * 0.2  # 支撑组
        self.step_phase += 1
        # 调用父类step，传入约束后的动作
        obs, reward, terminated, truncated, info = super().step(phase_action)
        # 自定义奖励：强化前进速度，惩罚侧翻
        forward_vel = self.robot_base_vel[0]  # x轴前进速度
        roll, pitch, _ = p.getEulerFromQuaternion(self.robot_base_orn)
        reward += forward_vel * 2  # 前进奖励翻倍
        reward -= abs(roll) * 0.5  # 侧翻惩罚
        reward -= abs(pitch) * 0.3  # 俯仰惩罚
        # 摔倒判定（pitch/roll超过30度）
        if abs(roll) > np.pi/6 or abs(pitch) > np.pi/6:
            terminated = True
            reward -= 10
        return obs, reward, terminated, truncated, info

# 1. 创建自定义A1环境
def make_a1_env(render=False):
    return A1CustomEnv(render=render)

env = make_vec_env(
    make_a1_env,
    n_envs=4,
    env_kwargs={"render": False}
)

# 2. PPO模型（适配步态控制）
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=5e-4,
    n_steps=4096,
    batch_size=256,
    gamma=0.98,
    ent_coeff=0.01,  # 熵奖励，鼓励探索但不偏离步态
    verbose=1
)

# 3. 训练（带进度打印）
total_timesteps = 2_000_000
model.learn(
    total_timesteps=total_timesteps,
    callback=lambda locals, globals: print(f"进度：{locals['total_timesteps']/total_timesteps*100:.1f}%，当前奖励：{locals['rewards'][0]:.2f}")
)
model.save("a1_gait_ppo")

# 4. 测试（可视化精准行走）
test_env = A1CustomEnv(render=True)
obs, info = test_env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    if terminated or truncated:
        obs, info = test_env.reset()
test_env.close()
p.disconnect()
2. Isaac Gym 
# 需先安装Isaac Gym和IsaacGymEnvs
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.utils.config_utils import sim_config
import torch
import numpy as np

class A1ObstacleTask(VecTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        # 1. 加载A1机器人（批量创建4096个环境）
        self.num_envs = cfg["numEnvs"]
        self.robot_handles = self.create_robots(self.num_envs, "a1")
        # 2. 加载障碍物（随机圆柱，高度0.1~0.3m，间距0.5~1m）
        self.obstacle_handles = []
        for env_idx in range(self.num_envs):
            obstacle_pos = np.array([1.0 + env_idx%10*0.8, 0, 0.15])  # 沿x轴排列
            obstacle_scale = np.random.uniform(0.1, 0.3)
            handle = self.gym.create_actor(
                self.envs[env_idx],
                self.gym.create_box(self.sim, 0.2, 0.2, obstacle_scale),
                obstacle_pos,
                [0,0,0,1],
                f"obstacle_{env_idx}",
                self.gym.create_material(self.sim, 0.8, 0.8, 0.1)  # 高摩擦地面
            )
            self.obstacle_handles.append(handle)
        # 3. 初始化状态/动作空间
        self.obs_dim = 48  # A1：12关节角度+12速度+6姿态+3速度+15接触力
        self.act_dim = 12  # 12关节力矩
        self.obs = torch.zeros((self.num_envs, self.obs_dim), device=self.device)
        self.rew = torch.zeros(self.num_envs, device=self.device)

    def compute_reward(self):
        # 自定义越障奖励
        self.rew[:] = 0.0
        # 前进奖励：每前进0.1m+1分
        forward_vel = self.base_vel[:, 0]
        self.rew += forward_vel * 10.0
        # 越障奖励：穿过障碍物+5分
        for env_idx in range(self.num_envs):
            robot_pos = self.base_pos[env_idx]
            obstacle_pos = self.gym.get_actor_transform(self.envs[env_idx], self.obstacle_handles[env_idx]).p
            if robot_pos[0] > obstacle_pos[0] + 0.2:  # 超过障碍物
                self.rew[env_idx] += 5.0
        # 惩罚：碰撞障碍物-3分，摔倒-10分
        collision = self.check_collision(self.robot_handles, self.obstacle_handles)
        self.rew[collision] -= 3.0
        fall = torch.abs(self.base_rot[:, 0]) > 0.8  # 姿态异常判定
        self.rew[fall] -= 10.0
        # 能耗惩罚：动作幅度越大，惩罚越高
        self.rew -= torch.norm(self.actions, dim=1) * 0.01

    def step(self, actions):
        # 执行动作，更新状态
        self.actions = actions
        self.gym.set_actor_dof_actuation_force(self.sim, self.robot_handles, actions)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        # 更新观测（关节角度、速度、姿态、速度、接触力）
        self.update_obs()
        # 计算奖励
        self.compute_reward()
        # 判定终止
        self.reset_buf = torch.abs(self.base_rot[:, 0]) > 0.9  # 摔倒重置
        return self.obs, self.rew, self.reset_buf, {}

# 注册自定义任务并训练
from isaacgymenvs import make_env
from ray.rllib.algorithms.ppo import PPOConfig
import ray

ray.init()
# 创建越障环境
env = make_env(
    "A1Obstacle",
    rl_device="cuda:0",
    sim_device="cuda:0",
    headless=False,
    env_cfg={"numEnvs": 4096, "maxEpisodeLength": 1000}
)
# RLlib PPO配置
config = PPOConfig()\
    .environment(env="isaacgym", env_config={"env_name": "A1Obstacle"})\
    .framework("torch")\
    .training(lr=5e-4, gamma=0.99, train_batch_size=16384)\
    .resources(num_gpus=1)

algo = config.build()
# 训练50轮，每轮打印越障成功率
for epoch in range(50):
    result = algo.train()
    success_rate = result["custom_metrics"].get("obstacle_success_rate", 0)
    print(f"Epoch {epoch}: 平均奖励={result['episode_reward_mean']:.2f}，越障成功率={success_rate:.2f}")
algo.stop()
ray.shutdown()
3.TensorBoard
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import numpy as np

# 自定义回调：实时绘制奖励曲线
class RewardCallback(BaseCallback):
    def __init__(self, check_freq=1000, log_dir="./logs/"):
        super().__init__()
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.rewards = []
        self.timesteps = []
        plt.ion()  # 开启交互模式
        self.fig, self.ax = plt.subplots(figsize=(10, 5))

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            # 记录当前平均奖励
            ep_rewards = self.locals["infos"]["episode_reward"]
            self.rewards.append(np.mean(ep_rewards))
            self.timesteps.append(self.num_timesteps)
            # 实时绘图
            self.ax.clear()
            self.ax.plot(self.timesteps, self.rewards, label="Average Reward")
            self.ax.set_xlabel("Timesteps")
            self.ax.set_ylabel("Reward")
            self.ax.set_title("A1 Walking Training Reward")
            self.ax.legend()
            self.fig.savefig(f"{self.log_dir}/reward_curve.png")
            plt.pause(0.1)
        return True

# 集成到PPO训练中
callback = RewardCallback(check_freq=5000, log_dir="./a1_logs/")
model.learn(total_timesteps=2_000_000, callback=callback)

# TensorBoard监控（SB3内置）
# 启动命令：tensorboard --logdir ./a1_logs/
# 可查看：奖励曲线、损失函数、动作分布、关节角度直方图
4.Sim-to-Real
def randomize_phy_params(physics_client):
    # 随机化地面摩擦（0.3~1.0）
    ground_id = p.loadURDF("plane.urdf")
    p.changeDynamics(
        ground_id, -1,
        lateralFriction=np.random.uniform(0.3, 1.0),
        spinningFriction=np.random.uniform(0.01, 0.1)
    )
    # 随机化机器人关节阻尼（0.1~0.5）
    robot_id = p.loadURDF("a1.urdf", [0,0,0.5])
    for joint_idx in range(p.getNumJoints(robot_id)):
        p.changeDynamics(
            robot_id, joint_idx,
            jointDamping=np.random.uniform(0.1, 0.5),
            mass=np.random.uniform(0.8, 1.2) * p.getDynamicsInfo(robot_id, joint_idx)[0]  # 质量±20%
        )
    # 随机化重力（9.8~10.2 m/s²）
    p.setGravity(0, 0, -np.random.uniform(9.8, 10.2))
    return robot_id

# 在环境reset时调用域随机化
class A1RandEnv(A1CustomEnv):
    def reset(self, seed=None, options=None):
        self.physics_client = p.connect(p.GUI if self.render else p.DIRECT)
        self.robot_id = randomize_phy_params(self.physics_client)
        return super().reset(seed, options)

# 训练带域随机化的环境
env = make_vec_env(A1RandEnv, n_envs=4)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=2_000_000)
