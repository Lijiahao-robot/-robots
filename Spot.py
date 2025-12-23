# 下载Spot的URDF模型，转换为USD格式（Isaac Gym要求）
git clone https://github.com/bdaiinstitute/spot_ros.git
# 使用Isaac Gym的urdf2usd工具转换
python /path/to/isaacgym/tools/urdf2usd.py spot_ros/spot_description/urdf/spot.urdf spot.usd
# 需安装Isaac Gym + IsaacGymEnvs + Ray RLlib
import torch
import numpy as np
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.utils.config_utils import sim_config
from isaacgym import gymapi, gymutil
import ray
from ray.rllib.algorithms.ppo import PPOConfig

# 自定义Spot任务类
class SpotTask(VecTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.cfg = cfg
        self.num_envs = cfg["numEnvs"]
        self.max_episode_length = cfg["maxEpisodeLength"]
        self.device = torch.device(f"{device_type}:{device_id}")

        # 1. 初始化仿真
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)

        # 2. 加载Spot USD模型
        spot_asset_file = "spot.usd"  # 替换为你的Spot USD路径
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.armature = 0.01  # Spot液压关节阻尼
        asset_options.use_mesh_materials = True
        spot_asset = self.gym.load_asset(self.sim, "", spot_asset_file, asset_options)

        # 3. 创建批量环境（4096个Spot并行）
        self.env_spacing = 2.0
        env_lower = gymapi.Vec3(-self.env_spacing, -self.env_spacing, 0.0)
        env_upper = gymapi.Vec3(self.env_spacing, self.env_spacing, self.env_spacing)
        self.envs = []
        self.spot_handles = []
        for i in range(self.num_envs):
            # 创建环境
            env = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            self.envs.append(env)
            # 放置Spot（随机初始位置）
            pos = gymapi.Vec3(
                np.random.uniform(-0.5, 0.5),
                np.random.uniform(-0.5, 0.5),
                0.6  # Spot初始高度
            )
            rot = gymapi.Quat(0, 0, 0, 1)
            spot_handle = self.gym.create_actor(env, spot_asset, pos, rot, "spot", i, 1, 0)
            self.spot_handles.append(spot_handle)
            # 设置Spot物理参数（液压驱动）
            dof_props = self.gym.get_actor_dof_properties(env, spot_handle)
            dof_props["driveMode"][:] = gymapi.DOF_MODE_EFFORT  # 力矩控制
            dof_props["effort"][:] = 100  # Spot液压最大力矩100N·m
            dof_props["damping"][:] = 5.0  # 液压阻尼
            self.gym.set_actor_dof_properties(env, spot_handle, dof_props)

        # 4. 状态/动作空间
        self.num_dofs = self.gym.get_actor_dof_count(self.envs[0], self.spot_handles[0])  # Spot：12个DOF
        self.obs_dim = self.num_dofs*2 + 6 + 3  # 关节角度+速度 + 基座姿态+速度
        self.act_dim = self.num_dofs
        self.obs = torch.zeros((self.num_envs, self.obs_dim), device=self.device)
        self.rew = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)

    def reset_idx(self, env_ids):
        # 重置指定环境的Spot
        for env_id in env_ids:
            # 随机初始姿态
            pos = gymapi.Vec3(
                np.random.uniform(-0.5, 0.5),
                np.random.uniform(-0.5, 0.5),
                0.6
            )
            rot = gymapi.Quat(0, 0, 0, 1)
            self.gym.set_actor_root_state(self.envs[env_id], self.spot_handles[env_id], pos, rot)
            # 重置关节
            dof_states = self.gym.get_actor_dof_states(self.envs[env_id], self.spot_handles[env_id], gymapi.STATE_ALL)
            dof_states["pos"][:] = 0.0
            dof_states["vel"][:] = 0.0
            self.gym.set_actor_dof_states(self.envs[env_id], self.spot_handles[env_id], dof_states, gymapi.STATE_ALL)
        self.reset_buf[env_ids] = False
        self.progress_buf[env_ids] = 0

    def compute_observations(self):
        # 获取批量Spot的状态
        dof_states = self.gym.get_actor_dof_states(self.envs[0], self.spot_handles[0], gymapi.STATE_ALL)
        dof_pos = torch.from_numpy(dof_states["pos"]).to(self.device).repeat(self.num_envs, 1)
        dof_vel = torch.from_numpy(dof_states["vel"]).to(self.device).repeat(self.num_envs, 1)
        
        # 基座姿态和速度（批量获取）
        root_states = self.gym.get_actor_root_states(self.envs[0], self.spot_handles[0])
        root_pos = torch.from_numpy(root_states["pos"]).to(self.device).repeat(self.num_envs, 1)
        root_rot = torch.from_numpy(root_states["rot"]).to(self.device).repeat(self.num_envs, 1)
        root_vel = torch.from_numpy(root_states["vel"]).to(self.device).repeat(self.num_envs, 1)

        # 拼接观测
        self.obs = torch.cat([dof_pos, dof_vel, root_rot, root_vel[:, :3]], dim=1)
        return self.obs

    def compute_reward(self):
        # Spot专属奖励（全地形越障）
        self.rew[:] = 0.0
        root_states = self.gym.get_actor_root_states(self.envs[0], self.spot_handles[0])
        root_vel = torch.from_numpy(root_states["vel"]).to(self.device).repeat(self.num_envs, 1)
        root_rot = torch.from_numpy(root_states["rot"]).to(self.device).repeat(self.num_envs, 1)

        # 1. 前进奖励（Spot目标：1.0m/s，重载场景）
        forward_vel = root_vel[:, 0]
        self.rew += 8 * torch.exp(-(forward_vel - 1.0)**2 / 0.3)

        # 2. 爬坡奖励（Spot核心能力）
        terrain_height = self.gym.get_terrain_height(self.envs[0], root_states["pos"][0], root_states["pos"][1])
        slope = torch.tensor(terrain_height, device=self.device).repeat(self.num_envs)
        self.rew += 5 * slope  # 爬坡越高，奖励越高

        # 3. 平衡奖励（Spot重载需更高稳定性）
        roll = torch.atan2(2*(root_rot[:, 0]*root_rot[:, 1] + root_rot[:, 2]*root_rot[:, 3]), 1 - 2*(root_rot[:, 1]**2 + root_rot[:, 2]**2))
        pitch = torch.asin(2*(root_rot[:, 0]*root_rot[:, 2] - root_rot[:, 3]*root_rot[:, 1]))
        self.rew -= 10 * (torch.abs(roll) + torch.abs(pitch))

        # 4. 越障奖励（Spot全地形特性）
        obstacle_pos = torch.tensor([1.0, 0.0, 0.3], device=self.device).repeat(self.num_envs, 1)
        dist_to_obstacle = torch.norm(root_pos[:, :2] - obstacle_pos[:, :2], dim=1)
        self.rew += 6 * (dist_to_obstacle < 0.2).float()  # 穿过障碍物加分

        # 5. 摔倒惩罚（Spot摔倒成本高，惩罚更重）
        self.rew[torch.abs(pitch) > 0.5] -= 100
        self.rew[torch.abs(roll) > 0.5] -= 100

        # 6. 能耗惩罚（液压系统能耗）
        actions = self.actions
        self.rew -= 0.02 * torch.norm(actions, dim=1)

    def step(self, actions):
        # 执行动作（液压力矩映射）
        self.actions = actions
        self.gym.set_actor_dof_actuation_force(self.sim, self.spot_handles, actions.cpu().numpy())
        
        # 仿真步
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # 更新进度和重置
        self.progress_buf += 1
        self.reset_buf = self.progress_buf >= self.max_episode_length

        # 计算观测和奖励
        self.compute_observations()
        self.compute_reward()

        # 重置需要重置的环境
        if torch.any(self.reset_buf):
            env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            self.reset_idx(env_ids)

        return self.obs, self.rew, self.reset_buf, {}

# 注册Spot任务并训练
def register_spot_task():
    from isaacgymenvs.utils.register import register_task
    register_task("Spot", SpotTask)

# 初始化Ray + 训练
ray.init(ignore_reinit_error=True)
register_spot_task()

# RLlib PPO配置（适配Spot的高维液压控制）
config = (
    PPOConfig()
    .environment(
        env="isaacgym",
        env_config={
            "env_name": "Spot",
            "numEnvs": 4096,
            "maxEpisodeLength": 1000
        }
    )
    .framework("torch")
    .rollouts(num_rollout_workers=0)
    .training(
        lr=4e-4,
        gamma=0.99,
        train_batch_size=32768,
        sgd_minibatch_size=8192,
        clip_range=0.25,
        vf_loss_coeff=0.6,  # 价值损失权重更高，适配Spot稳定性需求
        ent_coeff=0.015
    )
    .resources(num_gpus=1)
    .evaluation(evaluation_num_workers=1)
)

# 构建算法并训练
algo = config.build()
for epoch in range(60):
    result = algo.train()
    print(f"Epoch {epoch}: 平均奖励={result['episode_reward_mean']:.2f}，前进速度={result['custom_metrics']['forward_vel_mean']:.2f} m/s")
    # 每10轮保存模型
    if epoch % 10 == 0:
        algo.save(f"./spot_ppo_epoch_{epoch}")

# 评估Spot越障能力
eval_result = algo.evaluate()
print(f"Spot越障成功率：{eval_result['evaluation']['custom_metrics']['obstacle_success_rate']:.2f}")

algo.stop()
ray.shutdown()
