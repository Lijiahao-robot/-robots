# 1. 下载Atlas模型（转换为USD格式）
git clone https://github.com/bdaiinstitute/atlas_ros.git
python /path/to/isaacgym/tools/urdf2usd.py atlas_ros/atlas_description/urdf/atlas.urdf atlas.usd

# 2. 安装依赖（Isaac Gym + RLlib + 高性能计算库）
pip install ray[rllib]==2.6.0 torch==2.1.0 triton==2.1.0 isaacgym==2022.2
import torch
import numpy as np
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.utils.config_utils import sim_config
from isaacgym import gymapi, gymutil
import ray
from ray.rllib.algorithms.ppo import PPOConfig

# 自定义Atlas任务类（28DoF液压驱动）
class AtlasTask(VecTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        self.num_envs = cfg["numEnvs"]  # 4096并行环境（GPU加速）
        self.max_episode_length = cfg["maxEpisodeLength"]  # 1000步/episode
        self.device = torch.device(f"{device_type}:{device_id}")

        # 1. 加载Atlas USD模型
        atlas_asset_file = "atlas.usd"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.armature = 0.01  # 液压关节阻尼
        asset_options.use_mesh_materials = True
        self.atlas_asset = self.gym.load_asset(self.sim, "", atlas_asset_file, asset_options)

        # 2. 创建批量环境（带楼梯地形）
        self.envs = []
        self.atlas_handles = []
        env_lower = gymapi.Vec3(-5.0, -5.0, 0.0)
        env_upper = gymapi.Vec3(5.0, 5.0, 5.0)
        for i in range(self.num_envs):
            # 创建环境+楼梯地形（随机高度0.1~0.2m，步数5~8）
            env = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            self.envs.append(env)
            # 生成楼梯
            stair_height = np.random.uniform(0.1, 0.2)
            stair_steps = np.random.randint(5, 8)
            for step in range(stair_steps):
                stair_pos = gymapi.Vec3(step*0.3, 0.0, step*stair_height)
                stair_rot = gymapi.Quat(0, 0, 0, 1)
                stair_handle = self.gym.create_actor(
                    env,
                    self.gym.create_box(self.sim, 0.3, 1.0, stair_height),
                    stair_pos, stair_rot,
                    f"stair_{i}_{step}", i, 1, 0
                )
            # 加载Atlas（初始位置在楼梯前）
            atlas_pos = gymapi.Vec3(-1.0, 0.0, 1.0)  # Atlas身高~1.8m
            atlas_rot = gymapi.Quat(0, 0, 0, 1)
            atlas_handle = self.gym.create_actor(env, self.atlas_asset, atlas_pos, atlas_rot, "atlas", i, 1, 0)
            self.atlas_handles.append(atlas_handle)
            # 配置Atlas液压关节（最大力矩150N·m，阻尼5.0）
            dof_props = self.gym.get_actor_dof_properties(env, atlas_handle)
            dof_props["driveMode"][:] = gymapi.DOF_MODE_EFFORT  # 力矩控制
            dof_props["effort"][:] = 150.0
            dof_props["damping"][:] = 5.0
            self.gym.set_actor_dof_properties(env, atlas_handle, dof_props)

        # 3. 状态/动作空间定义（Atlas 28DoF）
        self.num_dofs = 28
        # 状态维度：28关节角度 + 28关节速度 + 6质心姿态 + 3质心速度 + 12接触力 + 5任务目标
        self.obs_dim = 28+28+6+3+12+5 = 82
        self.act_dim = 28  # 28关节力矩
        # 初始化张量（GPU批量计算）
        self.obs = torch.zeros((self.num_envs, self.obs_dim), device=self.device)
        self.rew = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)

    def reset_idx(self, env_ids):
        """重置指定环境的Atlas"""
        for env_id in env_ids:
            # 随机初始位置+姿态
            atlas_pos = gymapi.Vec3(-1.0 + np.random.uniform(-0.5, 0.5), 0.0, 1.0)
            atlas_rot = gymapi.Quat(0, 0, 0, 1)
            self.gym.set_actor_root_state(self.envs[env_id], self.atlas_handles[env_id], atlas_pos, atlas_rot)
            # 重置关节状态
            dof_states = self.gym.get_actor_dof_states(self.envs[env_id], self.atlas_handles[env_id], gymapi.STATE_ALL)
            dof_states["pos"][:] = 0.0  # 初始关节角度
            dof_states["vel"][:] = 0.0
            self.gym.set_actor_dof_states(self.envs[env_id], self.atlas_handles[env_id], dof_states, gymapi.STATE_ALL)
        self.reset_buf[env_ids] = False
        self.progress_buf[env_ids] = 0

    def compute_observations(self):
        """批量计算Atlas状态（GPU加速）"""
        # 1. 关节状态（角度+速度）
        dof_states = self.gym.get_actor_dof_states(self.envs[0], self.atlas_handles[0], gymapi.STATE_ALL)
        dof_pos = torch.from_numpy(dof_states["pos"]).to(self.device).repeat(self.num_envs, 1)
        dof_vel = torch.from_numpy(dof_states["vel"]).to(self.device).repeat(self.num_envs, 1)
        
        # 2. 质心（CoM）状态
        com_pos, com_rot = self.gym.get_actor_com_position(self.envs[0], self.atlas_handles[0])
        com_vel = self.gym.get_actor_com_velocity(self.envs[0], self.atlas_handles[0])
        com_pos = torch.from_numpy(com_pos).to(self.device).repeat(self.num_envs, 1)
        com_rot = torch.from_numpy(com_rot).to(self.device).repeat(self.num_envs, 1)
        com_vel = torch.from_numpy(com_vel).to(self.device).repeat(self.num_envs, 1)
        
        # 3. 接触力（脚与地面）
        contact_forces = self.gym.get_actor_contact_forces(self.envs[0], self.atlas_handles[0])
        contact_forces = torch.from_numpy(contact_forces).to(self.device).repeat(self.num_envs, 1)[:, :12]
        
        # 4. 任务目标（楼梯终点位置+目标速度）
        target_pos = torch.tensor([2.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
        target_vel = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        task_goal = torch.cat([target_pos, target_vel[:, :2]], dim=1)
        
        # 拼接观测
        self.obs = torch.cat([
            dof_pos, dof_vel, com_rot, com_vel, contact_forces, task_goal
        ], dim=1)
        return self.obs

    def compute_reward(self):
        """Atlas专属奖励函数（跑跳+上下楼梯）"""
        self.rew[:] = 0.0
        # 1. 质心（CoM）平衡奖励（核心：避免摔倒）
        com_rot = self.obs[:, 56:62]  # 质心姿态（roll/pitch/yaw）
        roll = com_rot[:, 0]
        pitch = com_rot[:, 1]
        self.rew -= 20 * (torch.abs(roll) + torch.abs(pitch))  # 平衡惩罚权重拉满
        # 摔倒判定（pitch/roll>45度）
        fall_mask = (torch.abs(roll) > np.pi/4) | (torch.abs(pitch) > np.pi/4)
        self.rew[fall_mask] -= 200  # 摔倒重罚

        # 2. 前进/爬楼梯奖励
        com_pos = self.obs[:, 62:65]  # 质心位置
        target_pos = self.obs[:, 77:80]  # 楼梯终点
        dist_to_target = torch.norm(com_pos - target_pos, dim=1)
        self.rew += 30 * torch.exp(-dist_to_target / 0.5)  # 指数奖励，靠近终点加分

        # 3. 速度奖励（跑跳目标：1.5m/s）
        com_vel = self.obs[:, 65:68]  # 质心速度
        forward_vel = com_vel[:, 0]
        self.rew += 15 * torch.exp(-(forward_vel - 1.5)**2 / 0.3)  # 速度越接近1.5m/s，奖励越高

        # 4. 脚接触奖励（单脚/双脚着地，避免浮空）
        contact_forces = self.obs[:, 68:80]  # 接触力
        foot_contact = torch.norm(contact_forces, dim=1) > 10.0  # 接触力>10N视为着地
        self.rew[foot_contact] += 5.0  # 着地加分

        # 5. 关节限位奖励（避免液压关节超限）
        dof_pos = self.obs[:, :28]
        dof_limit_low = torch.tensor([-np.pi/2]*28, device=self.device)
        dof_limit_high = torch.tensor([np.pi/2]*28, device=self.device)
        in_limit = (dof_pos > dof_limit_low) & (dof_pos < dof_limit_high)
        self.rew += 2.0 * torch.sum(in_limit, dim=1) / 28  # 平均限位奖励

        # 6. 能耗惩罚（液压系统能耗）
        actions = self.actions
        self.rew -= 0.05 * torch.norm(actions, dim=1)  # 动作幅度越大，惩罚越高

        # 7. 爬楼梯奖励（额外加分）
        stair_height = com_pos[:, 2] - 1.0  # 初始高度1.0m
        self.rew += 10 * stair_height  # 爬得越高，奖励越高

    def step(self, actions):
        """执行动作+更新状态"""
        # 动作裁剪：限制液压力矩（-150~150N·m）
        self.actions = torch.clamp(actions, -150.0, 150.0)
        # 批量设置关节力矩（GPU并行）
        self.gym.set_actor_dof_actuation_force(self.sim, self.atlas_handles, self.actions.cpu().numpy())
        
        # 仿真步（Atlas控制频率100Hz）
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # 更新进度和重置
        self.progress_buf += 1
        self.reset_buf = self.progress_buf >= self.max_episode_length

        # 计算观测和奖励
        self.compute_observations()
        self.compute_reward()

        # 重置摔倒/步数耗尽的环境
        fall_mask = (torch.abs(self.obs[:, 56]) > np.pi/4) | (torch.abs(self.obs[:, 57]) > np.pi/4)
        self.reset_buf = self.reset_buf | fall_mask
        if torch.any(self.reset_buf):
            env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            self.reset_idx(env_ids)

        return self.obs, self.rew, self.reset_buf, {}

# 注册Atlas任务
def register_atlas_task():
    from isaacgymenvs.utils.register import register_task
    register_task("Atlas", AtlasTask)

# 初始化Ray + 分布式训练
ray.init(ignore_reinit_error=True, num_gpus=1)
register_atlas_task()

# RLlib PPO配置（适配Atlas高维控制）
config = (
    PPOConfig()
    .environment(
        env="isaacgym",
        env_config={
            "env_name": "Atlas",
            "numEnvs": 4096,
            "maxEpisodeLength": 1000
        }
    )
    .framework("torch")
    .rollouts(num_rollout_workers=0)  # Isaac Gym自身并行，无需额外Worker
    .training(
        lr=2e-4,  # 低学习率，避免高维动作震荡
        gamma=0.99,  # 长期奖励（爬楼梯需要多步规划）
        train_batch_size=65536,  # 超大批次，适配4096环境
        sgd_minibatch_size=16384,
        clip_range=0.2,
        vf_loss_coeff=0.8,  # 价值损失权重提高，增强稳定性
        ent_coeff=0.01,  # 少量熵奖励，鼓励探索但不摔倒
        # 策略网络：深度残差网络（适配高维状态）
        model={
            "fcnet_hiddens": [1024, 1024, 512],
            "fcnet_activation": "relu",
            "use_residual": True,  # 残差连接，缓解梯度消失
        }
    )
    .resources(num_gpus=1)
    .evaluation(evaluation_num_workers=1)
)

# 训练Atlas（建议至少1000轮，GPU：A100/A800）
algo = config.build()
for epoch in range(1000):
    result = algo.train()
    # 打印关键指标
    avg_reward = result["episode_reward_mean"]
    avg_forward_vel = result["custom_metrics"].get("forward_vel_mean", 0.0)
    success_rate = result["custom_metrics"].get("stair_climb_success_rate", 0.0)
    print(f"Epoch {epoch}: 平均奖励={avg_reward:.2f} | 前进速度={avg_forward_vel:.2f}m/s | 爬楼梯成功率={success_rate:.2f}")
    # 每50轮保存模型
    if epoch % 50 == 0:
        algo.save(f"./atlas_ppo_epoch_{epoch}")

# 评估Atlas性能
eval_result = algo.evaluate()
print(f"最终评估：平均奖励={eval_result['evaluation']['episode_reward_mean']:.2f} | 爬楼梯成功率={eval_result['evaluation']['custom_metrics']['stair_climb_success_rate']:.2f}")

algo.stop()
ray.shutdown()
