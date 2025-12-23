import pybullet as p
import pybullet_envs
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# 1. 创建四足机器人环境（ANYmal）
env = make_vec_env(
    "AnymalBulletEnv-v0",  # PyBullet内置ANYmal四足机器人环境
    n_envs=4,
    env_kwargs={
        "render": False,
        "maxEpisodeSteps": 500,
        "controlMode": "torque",  # 力矩控制（足式机器人主流）
    }
)

# 2. PPO模型配置（适配足式机器人高维动作）
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-4,
    n_steps=4096,  # 高维动作需更多样本
    batch_size=128,
    gamma=0.99,    # 运动任务需长期奖励，gamma调高
    clip_range=0.3,
    verbose=1,
    tensorboard_log="./anymal_ppo_logs/"
)

# 3. 训练（至少200万步，建议GPU加速）
model.learn(total_timesteps=2_000_000)
model.save("anymal_walk_ppo")

# 4. 测试（可视化行走）
test_env = pybullet_envs.bullet.ymal_bullet_env.AnymalBulletEnv(
    render=True,
    maxEpisodeSteps=500
)
obs, info = test_env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    if terminated or truncated:
        obs, info = test_env.reset()

test_env.close()
p.disconnect()

2..
# 需先安装Isaac Gym（https://developer.nvidia.com/isaac-gym）
from isaacgymenvs import make_env
from ray.rllib.algorithms.ppo import PPOConfig
import ray

# 1. 初始化Ray分布式框架
ray.init(ignore_reinit_error=True)

# 2. 创建Isaac Gym四足机器人环境（A1）
env = make_env(
    "A1",  # 大疆A1四足机器人
    rl_device="cuda:0",
    sim_device="cuda:0",
    headless=False,  # 可视化
    env_cfg={
        "numEnvs": 4096,  # GPU并行4096个环境（核心优势）
        "maxEpisodeLength": 1000,
        "task": "walk",    # 任务：行走/越障/爬坡
        "domainRandomization": True,  # 域随机化（Sim-to-Real关键）
    }
)

# 3. 配置RLlib PPO（适配高维连续控制）
config = (
    PPOConfig()
    .environment(env="isaacgym", env_config={"env_name": "A1"})
    .framework("torch")
    .rollouts(num_rollout_workers=0)  # Isaac Gym自身并行，无需额外worker
    .training(
        lr=5e-4,
        gamma=0.99,
        train_batch_size=16384,
        sgd_minibatch_size=8192,
        clip_range=0.2,
        vf_loss_coeff=0.5,  # 价值损失权重，平衡策略与价值学习
    )
    .resources(num_gpus=1)  # 单GPU训练
)

# 4. 训练
algo = config.build()
for epoch in range(50):
    result = algo.train()
    print(f"Epoch {epoch}: 平均奖励={result['episode_reward_mean']:.2f}")
    # 每10轮保存模型
    if epoch % 10 == 0:
        algo.save(f"./a1_walk_ppo_epoch_{epoch}")

algo.stop()
ray.shutdown()
