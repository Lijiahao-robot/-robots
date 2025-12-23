# 安装：需下载Isaac Gym（https://developer.nvidia.com/isaac-gym）
from isaacgym import gymapi, gymutil
from isaacgymenvs import make_env

# 1. 创建四足机器人环境（ANYmal）
env = make_env(
    "Anymal",  # 环境名称：四足机器人
    rl_device="cuda:0",  # GPU加速
    sim_device="cuda:0",
    headless=False  # 可视化
)

# 2. 初始化PPO算法（Isaac Gym内置）
from isaacgymenvs.utils.rl_utils import get_ppo_agent
agent = get_ppo_agent(env)

# 3. 训练循环（GPU并行，数千环境同时运行）
for epoch in range(100):
    # 采样经验（GPU并行）
    obs, rewards, dones, infos = env.reset(), 0, False, {}
    for _ in range(env.max_episode_length):
        actions = agent.get_actions(obs)
        obs, rewards, dones, infos = env.step(actions)
    # 训练策略
    agent.train()
    print(f"Epoch {epoch}: 平均奖励={infos['episode_reward_mean']:.2f}")

# 4. 保存模型
agent.save("anymal_ppo_model")
env.close()
