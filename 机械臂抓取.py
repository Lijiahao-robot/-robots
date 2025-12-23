# 安装依赖
pip install stable-baselines3[extra] pybullet gymnasium numpy matplotlib
import pybullet as p
import pybullet_envs  # 注册PyBullet的机器人环境
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# 1. 创建机械臂抓取环境（向量环境加速训练）
# KukaDiverseObjectEnv：KUKA机械臂+10种随机物体，连续动作空间（关节力矩）
env = make_vec_env(
    "KukaDiverseObjectEnv-v0",
    n_envs=4,  # 4个并行环境
    env_kwargs={
        "isDiscrete": False,  # 连续动作空间（机械臂控制核心）
        "render": False,      # 训练时关闭可视化，测试时开启
        "maxSteps": 100,      # 每个episode最大步数
    }
)

# 2. 初始化PPO模型（适配连续动作的关键配置）
model = PPO(
    policy="MlpPolicy",  # 多层感知机策略（低维状态足够）
    env=env,
    learning_rate=3e-4,
    n_steps=2048,        # 每轮收集的样本数（连续控制建议增大）
    batch_size=64,
    gamma=0.95,          # 折扣因子（抓取任务短期奖励为主）
    gae_lambda=0.9,      # 广义优势估计，平衡偏差与方差
    clip_range=0.2,      # PPO核心：动作裁剪，保证训练稳定
    verbose=1,
    tensorboard_log="./kuka_ppo_logs/"
)

# 3. 训练模型（建议至少100万步，视GPU性能调整）
model.learn(total_timesteps=1_000_000)

# 4. 保存模型
model.save("kuka_grasp_ppo")

# 5. 测试模型（开启可视化）
del model
model = PPO.load("kuka_grasp_ppo")
test_env = pybullet_envs.bullet.kuka_diverse_object_gym_env.KukaDiverseObjectEnv(
    isDiscrete=False,
    render=True,  # 可视化机械臂抓取
    maxSteps=100
)

# 评估抓取成功率
mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=10)
print(f"平均奖励：{mean_reward:.2f} ± {std_reward:.2f}")

# 可视化测试
obs, info = test_env.reset()
for _ in range(500):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    if terminated or truncated:
        obs, info = test_env.reset()

test_env.close()
p.disconnect()
