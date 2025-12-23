# 安装：pip install stable-baselines3[extra] gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gymnasium import make

# 1. 创建环境（支持向量环境加速）
env = make_vec_env("CartPole-v1", n_envs=4)  # 4个并行环境

# 2. 初始化PPO模型（MlpPolicy适配低维状态）
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    verbose=1,
    tensorboard_log="./cartpole_ppo_logs/"
)

# 3. 训练模型
model.learn(total_timesteps=100000)

# 4. 保存模型
model.save("cartpole_ppo_model")

# 5. 加载模型并测试
del model
model = PPO.load("cartpole_ppo_model")
test_env = make("CartPole-v1", render_mode="human")  # 可视化
obs, info = test_env.reset()

for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    if terminated or truncated:
        obs, info = test_env.reset()

test_env.close()
