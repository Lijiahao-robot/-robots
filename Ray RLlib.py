# 安装：pip install ray[rllib] gymnasium
import ray
from ray.rllib.algorithms.ppo import PPOConfig

# 1. 初始化Ray（分布式框架）
ray.init(ignore_reinit_error=True)

# 2. 配置PPO算法
config = (
    PPOConfig()
    .environment("Pendulum-v1")  # 连续动作空间任务
    .framework("torch")  # 可选tf/torch
    .rollouts(num_rollout_workers=4)  # 4个采样进程
    .training(
        lr=3e-4,
        gamma=0.99,
        train_batch_size=4096,
        sgd_minibatch_size=512,
    )
    .evaluation(evaluation_num_workers=1)
)

# 3. 构建算法实例
algo = config.build()

# 4. 分布式训练
for i in range(10):
    result = algo.train()
    print(f"迭代{i+1}：奖励={result['episode_reward_mean']:.2f}")

# 5. 评估模型
eval_result = algo.evaluate()
print(f"评估奖励：{eval_result['evaluation']['episode_reward_mean']:.2f}")

# 6. 保存/加载模型
algo.save("./pendulum_rllib_ppo")
algo.stop()
ray.shutdown()
