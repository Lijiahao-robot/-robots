# 安装：pip install tensorflow tensorflow-agents gymnasium
import tensorflow as tf
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

# 1. 加载环境（TF封装）
train_env = tf_py_environment.TFPyEnvironment(suite_gym.load("CartPole-v1"))
eval_env = tf_py_environment.TFPyEnvironment(suite_gym.load("CartPole-v1"))

# 2. 自定义Q网络
q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=(128, 64)  # 自定义全连接层
)

# 3. 配置DQN Agent
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
train_step = tf.Variable(0)
agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step
)
agent.initialize()

# 4. 经验回放缓冲区
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=100000
)

# 5. 训练函数
def collect_step(environment, agent, buffer):
    time_step = environment.current_time_step()
    action_step = agent.collect_policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    buffer.add_batch(traj)

# 6. 训练循环
collect_steps_per_iteration = 10
batch_size = 64
for _ in range(1000):
    # 收集经验
    for _ in range(collect_steps_per_iteration):
        collect_step(train_env, agent, replay_buffer)
    # 采样经验并训练
    experience, unused_info = replay_buffer.get_next(
        sample_batch_size=batch_size, num_steps=2
    )
    train_loss = agent.train(experience).loss
    print(f"训练损失：{train_loss:.4f}")

# 7. 评估
def evaluate(environment, agent, num_episodes=10):
    total_reward = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_reward = 0.0
        while not time_step.is_last():
            action_step = agent.policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_reward += time_step.reward
        total_reward += episode_reward
    return total_reward / num_episodes

print(f"评估平均奖励：{evaluate(eval_env, agent):.2f}")
