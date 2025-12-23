环境搭建（复用之前的 SB3+PyBullet 环境）
pip install stable-baselines3[extra] pybullet gymnasium numpy matplotlib
import pybullet as p
import pybullet_data
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

# 自定义差速轮机器人环境
class DiffDriveEnv:
    def __init__(self, render=False):
        self.render = render
        self.physics_client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1/50)  # 差速轮控制频率50Hz

        # 1. 加载场景（地面+随机障碍物）
        self.ground_id = p.loadURDF("plane.urdf")
        self.obstacle_ids = []
        for _ in range(5):  # 生成5个随机圆柱障碍物
            obs_pos = np.random.uniform(-2, 2, 3)
            obs_pos[2] = 0.5  # 障碍物高度
            obs_id = p.loadURDF(
                "cylinder.urdf",
                obs_pos,
                p.getQuaternionFromEuler([0, 0, 0]),
                globalScaling=0.3  # 障碍物半径0.3m
            )
            self.obstacle_ids.append(obs_id)

        # 2. 加载差速轮机器人（TurtleBot3模型）
        self.robot_id = p.loadURDF(
            "turtlebot3_burger.urdf",  # PyBullet内置TurtleBot3模型
            [0, 0, 0.1],
            p.getQuaternionFromEuler([0, 0, 0])
        )
        # 差速轮关节ID（左轮/右轮）
        self.left_wheel_joint = 0
        self.right_wheel_joint = 1
        # 目标位置（随机生成）
        self.target_pos = np.random.uniform(-3, 3, 2)
        self.target_pos = np.clip(self.target_pos, -2.5, 2.5)  # 限制目标范围

        # 3. 状态/动作空间定义
        self.obs_dim = 8  # x/y位姿 + 偏航角 + 左右轮速度 + 目标相对x/y + 最近障碍物距离
        self.act_dim = 2  # 左右轮速度（-1~1 m/s）
        self.max_episode_steps = 500
        self.current_step = 0

    def reset(self, seed=None, options=None):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        # 重置场景
        p.loadURDF("plane.urdf")
        self.obstacle_ids = []
        for _ in range(5):
            obs_pos = np.random.uniform(-2, 2, 3)
            obs_pos[2] = 0.5
            obs_id = p.loadURDF("cylinder.urdf", obs_pos, p.getQuaternionFromEuler([0, 0, 0]), globalScaling=0.3)
            self.obstacle_ids.append(obs_id)
        # 重置机器人位置
        self.robot_id = p.loadURDF("turtlebot3_burger.urdf", [0, 0, 0.1], p.getQuaternionFromEuler([0, 0, 0]))
        # 重置目标位置
        self.target_pos = np.random.uniform(-3, 3, 2)
        self.target_pos = np.clip(self.target_pos, -2.5, 2.5)
        self.current_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        # 1. 机器人位姿和速度
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        base_euler = p.getEulerFromQuaternion(base_orn)
        yaw = base_euler[2]  # 偏航角（仅关注水平转向）
        # 2. 车轮速度
        left_wheel_vel = p.getJointState(self.robot_id, self.left_wheel_joint)[1]
        right_wheel_vel = p.getJointState(self.robot_id, self.right_wheel_joint)[1]
        # 3. 目标相对位置（机器人坐标系下）
        rel_target_x = (self.target_pos[0] - base_pos[0]) * np.cos(yaw) + (self.target_pos[1] - base_pos[1]) * np.sin(yaw)
        rel_target_y = -(self.target_pos[0] - base_pos[0]) * np.sin(yaw) + (self.target_pos[1] - base_pos[1]) * np.cos(yaw)
        # 4. 最近障碍物距离
        closest_obs_dist = self._get_closest_obstacle_dist()

        # 拼接观测
        obs = np.array([
            base_pos[0], base_pos[1], yaw,
            left_wheel_vel, right_wheel_vel,
            rel_target_x, rel_target_y,
            closest_obs_dist
        ], dtype=np.float32)
        return obs

    def _get_closest_obstacle_dist(self):
        # 计算机器人到最近障碍物的距离
        base_pos = p.getBasePositionAndOrientation(self.robot_id)[0]
        min_dist = float("inf")
        for obs_id in self.obstacle_ids:
            obs_pos = p.getBasePositionAndOrientation(obs_id)[0]
            dist = np.linalg.norm([base_pos[0]-obs_pos[0], base_pos[1]-obs_pos[1]])
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _compute_reward(self):
        reward = 0.0
        base_pos = p.getBasePositionAndOrientation(self.robot_id)[0]
        # 1. 到达目标奖励（核心）：距离越近，奖励越高
        dist_to_target = np.linalg.norm([base_pos[0]-self.target_pos[0], base_pos[1]-self.target_pos[1]])
        reward += 10 * np.exp(-dist_to_target / 0.5)  # 指数奖励，距离<0.1m时奖励≈10

        # 2. 避障奖励：远离障碍物（距离>0.3m加分）
        closest_obs_dist = self._get_closest_obstacle_dist()
        if closest_obs_dist > 0.3:
            reward += 2.0
        else:
            reward -= 5.0  # 靠近障碍物惩罚

        # 3. 碰撞惩罚：撞到障碍物直接扣大分
        if closest_obs_dist < 0.15:
            reward -= 20.0

        # 4. 速度奖励：保持合理速度（避免静止）
        left_vel = p.getJointState(self.robot_id, self.left_wheel_joint)[1]
        right_vel = p.getJointState(self.robot_id, self.right_wheel_joint)[1]
        avg_vel = (abs(left_vel) + abs(right_vel)) / 2
        reward += 0.5 * avg_vel

        # 5. 步数惩罚：鼓励快速到达目标
        reward -= 0.1

        return reward

    def step(self, action):
        # 动作裁剪：限制车轮速度（-1~1 m/s）
        action = np.clip(action, -1.0, 1.0)
        left_vel, right_vel = action[0], action[1]

        # 执行差速控制
        p.setJointMotorControl2(
            bodyUniqueId=self.robot_id,
            jointIndex=self.left_wheel_joint,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=left_vel,
            force=10.0
        )
        p.setJointMotorControl2(
            bodyUniqueId=self.robot_id,
            jointIndex=self.right_wheel_joint,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=right_vel,
            force=10.0
        )

        p.stepSimulation()
        self.current_step += 1

        # 状态更新
        obs = self._get_obs()
        reward = self._compute_reward()
        # 终止条件：到达目标/碰撞/步数耗尽
        dist_to_target = np.linalg.norm([p.getBasePositionAndOrientation(self.robot_id)[0][0]-self.target_pos[0], 
                                         p.getBasePositionAndOrientation(self.robot_id)[0][1]-self.target_pos[1]])
        terminated = (dist_to_target < 0.1) or (self._get_closest_obstacle_dist() < 0.15) or (self.current_step >= self.max_episode_steps)
        truncated = False
        info = {"dist_to_target": dist_to_target, "closest_obs_dist": self._get_closest_obstacle_dist()}

        return obs, reward, terminated, truncated, info

    def close(self):
        p.disconnect(self.physics_client)

# 注册环境
def make_diff_drive_env(render=False):
    return DiffDriveEnv(render=render)

# 1. 创建向量环境
env = make_vec_env(make_diff_drive_env, n_envs=4, env_kwargs={"render": False})

# 2. 配置PPO（适配差速轮低维控制）
eval_callback = EvalCallback(
    eval_env=make_diff_drive_env(render=True),
    eval_freq=5000,
    n_eval_episodes=10,
    save_best_model=True,
    best_model_save_path="./diff_drive_best/"
)

model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=5e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.95,
    ent_coeff=0.01,  # 少量探索，保证避障安全
    clip_range=0.2,
    verbose=1,
    tensorboard_log="./diff_drive_tensorboard/"
)

# 3. 训练（100万步，足够收敛）
model.learn(total_timesteps=1_000_000, callback=eval_callback)
model.save("diff_drive_ppo_final")

# 4. 测试（可视化避障+到达目标）
test_env = DiffDriveEnv(render=True)
obs, info = test_env.reset()
success_count = 0
for _ in range(10):  # 测试10轮
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        print(f"距离目标：{info['dist_to_target']:.2f}m，最近障碍物：{info['closest_obs_dist']:.2f}m")
        if terminated:
            if info['dist_to_target'] < 0.1:
                success_count += 1
            obs, info = test_env.reset()
            break
print(f"10轮测试成功率：{success_count/10*100:.1f}%")
test_env.close()
