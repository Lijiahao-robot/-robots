class OmniDriveEnv(DiffDriveEnv):
    def __init__(self, render=False):
        super().__init__(render)
        # 替换为全向轮机器人模型（4轮全向）
        self.robot_id = p.loadURDF(
            "omnirob.urdf",  # 全向轮URDF模型
            [0, 0, 0.1],
            p.getQuaternionFromEuler([0, 0, 0])
        )
        # 全向轮关节ID（前/后/左/右）
        self.wheel_joints = [0, 1, 2, 3]
        self.act_dim = 3  # x/y速度 + 角速度（连续）
        self.obs_dim = 9  # 位姿+速度+目标相对位置+障碍物距离

    def step(self, action):
        # 全向轮动作映射：x/y速度（-1~1） + 角速度（-π/2~π/2）
        action = np.clip(action, -1.0, 1.0)
        vx, vy, omega = action[0], action[1], action[2] * np.pi/2

        # 全向轮速度解算（4轮速度映射）
        wheel_speeds = [
            vx - vy - omega,  # 前左轮
            vx + vy + omega,  # 前右轮
            vx + vy - omega,  # 后左轮
            vx - vy + omega   # 后右轮
        ]

        # 执行全向控制
        for i, joint_id in enumerate(self.wheel_joints):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_id,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=wheel_speeds[i],
                force=10.0
            )

        p.stepSimulation()
        self.current_step += 1

        obs = self._get_obs()
        reward = self._compute_reward()
        dist_to_target = np.linalg.norm([p.getBasePositionAndOrientation(self.robot_id)[0][0]-self.target_pos[0], 
                                         p.getBasePositionAndOrientation(self.robot_id)[0][1]-self.target_pos[1]])
        terminated = (dist_to_target < 0.1) or (self._get_closest_obstacle_dist() < 0.15) or (self.current_step >= self.max_episode_steps)
        truncated = False
        info = {"dist_to_target": dist_to_target}

        return obs, reward, terminated, truncated, info

# 训练全向轮机器人
env = make_vec_env(lambda render=False: OmniDriveEnv(render), n_envs=4)
model = PPO("MlpPolicy", env, learning_rate=5e-4, n_steps=2048, verbose=1)
model.learn(total_timesteps=1_000_000)
model.save("omni_drive_ppo_final")
