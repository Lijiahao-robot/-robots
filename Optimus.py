def compute_reward(self):
    self.rew[:] = 0.0
    # 1. 行走稳定性（核心）
    com_rot = self.obs[:, 54:60]
    roll, pitch = com_rot[:, 0], com_rot[:, 1]
    self.rew -= 15 * (torch.abs(roll) + torch.abs(pitch))
    
    # 2. 低速行走奖励（Optimus目标：0.8m/s）
    com_vel = self.obs[:, 60:63]
    forward_vel = com_vel[:, 0]
    self.rew += 10 * torch.exp(-(forward_vel - 0.8)**2 / 0.2)
    
    # 3. 抓取奖励（手部关节接近目标物体）
    hand_joints = self.obs[:, 20:27]  # 手部7DoF
    target_hand_pos = torch.tensor([0.5, 0.0, 0.2], device=self.device).repeat(self.num_envs, 1)
    hand_dist = torch.norm(hand_joints - target_hand_pos, dim=1)
    self.rew += 20 * torch.exp(-hand_dist / 0.1)
    
    # 4. 能耗惩罚（重点优化）
    self.rew -= 0.1 * torch.norm(self.actions, dim=1)
    
    # 5. 摔倒惩罚
    fall_mask = (torch.abs(roll) > np.pi/4) | (torch.abs(pitch) > np.pi/4)
    self.rew[fall_mask] -= 150
