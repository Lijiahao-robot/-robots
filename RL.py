1.# 高层：指令层（速度/方向）→ 中层：步态层（行走/跑跳）→ 底层：关节层（力矩）
class HierarchicalPPO(nn.Module):
    def __init__(self, obs_dim, high_level_dim=2, mid_level_dim=8, low_level_dim=28):
        super().__init__()
        # 高层策略：输入状态→输出速度/方向指令（2维）
        self.high_level = nn.Sequential(nn.Linear(obs_dim, 512), nn.ReLU(), nn.Linear(512, high_level_dim))
        # 中层策略：输入指令+状态→输出步态参数（8维：步长/步频/抬脚高度等）
        self.mid_level = nn.Sequential(nn.Linear(obs_dim+high_level_dim, 512), nn.ReLU(), nn.Linear(512, mid_level_dim))
        # 底层策略：输入步态+状态→输出关节力矩（28维）
        self.low_level = nn.Sequential(nn.Linear(obs_dim+mid_level_dim, 1024), nn.ReLU(), nn.Linear(1024, low_level_dim))

    def forward(self, obs):
        high_out = self.high_level(obs)
        mid_out = self.mid_level(torch.cat([obs, high_out], dim=1))
        low_out = self.low_level(torch.cat([obs, mid_out], dim=1))
        return low_out
      2.Sim-to-Real
def randomize_atlas_phy_params(self, env_id):
    """随机化Atlas物理参数，缩小仿真-真实差距"""
    # 1. 关节参数随机化（力矩/阻尼/摩擦 ±30%）
    dof_props = self.gym.get_actor_dof_properties(self.envs[env_id], self.atlas_handles[env_id])
    dof_props["effort"][:] *= np.random.uniform(0.7, 1.3)
    dof_props["damping"][:] *= np.random.uniform(0.7, 1.3)
    dof_props["friction"][:] *= np.random.uniform(0.7, 1.3)
    self.gym.set_actor_dof_properties(self.envs[env_id], self.atlas_handles[env_id], dof_props)
    
    # 2. 地面参数随机化（摩擦0.5~1.5，弹性0.1~0.3）
    ground_mat = self.gym.create_material(self.sim, 
        np.random.uniform(0.5, 1.5),  # 静摩擦
        np.random.uniform(0.5, 1.5),  # 动摩擦
        np.random.uniform(0.1, 0.3)   # 弹性
    )
    self.gym.set_actor_material(self.envs[env_id], self.ground_id, ground_mat)
    
    # 3. 质量分布随机化（整体质量±10%，局部质量±20%）
    total_mass = self.gym.get_actor_mass(self.envs[env_id], self.atlas_handles[env_id])
    self.gym.set_actor_mass(self.envs[env_id], self.atlas_handles[env_id], total_mass * np.random.uniform(0.9, 1.1))
    # 4. 重力随机化（9.8~10.2 m/s²）
    self.gym.set_gravity(self.sim, gymapi.Vec3(0, 0, -np.random.uniform(9.8, 10.2)))
3.Sim-to-Real 
# 用真实Atlas/Optimus采集的少量数据（1000步）微调策略
def fine_tune_with_real_data(model, real_data_loader, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # 极低学习率
    for epoch in range(epochs):
        for obs, actions, rewards in real_data_loader:
            pred_actions = model(obs)
            # 损失：模仿真实动作 + 奖励优化
            loss = nn.MSELoss()(pred_actions, actions) - 0.1 * torch.mean(rewards)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"微调Epoch {epoch}: Loss={loss.item():.4f}")
    return model      
