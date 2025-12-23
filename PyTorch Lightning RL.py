# 安装：pip install pytorch-lightning gymnasium torch
import torch
import torch.nn as nn
import pytorch_lightning as pl
from gymnasium import make

# 1. 自定义PPO策略网络
class PPONet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.actor = nn.Linear(64, act_dim)  # 动作预测
        self.critic = nn.Linear(64, 1)       # 价值预测

    def forward(self, x):
        x = self.fc(x)
        return self.actor(x), self.critic(x)

# 2. Lightning RL模块（标准化训练流程）
class PPOAgent(pl.LightningModule):
    def __init__(self, obs_dim=4, act_dim=2, lr=3e-4):
        super().__init__()
        self.net = PPONet(obs_dim, act_dim)
        self.lr = lr
        self.env = make("CartPole-v1")

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        # 简化版PPO训练逻辑（核心：策略梯度+价值损失）
        obs, actions, rewards = batch
        logits, value = self.net(obs)
        loss = -torch.mean(torch.log_softmax(logits, dim=-1)[range(len(actions)), actions] * rewards) + nn.MSELoss()(value.squeeze(), rewards)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.lr)

# 3. 数据加载（简化版，实际需经验回放）
class RLDataset(torch.utils.data.Dataset):
    def __init__(self, env, size=1000):
        self.obs, self.actions, self.rewards = [], [], []
        obs, _ = env.reset()
        for _ in range(size):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            self.obs.append(obs), self.actions.append(action), self.rewards.append(reward)
            obs = next_obs if not (terminated or truncated) else env.reset()[0]

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return torch.tensor(self.obs[idx], dtype=torch.float32), torch.tensor(self.actions[idx]), torch.tensor(self.rewards[idx], dtype=torch.float32)

# 4. 训练
dataset = RLDataset(make("CartPole-v1"), size=10000)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
agent = PPOAgent()
trainer = pl.Trainer(max_epochs=10, accelerator="cpu")  # 可选GPU
trainer.fit(agent, dataloader)
