# -robots
以下是覆盖主流场景的强化学习框架大全，包含框架定位、核心特性、适用场景，以及可直接运行的示例代码（涵盖单机入门、分布式训练、机器人控制、多智能体等场景），方便你按需选择和落地。
# 1. 创建虚拟环境（推荐，避免依赖冲突）
conda create -n quadruped_rl python=3.9
conda activate quadruped_rl  # Windows/Linux通用
# 若无conda：python -m venv quadruped_rl && source quadruped_rl/bin/activate（Linux）/ quadruped_rl\Scripts\activate（Windows）
# 2. 安装核心依赖
pip install --upgrade pip
pip install stable-baselines3[extra]==2.0.0  # 稳定版本，避免API变动
pip install pybullet==3.2.5 gymnasium==0.29.1 numpy==1.26.0 matplotlib==3.8.0
pip install torch==2.0.1  # CPU/GPU通用，GPU需匹配CUDA版本（见下方备注）
# 3. 验证安装
python -c "import pybullet; import stable_baselines3; print('安装成功！')"
验证足式机器人环境
import pybullet as p
import pybullet_envs  # 自动注册PyBullet的机器人环境
import time

# 1. 连接PyBullet仿真器（GUI模式，可视化）
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.81)  # 设置重力

# 2. 加载ANYmal四足机器人模型
robot_id = p.loadURDF(
    "pybullet_data/anymal/anymal.urdf",  # PyBullet内置模型路径
    [0, 0, 0.5],  # 初始位置
    p.getQuaternionFromEuler([0, 0, 0])  # 初始姿态
)

# 3. 加载地面
p.loadURDF("pybullet_data/plane.urdf")

# 4. 可视化运行（控制机器人关节随机运动）
joint_num = p.getNumJoints(robot_id)
for _ in range(1000):
    # 随机设置关节力矩（模拟动作输出）
    for i in range(joint_num):
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=i,
            controlMode=p.TORQUE_CONTROL,
            force=0.5 * (p.getJointState(robot_id, i)[1] - 0.1)  # 简单力矩控制
        )
    p.stepSimulation()
    time.sleep(1/240)  # 仿真步长240Hz

# 5. 断开连接
p.disconnect()
1.系统依赖安装（Linux 为例）
# 安装基础依赖
sudo apt update && sudo apt install -y \
    libpython3.8-dev libglew-dev libboost-all-dev \
    libssl-dev libx11-dev libxcursor-dev libxrandr-dev \
    libxinerama-dev libxi-dev libxxf86vm-dev libasound2-dev

# 安装CUDA（需匹配Isaac Gym版本，推荐11.8）
# 参考：https://developer.nvidia.com/cuda-11-8-0-download-archive
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit
echo "export PATH=/usr/local/cuda-11.8/bin:\$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
2.Isaac Gym 安装
# 1. 解压安装包
tar -xvf isaacgym_ubuntu-2022.2.tar.gz
cd isaacgym
# 2. 安装Python依赖
pip install -r requirements.txt
# 3. 验证Isaac Gym核心库
python examples/python/1080_anymal.py  # 运行ANYmal机器人示例
3.Isaac Gym Envs（强化学习适配）安装
# 1. 克隆仓库
git clone https://github.com/NVIDIA-Omniverse/IsaacGymEnvs.git
cd IsaacGymEnvs
# 2. 安装依赖
pip install -e .
# 3. 验证四足机器人RL环境
python train.py task=anymal headless=False  # 训练ANYmal行走，开启可视化
4.Ray RLlib 集成
# 安装Ray RLlib
pip install ray[rllib]==2.6.0 torch==2.0.1
# 验证RLlib+Isaac Gym
python examples/rllib/ppo_anymal.py
