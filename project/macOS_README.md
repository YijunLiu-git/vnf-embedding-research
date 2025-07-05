# macOS系统使用说明

## 🍎 macOS环境配置

您的系统使用Homebrew管理的Python，pip被限制安装。以下是几种解决方案：

### 方案1: 使用现有环境（推荐）
大部分依赖已经安装，直接运行：
```bash
python project/test_system.py
cd project && python main_multi_agent.py
```

### 方案2: 创建虚拟环境
```bash
python3 -m venv vnf_env
source vnf_env/bin/activate
pip install torch torch-geometric numpy networkx matplotlib pyyaml gym scipy
```

### 方案3: 使用Homebrew安装
```bash
brew install python-packaging
brew install numpy
```

### 方案4: 强制安装（不推荐）
```bash
pip install --break-system-packages --user packaging
```

## 🚀 快速开始

1. 测试系统：
```bash
python project/test_system.py
```

2. 运行训练：
```bash
cd project
python main_multi_agent.py
```

3. 查看结果：
```bash
cat results/macos_training_results.json
```

## 📊 预期输出

训练成功后应该看到：
```
🎉 macOS兼容训练完成!
总episodes: 15
平均奖励: 45.67
平均SAR: 0.823
成功率: 0.867
✅ 结果已保存到 results/macos_training_results.json
```

## 🔧 故障排除

如果遇到模块缺失错误：
1. 检查是否在正确目录
2. 确认Python版本 >= 3.7
3. 尝试创建虚拟环境
4. 联系管理员安装缺失包
