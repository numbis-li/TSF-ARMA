# Contributing to TSF-ARMA

我们非常欢迎您为 TSF-ARMA 项目做出贡献！无论是修复 bug、改进文档还是添加新功能，您的贡献都将帮助我们使项目变得更好。

## 如何贡献

### 报告问题

如果您发现了 bug 或有新的功能建议，请：

1. 首先检查是否已经有相关的 issue
2. 如果没有，创建一个新的 issue，并：
   - 清晰描述问题/建议
   - 提供复现步骤（如果是 bug）
   - 提供相关的代码片段或错误信息
   - 说明您的运行环境（Python 版本、操作系统等）

### 提交代码

1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建一个 Pull Request

### 代码风格

- 遵循 PEP 8 Python 代码风格指南
- 使用有意义的变量名和函数名
- 添加必要的注释和文档字符串
- 确保代码可以在 CPU 环境下运行

### 提交信息规范

提交信息应该清晰描述更改内容，建议格式如下：

```
类型: 简短描述

详细描述（如果需要）
```

类型可以是：
- feat: 新功能
- fix: 修复 bug
- docs: 文档更新
- style: 代码格式调整
- refactor: 代码重构
- test: 添加测试
- chore: 构建过程或辅助工具的变动

### 文档贡献

如果您想改进文档，请：

1. 确保文档清晰易懂
2. 提供必要的代码示例
3. 检查拼写和语法
4. 保持文档结构的一致性

## 开发设置

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/TSF-ARMA.git
cd TSF-ARMA
```

2. 创建虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 测试

在提交代码之前，请确保：

1. 所有测试都能通过
2. 新功能有相应的测试用例
3. 代码在 CPU 环境下可以正常运行
4. 内存使用在合理范围内（≤8GB）

## 问题反馈

如果您有任何问题或建议，可以：

1. 在 GitHub Issues 中提出
2. 发送邮件到维护者邮箱
3. 在讨论区参与讨论

感谢您的贡献！ 