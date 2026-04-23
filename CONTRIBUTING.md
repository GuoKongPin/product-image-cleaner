# 贡献指南 | Contributing Guide

感谢你对 Product Image Cleaner 项目的关注！我们欢迎各种形式的贡献。

## 如何贡献 | How to Contribute

### 报告问题 | Reporting Issues

如果你发现了 bug 或有功能建议，请通过 [GitHub Issues](https://github.com/yourusername/product-image-cleaner/issues) 提交。

提交时请包含：
- 问题描述
- 复现步骤
- 期望行为 vs 实际行为
- 系统环境（OS、Python 版本等）
- 相关截图或日志

### 提交代码 | Submitting Code

1. **Fork 仓库**
   ```bash
   git clone https://github.com/yourusername/product-image-cleaner.git
   cd product-image-cleaner
   ```

2. **创建分支**
   ```bash
   git checkout -b feature/your-feature-name
   # 或
   git checkout -b fix/your-bug-fix
   ```

3. **开发并测试**
   ```bash
   # 安装开发依赖
   pip install -r backend/requirements.txt
   
   # 运行测试
   cd backend
   python -m pytest tests/
   ```

4. **提交更改**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

   提交信息规范：
   - `feat:` 新功能
   - `fix:` 修复 bug
   - `docs:` 文档更新
   - `refactor:` 代码重构
   - `perf:` 性能优化
   - `test:` 测试相关

5. **推送并创建 PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   然后在 GitHub 上创建 Pull Request。

## 开发规范 | Development Guidelines

### 代码风格 | Code Style

- 遵循 PEP 8 Python 代码规范
- 使用 4 空格缩进
- 最大行长度 100 字符
- 函数和类添加 docstring

### 项目结构 | Project Structure

```
backend/
├── app.py              # 主服务入口
├── migan_inpainter.py  # MI-GAN 修复器
├── lama_inpainter.py   # LaMa 修复器
└── tests/              # 测试文件
```

### 添加新功能 | Adding Features

1. 在 `app.py` 中添加 API 端点
2. 在 `frontend/index.html` 中添加对应的前端界面
3. 更新 `README.md` 文档
4. 添加单元测试

## 测试 | Testing

```bash
cd backend

# 运行所有测试
python -m pytest

# 运行特定测试
python -m pytest tests/test_inpainting.py

# 生成覆盖率报告
python -m pytest --cov=app tests/
```

## 文档 | Documentation

- 更新 `README.md` 中的 API 说明
- 更新 `MODELS.md` 中的模型信息
- 代码中添加清晰的注释

## 行为准则 | Code of Conduct

- 尊重所有贡献者
- 接受建设性的批评
- 关注对社区最有利的事情
- 展现同理心

## 联系方式 | Contact

如有问题，可以通过以下方式联系：
- GitHub Issues: [提交问题](https://github.com/yourusername/product-image-cleaner/issues)
- Email: your.email@example.com

---

再次感谢你的贡献！
