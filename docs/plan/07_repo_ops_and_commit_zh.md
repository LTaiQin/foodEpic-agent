# 07 仓库、提交与运维计划

## 7.1 仓库绑定

本地仓库：

```text
/22liushoulong/agent/hd-epic
```

远端仓库：

```text
git@github.com:LTaiQin/foodEpic-agent.git
```

当前分支：

```text
main
```

## 7.2 SSH 配置

项目专用 key：

```text
.secrets/ssh/id_ed25519_foodEpic_agent
.secrets/ssh/id_ed25519_foodEpic_agent.pub
```

注意：

- `.secrets/` 不进入 git。
- 私钥不能输出到文档、日志或提交中。
- 当前 SSH 使用 `ssh.github.com:443`，并通过 `127.0.0.1:7890` SOCKS 代理。

## 7.3 数据不入库规则

必须忽略：

- `data/`
- `data-test/`
- `annotations/`
- `outputs/`
- `.secrets/`
- `*.mp4`
- `*.hdf5`
- `*.duckdb`
- `*.parquet`
- `*.zip`
- `*.gz`
- 日志和 state 文件。

允许入库：

- 代码。
- 轻量配置。
- 计划文档。
- 测试。
- 小型示例 JSON。
- 结果摘要，不包含大文件和敏感信息。

## 7.4 自动验证提交流程

每次代码修改后：

1. 运行验证。
2. 验证通过后提交。
3. 需要同步远端时 push。

推荐命令：

```bash
scripts/verify_and_commit.sh "commit message" python -m compileall food_agent scripts
git push
```

有测试时：

```bash
scripts/verify_and_commit.sh "commit message" pytest
git push
```

文档修改时：

```bash
scripts/verify_and_commit.sh "commit message" bash -n scripts/verify_and_commit.sh
git push
```

## 7.5 提交信息规范

建议格式：

- `Add dataset manifest builder`
- `Implement LightAgent wrapper`
- `Add VQA baseline runner`
- `Document experiment plan`
- `Fix event time alignment`

避免：

- `update`
- `fix`
- `test`
- `misc`

## 7.6 每次实施记录

每次阶段性实现后应该记录：

- 改了什么。
- 验证命令。
- 验证结果。
- 是否 push。
- 下一步。

## 7.7 大文件风险检查

提交前运行：

```bash
git status --short --ignored
git diff --cached --stat
```

如果看到以下内容进入 staged，必须停止：

- `data/`
- `annotations/`
- `.secrets/`
- `outputs/`
- `.mp4`
- `.hdf5`
- `.duckdb`
- `.parquet`
- 大型 `.gz`

## 7.8 远程同步策略

默认：

- 验证通过后 commit。
- 用户要求或阶段完成后 push。

如果 push 失败：

- 保留本地 commit。
- 检查 SSH 认证。
- 检查代理。
- 不重置本地分支。

