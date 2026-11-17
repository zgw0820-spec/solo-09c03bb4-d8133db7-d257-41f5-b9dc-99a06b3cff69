# 风机停机预警训练平台（可续跑、可撤销）

用 FastAPI + NumPy + SQLite 实现的风机停机预警逻辑回归训练平台。训练由两个工作器
容器执行，每个批次都把权重、动量、损失和进度作为**一个原子断点**提交；工作器退出后
另一个工作器自动接管，并从最近断点继续。最终权重与不中断训练逐位一致。

发现样本标签错误时，可通过 `POST /jobs/{job_id}/cancel` 撤销排队或运行中的任务：
已确认断点保留供排查，进度/权重/损失永久冻结，工作器释放租约并转去处理下一任务。

## 组成

| 组件 | 说明 |
| --- | --- |
| `api` | FastAPI，提交任务、查询进度/断点/权重、Swagger 演示 |
| `worker-1` / `worker-2` | 同一镜像的 `python -m app.worker`，共享同一个 SQLite 文件 |
| `training-data` | Docker 卷，挂载到三个容器的 `/data` |

两个工作器共享**同一个** SQLite 数据库（`/data/training.db`），因此"检查租约并写入
断点"和"写入断点"是同一个 `BEGIN IMMEDIATE` 事务，不需要跨库协调。

## 训练约定

- 输入 8～64 条记录，每条是风机振动/电流等特征，标签 `0`=正常、`1`=停机预警；
  各特征列按提交样本标准化后追加偏置列，权重长度为 `特征数 + 1`。
- 轮数 2～5；按输入顺序每 8 条一批（最后一批可不足 8 条）。
- 逻辑回归（sigmoid + 二元交叉熵），SGD，学习率 `0.1`，动量 `0.9`，float64。
- 每批提交的断点包含：`step`、`training_epoch`、`batch_in_epoch`、`weights`、
  `momentum_velocity`、`batch_loss`、`loss`（全量交叉熵）、`progress`、提交时间、
  `worker_id`、`lease_epoch`。

## 启动

```bash
docker compose up -d --build
# Swagger: http://localhost:8000/docs
```

`worker-1` / `worker-2` 故意不设 `restart` 策略，便于演示接管。被 kill 的工作器用
`docker compose start worker-1` 恢复（它会以空闲状态重新加入，不会抢占已完成任务）。

## API

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `POST` | `/jobs` | 提交 `{"features": [[...]], "labels": [...], "epochs": 3}`，返回 201 与任务 |
| `GET` | `/jobs` | 最近任务列表 |
| `GET` | `/jobs/{job_id}` | 已确认步数、进度、损失、当前权重；完成时含 `final_weights` |
| `POST` | `/jobs/{job_id}/cancel` | 撤销排队/运行中的任务（请求体可带 `reason`），冻结终态并释放租约 |
| `GET` | `/jobs/{job_id}/checkpoints` | 全部批次断点（可按 `after_step` 增量拉取） |
| `GET` | `/jobs/{job_id}/events` | 审计事件：创建、开始/从第几步续训、完成、撤销、丢失租约 |
| `GET` | `/workers` | 工作器心跳与租约持有者 |
| `GET` | `/health` | 健康检查 |

`Swagger`（`/docs`）里直接 `POST /jobs` 提交即可看到训练开始；`GET /jobs/{id}` 与
`GET /jobs/{id}/checkpoints` 查看进度与每个断点的权重/动量/损失。

## 续跑与防回退机制

1. 工作器每批在**同一个事务**里写检查点、更新任务进度、刷新租约心跳。任何一项失败
   则整体回滚，不会出现"权重已变但进度没变"的中间态。
2. 租约保存在同一数据库的 `leases` 表。工作器每 `1.5s` 心跳；持有者心跳超过
   `6s`（`TRAINING_LEASE_SECONDS`）未更新，另一个工作器才可接管，接管时
   `epoch + 1`。
3. 每次提交前在事务内校验 `(worker_id, epoch)` 仍是租约持有者。被接管后仍存活（或
   迟到唤醒）的旧工作器校验失败，直接 `LeaseLost` 退出：既不能回退断点，也不能把
   别人的完成状态改写，更不能重复提交完成事件。
4. 完成状态由 `step == total_steps` 在提交事务内一次性置位；旧工作器无法再写入。
5. SQLite 使用 WAL + `synchronous=FULL`，已提交的断点在进程崩溃/容器重启后仍然存在。

## 撤销（取消）语义

任务状态机为 `queued → running → completed | cancelled`，`completed` 与 `cancelled`
都是终态。`POST /jobs/{job_id}/cancel` 在一个 `BEGIN IMMEDIATE` 事务内完成所有
判定与写入，与批次提交、完成提交互斥，因此交错时只有一个确定终态：

1. **已完成则拒绝撤销**：对 `completed` 任务调用返回 `409`，完成结果保持不变。
2. **首次撤销成功**：任务置为 `cancelled`，写入 `job_cancelled` 审计事件（含当时
   已确认步数与可选 `reason`），并在同一事务内删除租约行。返回体带
   `already_cancelled=false`、`cancel_reason`、`cancelled_at`。
3. **断点保留、状态冻结**：已确认的检查点、权重、动量与损失原样保留用于排查；之后
   进度、权重与损失不再变化，`final_weights` 始终为 `null`，`GET /jobs/{id}` 与
   `GET /jobs/{id}/checkpoints` 可随时复盘。
4. **工作器释放租约并接单**：租约删除后，持有者的下一次续约/提交围栏失败，工作器
   记录 `training_aborted` 事件、清理本地租约，主循环立即轮询并处理下一任务。
5. **迟到工作器围栏**：`advance_job` 在事务内先检查 `cancelled` 再校验租约 epoch，
   迟到或被暂停后唤醒的旧工作器只能得到 `JobCancelled`/`LeaseLost`，无法提交断点，
   也无法把任务改回 `running`（进度更新的 `WHERE status IN ('queued','running')`
   条件不再匹配）。
6. **重复撤销幂等**：对已撤销任务再次调用不写任何数据，返回同一终态且
   `already_cancelled=true`，首次撤销的 `reason`、`finished_at` 保持不变。
7. **重启后仍不可续跑**：撤销状态持久化在 SQLite；`lease.acquire` 对
   `completed/cancelled` 任务一律返回 `None`，工作器故障或 API 重启后都不会接管
   已撤销任务，而其他任务的接管续训不受影响。

`Swagger`（`/docs`）的 `training` 标签下可直接试用 `POST /jobs/{job_id}/cancel`。

## 演示与验证

```bash
# 完整演示（Docker）：提交 -> 观察 -> kill 当前持有者 -> 接管续跑 -> 对比不中断训练
python3 scripts/demo.py

# 没有 Docker 时用本机进程演示同样流程
python3 scripts/demo.py --local

# 不依赖 Docker 的四项自检（compose 结构、崩溃接管、迟到围栏、撤销）
python3 scripts/selftest.py

# 单元/集成测试（含真实 uvicorn + 两个工作器进程、SIGKILL 接管、迟到工作器围栏）
python3 -m pytest -q
```

演示脚本第 6 步会用同一份输入再训练一次（不中断），并断言两次的 `final_weights`、
`loss`、`momentum_velocity` 完全相等。

## 环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `TRAINING_DATA_DIR` | `./data` | 数据库目录 |
| `TRAINING_DB_PATH` | `$TRAINING_DATA_DIR/training.db` | 共享数据库 |
| `TRAINING_LEASE_SECONDS` | `6` | 租约超时 |
| `TRAINING_HEARTBEAT_INTERVAL` | `1.0` | 心跳间隔（必须小于租约） |
| `TRAINING_POLL_INTERVAL` | `0.3` | 空闲轮询/批间间隔 |
| `TRAINING_BATCH_DELAY_SECONDS` | `0` | 模拟每批耗时，便于演示 |
| `TRAINING_DEMO_PAUSE_FILE` | 未设置 | 演示用：文件存在时在批间暂停（不心跳） |
| `TRAINING_DEMO_CRASH_AFTER` | 未设置 | 演示用：指定步数提交后 `os._exit` |

## 目录

```
app/main.py       API 与 Swagger
app/worker.py     工作器主循环、租约续训
app/store.py      任务状态机、批次提交、断点查询
app/lease.py      租约获取/续约/围栏校验
app/model.py      逻辑回归、交叉熵、SGD + 动量
app/features.py   输入校验与标准化
app/db.py         SQLite 连接与表结构
tests/            单元、API、进程级集成测试
scripts/demo.py   Swagger 同款接口的端到端演示
scripts/selftest.py  无 Docker 自检
```
