# 夾取任務：怎麼訓練

`Isaac-Grasp-Ur5Robotiq-v0`：在圓柱之間把立方體夾起來。設計見 [grasp_rl_plan.md](grasp_rl_plan.md)，這份只講**怎麼跑、跑出來是什麼、還缺什麼**。

狀態（2026-09-24）：**訓練模式可以開始訓練**；評估模式（開 mapper、真的感知）還沒做。

## 一步是什麼

一個 env step = 每個環境做**一次夾取嘗試**，同步進行：

1. policy 給 `[dx, dy, dz, yaw]`（都在 [−1, 1]），相對於**感知估計**的立方體中心，世界方向。
2. 規劃到上方 → 直線下降（server 檢查碰撞）→ 夾 → 直線抬起（檢查）→ 停 30 步。
3. 抬高 ≥ 75 mm 且夾爪讀數 < 0.60 rad（夾住時 0.18–0.46、夾空 0.78–0.80）= 成功，episode 結束。
4. **一個 episode 只有一次嘗試**（`max_attempts=1`，2026-09-24 定案）。調高 `max_attempts` 時，失敗後會張開、沿原路退回預抓取點（`Retrace`）、規劃回 HOME、重新感知，再試下一次。
   規劃回 HOME 失敗時依序改用：規劃到 HOME 的工具位姿再補關節、直線上抬越過圓柱再回 HOME；都失敗才瞬移回去並記錄（`stuck_total`）。每一層都是隨機動作的壓力測試裡碰到過、才加上的。

觀測：actor 29 維（實機拿得到的），critic 45 維（加上真值）。reward：成功 +1、每次嘗試 −0.05、動作失敗 −0.1、碰到圓柱 −0.3、把立方體推走 −0.5。全部在 `scripts/grasp/task.py` 的 `TaskCfg`。

## 訓練模式在做什麼

- **planner 被告知每個環境的圓柱**（`op "world"`，圓柱以 70 × 70 mm 外接長方體給 planner）。
- **不 render**：沒有相機。policy 看到的立方體與圓柱是**真值加誤差模型**（位置 σ 2 mm、yaw σ 2°，被擋住越多越大；圓柱 2% 漏看、2% 誤判）。
- 配置來自**可行配置庫**（`scripts/grasp/layouts/train.npz`），每一組都事先用 cuRobo 確認至少有一種夾法可行；**全部從「只有一種夾法可行」的配置抽**（`one_grip_fraction=1.0`，訓練庫 539 組、評估庫 64 組）。
- 圓柱在物理裡是 kinematic 碰撞體，有接觸感測器。

## 先跑的檢查（都會自己起 planner server）

```bash
python tools/check_lab_modularity.py        # 相依規則、註冊、三個場景都生得起來
python tools/check_grasp_lab.py             # 空桌面物理夾取 20 次
python tools/check_grasp_env.py             # gym env + oracle：成功率、每步時間、rsl_rl wrapper
python tools/check_grasp_env.py --policy random --episodes 30   # 跑遍失敗路徑
```

## 訓練

```bash
python scripts/grasp/isaaclab_train.py --smoke                    # 2 envs、3 次迭代：能不能跑
python scripts/grasp/isaaclab_train.py --num_envs 8 --run-name first
python scripts/grasp/isaaclab_train.py --num_envs 16 --num-servers 8 --max-iterations 1000
tensorboard --logdir logs/rsl_rl/grasp
```

- 物理預設在 CPU（這個規模比 GPU 快，plan §9），網路在 `--rl-device cuda:0`。
- 預設每個環境一個 planner server（每個約 2 GB RAM、0.55 GB GPU）；`--num-servers` 可以比環境少，server 會在請求之間換世界。
- checkpoint 與 tensorboard 在 `logs/rsl_rl/grasp/<時間>_<run-name>/`，每 25 次迭代存一次；`--resume <model_N.pt>` 接著訓。
- tensorboard 裡除了 reward，還有 `episode/success`、`attempt/<結果>`（規劃不到、IK 失敗、直線移動被拒、夾空、碰到圓柱）、`stuck_total`（回 HOME 失敗、只好瞬移回去的次數，應該一直是 0）。
- PPO 設定在 `scripts/grasp/lab_rl_cfg.py`：episode 最多 3 步，所以 `gamma` 0.9；每次迭代每環境 8 步、8 個 epoch。**沒調過**，是起點。

## 量測（2026-09-24）

**配置庫**（`tools/grasp_layout_bank.py`，每根圓柱 60% 機率擺在立方體 15–120 mm 內）：

| | 訓練庫 | 評估庫 |
|---|---|---|
| 產生 / 嘗試 | 3000 / 6082 | 300 / 571 |
| 有 15 mm 餘量的可用配置 | 2575 | 259 |
| 其中只有一種夾法可行 | 539 | 64 |
| 圓柱數 0 / 1 / 2 / 3 / 4 | 1197 / 806 / 506 / 302 / 189 | 120 / 73 / 52 / 35 / 20 |

**現在的設定下的基準**（一次機會、只用單一夾法可行的配置；評估庫，8 個環境，有誤差模型）：

| | 成功 | 碰到圓柱 |
|---|---|---|
| oracle（知道哪種夾法可行，對準估計中心） | 63/64（98%） | 0 |
| 均勻隨機動作 | 29/64（45%） | 1 |

這個差距（45% → 98%）就是 policy 要學的：從感知到的圓柱位置判斷哪一組面夾得到，再對準。oracle 是靠配置庫知道答案的，policy 沒有這個資訊。

**先前的設定下的基準**（三次機會、一半 episode 抽單一夾法配置；評估庫，8 個環境）：

| | episode 成功（≤ 3 次） | 第一次就成功 | 碰到圓柱 | 回 HOME 失敗 |
|---|---|---|---|---|
| oracle（知道哪種夾法可行，對準估計中心） | 55/55（100%） | 54/55（98%） | 0 | 0 |
| 均勻隨機動作 | 46/48（96%） | 31/48（65%） | 0 | 0 |

只用「只有一種夾法可行」的配置（訓練庫，`one_grip_fraction=1.0`）：

| | episode 成功（≤ 3 次） | 第一次就成功 |
|---|---|---|
| oracle | 51/52（98%） | 50/52（96%） |
| 均勻隨機動作 | 38/48（79%） | 24/48（50%） |

**吞吐量**：8 個環境、CPU 物理，每個 env step 約 2.8–3.6 秒，**約 2.5 次嘗試/秒**。一次 PPO 迭代（8 步 × 8 環境 = 64 次嘗試）約 25 秒；500 次迭代約 3.5 小時。

**第一次訓練**（40 次迭代、2560 次嘗試、17 分鐘）：reward 0.80 → 0.82 左右、單次嘗試成功率 0.6–0.77，雜訊大，**還看不出有沒有在學**；動作標準差 0.50 → 0.48。

### 為什麼改成一次機會、只用單一夾法可行的配置（已定案）

原本的設定下，隨機動作在 3 次內就有 96% 成功，第一次 65%。原因是 45 mm 的立方體斜著夾也夾得住（夾住時夾爪停在 0.18–0.46 rad 都算）。所以 episode 成功率幾乎沒有空間，能學的只剩「第一次就成功」（65% → oracle 的 98%），reward 的差距只有約 0.1–0.2，訊號很弱。三個方向：

1. **一次機會**（`max_attempts=1`）：成功率就是第一次成功率，隨機 65% vs oracle 98%，差距直接變成 reward。實機上也比較像「一次夾好」。
2. **只用只有一種夾法可行的配置**（`one_grip_fraction=1.0`）：選錯面就規劃不到或被拒。量到的：隨機第一次成功 50%、oracle 96%。
3. **更嚴的成功判定**：斜著夾在模擬裡夾得住，**實機上 2F-85 夾在立方體的邊上很可能會滑**。這是 sim-to-real 的風險：policy 可能學會實機上不可靠的夾法。可以要求夾取角度接近某一組面，或抬起後搖晃一下仍然拿著。

**定案：1 + 2**（上面的「現在的設定」）。3 在上實機前處理。

## 還沒做、大規模訓練前要知道的

1. **評估模式**：開 mapper、用真的感知模組（`grasp/perception.py`，plan §4）量成功率。現在 `mapping=True` 會直接報錯，不會假裝可以。
2. **直線移動的碰撞檢查只查 planner 被告知的東西**：訓練模式下就是全部；評估模式下還要查地圖（ESDF），目前只查桌子。
3. **`MoveTo` 不確認有沒有到達**：被擋住時（例如評估時地圖漏了圓柱）仍回報成功。訓練模式下 planner 知道圓柱，不會發生。
4. **規模化（plan §9，M5）**：批次規劃（`BatchMotionPlanner`）、每環境 Python 迴圈向量化、GPU PhysX 基準測試。
5. **誤差模型是初值**，要用評估模式和實機量到的感知誤差校正。
6. **回 HOME 偶爾會卡住**（只在 `max_attempts` > 1 時；一次機會時每個 episode 結束就 reset）：隨機動作的壓力測試裡約每 500 次嘗試 1 次，都在很窄的配置（例如預抓取點離圓柱只有 11 mm）：直線上抬會更靠近圓柱、兩種規劃回 HOME 都失敗。模擬裡會瞬移回去並計入 `stuck_total`；實機上這就是需要人處理的情況。oracle 從沒卡過。
