| Item                      | Smoke test v0 |
|---                        |---|
| Tasks                     | 2 |
| Real demos                | 20–50 / task |
| Synthetic budget          | 10–20 / group |
| Groups |

redundant:
- initial state 与已有 trajectory 很接近
- object pose / approach path 高度重复

diverse:
- 明显增加 object pose / trajectory / initial-state coverage

policy_hard:
- base policy 在对应 initial condition 上实际失败或低成功率

corrupted:
- controlled artificial corruption

random:
- seeded random sampling


| Policy                    | lightweight BC / Diffusion Policy |
| Seeds                     | 3 |
| Metric                    | success rate |
| Main comparison           | valid-diverse vs valid-redundant |
| Key question              | `Validity != Utility?` |
