# SafeShift Risk Report — Connected-Vehicle / Software-Defined-Vehicle E-E Architecture (illustrative)

- Components: 14  |  Interfaces: 19
- Model mode: **learned**
- Interfaces flagged HIGH risk: **12**

## Ranked integration-risk hotspots

| Rank | Interface | From → To | Protocol | Risk | Band |
|-----:|-----------|-----------|----------|-----:|------|
| 1 | if_cam_fusion | cam → fusion | Ethernet | 0.92 | HIGH |
| 2 | if_radar_fusion | radar → fusion | CAN-FD | 0.90 | HIGH |
| 3 | if_v2x_cgw | v2x → cgw | Ethernet | 0.89 | HIGH |
| 4 | if_fusion_adas | fusion → adas_dc | Ethernet | 0.84 | HIGH |
| 5 | if_cgw_adas | cgw → adas_dc | Ethernet | 0.83 | HIGH |
| 6 | if_adas_fusion | adas_dc → fusion | Ethernet | 0.81 | HIGH |
| 7 | if_ota_adas | ota → adas_dc | Ethernet | 0.81 | HIGH |
| 8 | if_vcu_brake | vcu → brake | CAN-FD | 0.74 | HIGH |
| 9 | if_ota_vcu | ota → vcu | CAN-FD | 0.74 | HIGH |
| 10 | if_adas_brake | adas_dc → brake | FlexRay | 0.73 | HIGH |

## Why these were flagged

- **if_cam_fusion** (risk 0.92): safety-related, timing-critical, crosses a supplier boundary, target sits in a dependency cycle, higher-complexity protocol, involves an immature component, high ASIL safety level.
- **if_radar_fusion** (risk 0.90): safety-related, timing-critical, crosses a supplier boundary, target sits in a dependency cycle, involves an immature component, high ASIL safety level.
- **if_v2x_cgw** (risk 0.89): safety-related, timing-critical, crosses a supplier boundary, target sits in a dependency cycle, higher-complexity protocol.
- **if_fusion_adas** (risk 0.84): safety-related, timing-critical, target sits in a dependency cycle, higher-complexity protocol, involves an immature component, high ASIL safety level.
- **if_cgw_adas** (risk 0.83): safety-related, timing-critical, target sits in a dependency cycle, higher-complexity protocol, high ASIL safety level.

_Scores are relative indicators of where integration review should be focused first; they are produced from the architecture description and (in learned mode) a synthetic training set. They are decision-support signals, not guarantees._