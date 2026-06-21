# SafeShift Risk Report — Reference ADAS / E-E Architecture (illustrative)

- Components: 11  |  Interfaces: 12
- Model mode: **learned**
- Interfaces flagged HIGH risk: **7**

## Ranked integration-risk hotspots

| Rank | Interface | From → To | Protocol | Risk | Band |
|-----:|-----------|-----------|----------|-----:|------|
| 1 | if_vpm_fusion | vpm → fusion | Ethernet | 0.92 | HIGH |
| 2 | if_rf_fusion | radar_front → fusion | CAN-FD | 0.90 | HIGH |
| 3 | if_rc_fusion | radar_corner → fusion | CAN-FD | 0.84 | HIGH |
| 4 | if_adc_fusion | adc → fusion | Ethernet | 0.83 | HIGH |
| 5 | if_fusion_adc | fusion → adc | Ethernet | 0.82 | HIGH |
| 6 | if_adc_brake | adc → brake | FlexRay | 0.75 | HIGH |
| 7 | if_adc_steer | adc → steer | FlexRay | 0.75 | HIGH |
| 8 | if_cam_vpm | cam_front → vpm | Ethernet | 0.58 | MEDIUM |
| 9 | if_pam_adc | pam → adc | CAN | 0.37 | MEDIUM |
| 10 | if_adc_gw | adc → gw | Ethernet | 0.27 | LOW |

## Why these were flagged

- **if_vpm_fusion** (risk 0.92): safety-related, timing-critical, crosses a supplier boundary, target sits in a dependency cycle, higher-complexity protocol, involves an immature component, high ASIL safety level.
- **if_rf_fusion** (risk 0.90): safety-related, timing-critical, crosses a supplier boundary, target sits in a dependency cycle, involves an immature component, high ASIL safety level.
- **if_rc_fusion** (risk 0.84): safety-related, crosses a supplier boundary, target sits in a dependency cycle, involves an immature component, high ASIL safety level.
- **if_adc_fusion** (risk 0.83): safety-related, timing-critical, target sits in a dependency cycle, higher-complexity protocol, involves an immature component, high ASIL safety level.
- **if_fusion_adc** (risk 0.82): safety-related, timing-critical, target sits in a dependency cycle, higher-complexity protocol, involves an immature component, high ASIL safety level.

_Scores are relative indicators of where integration review should be focused first; they are produced from the architecture description and (in learned mode) a synthetic training set. They are decision-support signals, not guarantees._