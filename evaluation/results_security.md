# SafeShift — External Attack-Surface Overlap (connected-vehicle example)

- Architecture: Connected-Vehicle / Software-Defined-Vehicle E-E Architecture (illustrative)
- Components: 14 | Interfaces: 19
- External entry points (off-board connectivity): ivi, ota, tcu, v2x
- Externally-reachable interfaces: **15 / 19**
- HIGH-risk interfaces: **12**; of those, externally reachable: **10 / 12**
- Of the top-10 integration-risk hotspots, externally reachable: **8 / 10**

| Rank | Interface | From → To | Risk | Band | Externally reachable (R155 surface) |
|-----:|-----------|-----------|-----:|------|:--:|
| 1 | if_cam_fusion | cam → fusion | 0.92 | HIGH | no |
| 2 | if_radar_fusion | radar → fusion | 0.90 | HIGH | no |
| 3 | if_v2x_cgw | v2x → cgw | 0.89 | HIGH | yes |
| 4 | if_fusion_adas | fusion → adas_dc | 0.84 | HIGH | yes |
| 5 | if_cgw_adas | cgw → adas_dc | 0.83 | HIGH | yes |
| 6 | if_adas_fusion | adas_dc → fusion | 0.81 | HIGH | yes |
| 7 | if_ota_adas | ota → adas_dc | 0.81 | HIGH | yes |
| 8 | if_vcu_brake | vcu → brake | 0.74 | HIGH | yes |
| 9 | if_ota_vcu | ota → vcu | 0.74 | HIGH | yes |
| 10 | if_adas_brake | adas_dc → brake | 0.73 | HIGH | yes |
| 11 | if_adas_steer | adas_dc → steer | 0.72 | HIGH | yes |
| 12 | if_cgw_vcu | cgw → vcu | 0.72 | HIGH | yes |
| 13 | if_hsm_cgw | hsm → cgw | 0.58 | MEDIUM | no |
| 14 | if_ivi_cgw | ivi → cgw | 0.56 | MEDIUM | yes |
| 15 | if_tcu_cgw | tcu → cgw | 0.56 | MEDIUM | yes |
| 16 | if_cgw_ivi | cgw → ivi | 0.50 | MEDIUM | yes |
| 17 | if_ota_cgw | ota → cgw | 0.39 | MEDIUM | yes |
| 18 | if_cgw_bcm | cgw → bcm | 0.21 | LOW | yes |
| 19 | if_hsm_ota | hsm → ota | 0.15 | LOW | no |

_An interface is 'externally reachable' if its source component is reachable, in the directed dependency graph, from an off-board connectivity entry point. These are the interfaces on the cyber attack-propagation surface that UNECE R155/R156 and ISO-SAE 21434 govern. The overlap with SafeShift's integration-risk hotspots is the point: the same interfaces concentrate integration risk and security exposure._