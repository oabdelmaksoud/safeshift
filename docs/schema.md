# Architecture Schema

SafeShift reads JSON or YAML. Two top-level lists: `components` and `interfaces`.

## Component
| Field | Type | Default | Notes |
|-------|------|---------|-------|
| id | string | — | unique identifier (required) |
| name | string | "" | human-readable name |
| kind | string | "ecu" | ecu \| software_module \| sensor \| domain_controller \| gateway |
| supplier | string | "unknown" | used to detect supplier-boundary crossings |
| asil | string | "QM" | QM, A, B, C, D (ISO 26262) |
| maturity | float | 0.5 | 0 = new/unproven, 1 = mature/reused |

## Interface
| Field | Type | Default | Notes |
|-------|------|---------|-------|
| id | string | — | unique identifier (required) |
| source | string | — | component id |
| target | string | — | component id |
| protocol | string | "CAN" | CAN \| CAN-FD \| Ethernet \| FlexRay \| LIN \| SPI \| internal |
| signals | int | 1 | number of signals carried |
| safety_related | bool | false | |
| timing_critical | bool | false | |

See `examples/example_adas_architecture.yaml` for a complete, illustrative architecture.
