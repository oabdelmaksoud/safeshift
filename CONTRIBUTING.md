# Contributing to SafeShift

SafeShift is an open, vendor-neutral effort to bring shift-left validation to automotive
software. Contributions are welcome from OEMs, suppliers, researchers, and individual engineers.

## Ways to contribute
- **Schema extensions:** richer architecture descriptions (e.g., ARXML import, AUTOSAR mappings).
- **Risk features:** additional, explainable features grounded in integration experience.
- **Validation data:** representative (non-proprietary) architectures and labeled outcomes.
- **Reports/integrations:** CI plugins, dashboards, exporters.

## Ground rules
- Do not contribute proprietary or confidential employer data or code.
- Keep safety-relevant logic explainable; prefer transparent features over opaque models.
- Add tests for new behavior (`pytest`).

## Dev setup
```bash
pip install -e ".[dev]"
pytest
```
