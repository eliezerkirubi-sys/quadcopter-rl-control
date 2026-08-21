# feat(env): add survival bonus and terminal condition rules (#8)

## Summary
Resolves #8 by implementing episode termination bounds (ground crash, altitude breach, horizontal drift, and excessive tilt) alongside a positive constant survival bonus and crash penalties in `QuadcopterHoverEnv`.

### Changes
- Implemented `QuadcopterHoverEnv` in `src/custom_hover_env.py` with configurable bounds (`min_z`, `max_z`, `max_xy_distance`, `max_tilt_angle`).
- Implemented `_computeTerminated()` logic for crash and out-of-bounds detection.
- Added constant positive survival bonus and attitude/rate stabilization rewards.
- Added comprehensive unit tests in `tests/test_survival_and_termination.py`.

Closes #8
