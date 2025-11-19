# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2025-11-19
### Added
- Initial release of `megaman2_bot.lua`.
- Q-learning agent for Mega Man 2 on FCEUX.
- Per-room Q-tables to allow room-specific learning.
- Blocked detection: penalizes Mega Man when stuck and increases exploration.
- Reward shaping based on horizontal progress, time penalty, and health loss.
- GUI overlay showing frame, episode, room ID, action, state, reward, and exploration/exploitation graph.
- Logging to `m2_qlog.csv` for analysis.
- Automatic save/load of Q-tables per room.

### Changed
- N/A

### Fixed
- N/A
