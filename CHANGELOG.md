# Changelog

All notable changes to this project will be documented in this file.

## [1.1.0] - 2025-11-19
### Added
- Ladder-aware actions: Can now climb up and down ladders.
- Action cooldown to reduce input spamming for smoother movements.
- Universal blocked detection: works across all rooms.

### Changed
- Reward system updated:
  - Progress scaled to 100
  - Time penalty set to -10
  - Death penalty remains -50

### Fixed
- Bot now attempts vertical movement when ladders are present.

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
