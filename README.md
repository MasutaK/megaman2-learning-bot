# Mega Man 2 Q-Learning Bot for FCEUX

This repository contains a Lua script `megaman2_bot.lua` that implements a Q-learning agent to play **Mega Man 2** on the FCEUX emulator. The bot learns per-room strategies and adapts to blocked situations while tracking progress with a simple overlay.

---

## Features

- **Per-room Q-tables:** The agent maintains separate Q-tables for each room, allowing it to learn room-specific strategies independently.  
- **Q-learning:** Implements tabular Q-learning with epsilon-greedy exploration, learning from rewards shaped by progress, time, and health loss.  
- **Blocked detection:** If Mega Man is stuck (e.g., colliding against walls or not moving), the bot applies a penalty and increases exploration to find a better path.  
- **Progress visualization:** Displays frame number, episode, current room, action, state, reward, and a mini exploration/exploitation graph on the FCEUX GUI overlay.  
- **Logging:** Saves frame-by-frame logs to `m2_qlog.csv` for later analysis.  
- **Automatic save/load:** Periodically saves Q-tables for each room, allowing learning progress to persist across sessions.

---

## Configuration

The script uses the following main parameters:

- `ALPHA` – learning rate (default: 0.2)  
- `GAMMA` – discount factor for future rewards (default: 0.95)  
- `EPSILON` – initial exploration rate (default: 0.6)  
- `EPS_DECAY` – per-frame decay of epsilon (default: 0.99995)  
- `EPS_MIN` – minimum epsilon (default: 0.05)  
- `TIME_PENALTY` – small negative reward per frame to encourage faster progress  
- `DEATH_PENALTY` – large negative reward when Mega Man loses health  
- `BLOCKED_CHECK_FRAMES` – number of frames to detect if Mega Man is stuck  

The script discretizes state space using `X_BUCKETS`, `Y_BUCKETS`, and `HP_BUCKETS` to reduce the Q-table size.

---

## How It Works

1. **Initialization:**  
   The bot loads or creates a Q-table for the current room and records the initial state using FCEUX `savestate`.

2. **Main Loop:**  
   - Reads the current position, health, and room.  
   - Detects if Mega Man dies; if so, reloads the start state of the room.  
   - Chooses an action using epsilon-greedy strategy.  
   - Advances one frame with `joypad.set()` and `emu.frameadvance()`.  
   - Observes the next state and computes a reward based on:
     - Horizontal progress (`delta_x`)  
     - Time penalty  
     - Death penalty  
   - Detects if Mega Man is blocked; if yes, penalizes reward and forces full exploration.  
   - Updates the Q-table and decays epsilon.  
   - Periodically logs data and saves Q-tables.  
   - Updates the FCEUX GUI overlay with the current episode, room, action, reward, and exploration graph.

3. **Blocked detection:**  
   Tracks the last `BLOCKED_CHECK_FRAMES` positions and penalizes Mega Man if he is not moving enough, increasing exploration to find a better path.

---

## Usage

1. Open **FCEUX** and load Mega Man 2 ROM.  
2. Go to `File → Lua → New Lua Script` and select `megaman2_bot.lua`.  
3. Run the script. You will see:
   - Overlay with progress and current state
   - Automatic Q-table saving per room
   - Exploration/exploitation mini graph

---

## Files

- `megaman2_bot.lua` – main bot script  
- `mm2_qtable_room_<room_id>.csv` – per-room Q-tables saved automatically  
- `m2_qlog.csv` – log of frames, states, actions, rewards, exploration, and episodes

---

## Notes
 
- The script is designed for FCEUX and may not work on other NES emulators.  
- Adjust learning parameters to improve training speed and stability.

---

## License

This project is open source and can be freely used and modified. No warranty is provided.
