-- megaman2_bot.lua
-- Q-learning agent for Mega Man 2 (FCEUX) with per-room Q-tables, progress graph, and blocked detection

-- ========== CONFIG ==========
local Q_SAVE_PREFIX = "mm2_qtable_room_"
local LOG_FILE = "mm2_qlog.csv"

-- RAM addresses
local ADDR_X = 0x0460
local ADDR_Y = 0x04A0
local ADDR_HP = 0x06C0
local ADDR_SCROLL_STAT = 0x001E

-- learning params
local ALPHA = 0.2
local GAMMA = 0.95
local EPSILON = 0.6
local EPS_DECAY = 0.99995
local EPS_MIN = 0.05

-- state discretization
local X_BUCKETS = 16
local Y_BUCKETS = 8
local HP_BUCKETS = 4

-- reward shaping
local TIME_PENALTY = 0.01
local DEATH_PENALTY = -50
local PROGRESS_SCALE = 1.0

-- save / log frequency
local SAVE_EVERY = 5000
local LOG_EVERY = 1200
local BLOCKED_CHECK_FRAMES = 15
local BLOCKED_DELTA_THRESHOLD = 1

-- actions
local ACTIONS = {
    {name="NONE", map={}},
    {name="RIGHT", map={["right"]=true}},
    {name="LEFT",  map={["left"]=true}},
    {name="JUMP",  map={["A"]=true}},
    {name="SHOOT", map={["B"]=true}},
    {name="JUMP_RIGHT", map={["A"]=true, ["right"]=true}},
}

-- ----------------------------------------
-- STATE & RESET
-- ----------------------------------------
local MIN_LIFE = 1
local death_wait_frames = 0
local episode = 1

local room_start_state = savestate.create()
savestate.save(room_start_state)

-- ----------------------------------------
-- HELPERS
-- ----------------------------------------
local function get_bucket(val, maxval, buckets)
    local b = math.floor((val / (maxval + 1)) * buckets)
    if b < 0 then b = 0 end
    if b >= buckets then b = buckets - 1 end
    return b
end
local function r(addr) return memory.readbyte(addr) or 0 end
local function state_to_key(xb, yb, hpb, sstat)
    return tostring(xb) .. "_" .. tostring(yb) .. "_" .. tostring(hpb) .. "_" .. tostring(sstat)
end

-- ----------------------------------------
-- Q-tables per room
-- ----------------------------------------
local Q_rooms = {}

local function save_qtable_room(room_id)
    local Q = Q_rooms[room_id]
    if not Q then return end
    local f = io.open(Q_SAVE_PREFIX .. room_id .. ".csv", "w")
    if not f then return end
    for k, tab in pairs(Q) do
        for ai, qv in pairs(tab) do
            f:write(string.format("%s,%d,%.6f\n", k, ai, qv))
        end
    end
    f:close()
end

local function load_qtable_room(room_id)
    local f = io.open(Q_SAVE_PREFIX .. room_id .. ".csv", "r")
    Q_rooms[room_id] = Q_rooms[room_id] or {}
    local Q = Q_rooms[room_id]
    if not f then return end
    for line in f:lines() do
        local k, ai, qv = line:match("([^,]+),([^,]+),([^,]+)")
        if k and ai and qv then
            ai = tonumber(ai)
            qv = tonumber(qv)
            Q[k] = Q[k] or {}
            Q[k][ai] = qv
        end
    end
    f:close()
end

-- ----------------------------------------
-- Q-learning helpers
-- ----------------------------------------
local function choose_action(key, epsilon, room_id)
    local Q = Q_rooms[room_id]
    local tab = Q[key]
    if math.random() < epsilon or tab == nil then
        return math.random(1, #ACTIONS), true
    end
    local best_ai = 1
    local best_q = -1e9
    for ai = 1, #ACTIONS do
        local qv = tab[ai] or 0
        if qv > best_q then
            best_q = qv
            best_ai = ai
        end
    end
    return best_ai, false
end

local function max_q(key, room_id)
    local Q = Q_rooms[room_id]
    local tab = Q[key]
    if tab == nil then return 0 end
    local best = -1e9
    for ai = 1, #ACTIONS do
        local qv = tab[ai] or 0
        if qv > best then best = qv end
    end
    if best < -1e8 then return 0 end
    return best
end

local function update_q(s_key, ai, reward, s2_key, room_id)
    local Q = Q_rooms[room_id]
    Q[s_key] = Q[s_key] or {}
    local q = Q[s_key][ai] or 0
    local target = reward + GAMMA * max_q(s2_key, room_id)
    local newq = q + ALPHA * (target - q)
    Q[s_key][ai] = newq
end

-- ----------------------------------------
-- Logging
-- ----------------------------------------
local logf = io.open(LOG_FILE, "a")
if logf then logf:write("frame,xb,yb,hb,sstat,action,reward,epsilon,episode,room,explore\n") end

-- ----------------------------------------
-- State helpers
-- ----------------------------------------
local frame = 0
local eps = EPSILON
local function get_state_key()
    local x = r(ADDR_X)
    local y = r(ADDR_Y)
    local hp = r(ADDR_HP)
    local sstat = r(ADDR_SCROLL_STAT)
    local xb = get_bucket(x,255,X_BUCKETS)
    local yb = get_bucket(y,255,Y_BUCKETS)
    local hpb = get_bucket(hp,255,HP_BUCKETS)
    return state_to_key(xb,yb,hpb,sstat), xb, yb, hpb, sstat, x, y, hp
end

-- ----------------------------------------
-- INITIAL STATE
-- ----------------------------------------
local current_action = 1
local explore = false
local s_key, xb, yb, hpb, sstat, cur_x, cur_y, cur_hp = get_state_key()
local cur_room = r(ADDR_SCROLL_STAT)
Q_rooms[cur_room] = Q_rooms[cur_room] or {}
load_qtable_room(cur_room)
current_action, explore = choose_action(s_key, eps, cur_room)

-- ----------------------------------------
-- BLOCKED DETECTION
-- ----------------------------------------
local last_positions = {}
local function is_blocked(nx, ny)
    table.insert(last_positions, {x=nx, y=ny})
    if #last_positions > BLOCKED_CHECK_FRAMES then
        table.remove(last_positions, 1)
    end
    local min_x, max_x = last_positions[1].x, last_positions[1].x
    local min_y, max_y = last_positions[1].y, last_positions[1].y
    for _, pos in ipairs(last_positions) do
        if pos.x < min_x then min_x = pos.x end
        if pos.x > max_x then max_x = pos.x end
        if pos.y < min_y then min_y = pos.y end
        if pos.y > max_y then max_y = pos.y end
    end
    local dx = max_x - min_x
    local dy = max_y - min_y
    return (dx <= BLOCKED_DELTA_THRESHOLD and dy <= BLOCKED_DELTA_THRESHOLD)
end

-- ----------------------------------------
-- MAIN LOOP
-- ----------------------------------------
while true do
    frame = frame + 1

    -- detect room change
    local new_room = r(ADDR_SCROLL_STAT)
    if new_room ~= cur_room then
        cur_room = new_room
        Q_rooms[cur_room] = Q_rooms[cur_room] or {}
        load_qtable_room(cur_room)
        room_start_state = savestate.create()
        savestate.save(room_start_state)
        last_positions = {}
        print(string.format(">>> New room detected. Room ID: %d", cur_room))
    end

    -- detect death
    local hp = r(ADDR_HP)
    if hp <= MIN_LIFE then
        death_wait_frames = death_wait_frames + 1
        if death_wait_frames > 20 then
            episode = episode + 1
            print(string.format(">>> Mega Man died. Resetting to start of room (episode #%d)", episode))
            savestate.load(room_start_state)
            emu.frameadvance()
            death_wait_frames = 0
            s_key, xb, yb, hpb, sstat, cur_x, cur_y, cur_hp = get_state_key()
            current_action, explore = choose_action(s_key, eps, cur_room)
            last_positions = {}
        end
    else
        death_wait_frames = 0

        -- execute action
        local act = ACTIONS[current_action]
        joypad.set(1, act.map)
        emu.frameadvance()

        -- next state
        local s2_key, xb2, yb2, hpb2, sstat2, nx, ny, nhp = get_state_key()

        -- reward
        local delta_x = nx - cur_x
        if delta_x < -128 then delta_x = delta_x + 256 end
        local reward = PROGRESS_SCALE * delta_x - TIME_PENALTY
        if nhp < cur_hp then reward = reward + DEATH_PENALTY end

        -- blocked check
        if is_blocked(nx, ny) then
            reward = reward - 1.0
            current_action, explore = choose_action(s_key, 1.0, cur_room) -- full exploration
        else
            current_action, explore = choose_action(s2_key, eps, cur_room)
        end

        -- update Q
        update_q(s_key, current_action, reward, s2_key, cur_room)

        -- decay epsilon
        eps = math.max(EPS_MIN, eps * EPS_DECAY)

        -- log periodically
        if frame % LOG_EVERY == 0 and logf then
            logf:write(string.format("%d,%d,%d,%d,%d,%s,%.4f,%.4f,%d,%d,%d\n",
                frame, xb2, yb2, hpb2, sstat2, ACTIONS[current_action].name, reward, eps, episode, cur_room, explore and 1 or 0))
            logf:flush()
        end

        -- save all Q-tables periodically
        if frame % SAVE_EVERY == 0 then
            for room_id,_ in pairs(Q_rooms) do
                save_qtable_room(room_id)
            end
        end

        -- prepare next iteration
        s_key = s2_key
        cur_x = nx; cur_y = ny; cur_hp = nhp

        -- overlay graphics
        gui.text(6,10,string.format("MM2 Q-learning | frame:%d eps:%.3f", frame, eps))
        gui.text(6,24,string.format("Episode: %d", episode))
        gui.text(6,38,string.format("Room ID: %d", cur_room))
        gui.text(6,52,string.format("Action: %s", ACTIONS[current_action].name))
        gui.text(6,66,string.format("State xb:%d yb:%d hpb:%d sstat:%d", xb2, yb2, hpb2, sstat2))
        gui.text(6,82,string.format("Reward: %.3f", reward))

        -- mini graph for exploration vs exploitation
        local bar_width = 50
        local explore_len = math.floor(bar_width * (explore and 1 or 0))
        gui.box(6, 98, 6 + bar_width, 102, 0xFF000000, 0xFF4444FF)
        if explore then
            gui.box(6, 98, 6 + explore_len, 102, 0xFF4444FF, 0xFF4444FF)
        else
            gui.box(6, 98, 6 + bar_width, 102, 0xFF44FF44, 0xFF44FF44)
        end
    end
end

-- save on exit
for room_id,_ in pairs(Q_rooms) do
    save_qtable_room(room_id)
end
if logf then logf:close() end
