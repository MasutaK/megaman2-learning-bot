-- megaman2_bot.lua
-- Learning agent for Mega Man 2 (FCEUX)

-- ========== CONFIG ==========
local Q_SAVE_PREFIX = "mm2_qtable_room_"
local LOG_FILE = "mm2_qlog.csv"

-- RAM addresses
local ADDR_X      = 0x0460
local ADDR_Y      = 0x04A0
local ADDR_HP     = 0x06C0
local ADDR_SCROLL = 0x001E
local ADDR_LADDER = 0x04F8   -- 1 = on ladder

-- learning params
local ALPHA = 0.2
local GAMMA = 0.95
local EPSILON = 0.6
local EPS_DECAY = 0.99995
local EPS_MIN = 0.05

-- discretization
local X_BUCKETS = 16
local Y_BUCKETS = 8
local HP_BUCKETS = 4

-- reward shaping
local TIME_PENALTY = 10
local DEATH_PENALTY = -50
local PROGRESS_SCALE = 100

-- save/log frequency
local SAVE_EVERY = 5000
local LOG_EVERY = 1200
local BLOCKED_CHECK_FRAMES = 1000
local BLOCKED_DELTA_THRESHOLD = 1

-- action cooldown (important to avoid spamming inputs)
local ACTION_COOLDOWN = 6
local action_timer = 0
local last_action = 1

-- ============================================
-- ACTIONS (with ladder support)
-- ============================================
local ACTIONS = {
    {name="NONE", map={}},

    {name="RIGHT", map={right=true}},
    {name="LEFT",  map={left=true}},
    {name="JUMP",  map={A=true}},
    {name="SHOOT", map={B=true}},
    {name="JUMP_RIGHT", map={A=true, right=true}},

    -- ladder actions
    {name="UP",        map={up=true}},
    {name="DOWN",      map={down=true}},
    {name="UP_RIGHT",  map={up=true, right=true}},
    {name="UP_LEFT",   map={up=true, left=true}},
    {name="JUMP_UP",   map={A=true, up=true}}, -- grab ladder bottom
}

------------------------------------------------
-- STATE & RESET
------------------------------------------------
local MIN_LIFE = 1
local death_wait_frames = 0
local episode = 1

local room_start_state = savestate.create()
savestate.save(room_start_state)

------------------------------------------------
-- HELPERS
------------------------------------------------
local function r(addr) return memory.readbyte(addr) or 0 end

local function get_bucket(v, maxv, buckets)
    local b = math.floor((v / (maxv + 1)) * buckets)
    if b < 0 then b = 0 end
    if b >= buckets then b = buckets - 1 end
    return b
end

local function state_to_key(xb, yb, hpb, scroll)
    return tostring(xb) .. "_" .. tostring(yb) .. "_" .. tostring(hpb) .. "_" .. tostring(scroll)
end

-- Ladder detection (universal)
local function is_on_ladder()
    return r(ADDR_LADDER) == 1
end

-- Universal ladder bottom detector (approx, works across stages)
-- Slightly generous alignment window to allow the agent to learn alignment too.
local function can_grab_ladder_bottom(x, y)
    local tx = x % 16
    local aligned = (tx >= 5 and tx <= 11) -- slightly wider tolerance
    -- Broad Y window to cover many rooms; the agent will learn precise alignment via Q.
    local y_ok = (y >= 100 and y <= 180)
    return aligned and y_ok
end

------------------------------------------------
-- Q-TABLES
------------------------------------------------
local Q_rooms = {}

local function save_qtable_room(id)
    local Q = Q_rooms[id]
    if not Q then return end
    local f = io.open(Q_SAVE_PREFIX..id..".csv","w")
    if not f then return end
    for k, tab in pairs(Q) do
        for ai, qv in pairs(tab) do
            f:write(string.format("%s,%d,%.6f\n", k, ai, qv))
        end
    end
    f:close()
end

local function load_qtable_room(id)
    Q_rooms[id] = Q_rooms[id] or {}
    local Q = Q_rooms[id]
    local f = io.open(Q_SAVE_PREFIX..id..".csv","r")
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

------------------------------------------------
-- Q-LEARNING
------------------------------------------------
local function max_q(key, room)
    local Q = Q_rooms[room]
    if not Q then return 0 end
    local tab = Q[key]
    if not tab then return 0 end
    local best = -1e9
    for ai = 1, #ACTIONS do
        local qv = tab[ai] or 0
        if qv > best then best = qv end
    end
    if best < -1e8 then return 0 end
    return best
end

-- Ladder-aware action selector
local function choose_action(key, eps, room, x, y)
    local on_ladder = is_on_ladder()
    local can_down  = can_grab_ladder_bottom(x, y)

    local allowed = {}

    if on_ladder then
        -- allow vertical ladder movement + option to step off horizontally or do nothing
        for i,a in ipairs(ACTIONS) do
            if a.name=="UP" or a.name=="DOWN" or a.name=="UP_RIGHT" or a.name=="UP_LEFT" or a.name=="NONE" then
                table.insert(allowed, i)
            end
        end

        -- ensure left/right are available to exit ladder horizontally if needed
        for i,a in ipairs(ACTIONS) do
            if a.name=="LEFT" or a.name=="RIGHT" then
                table.insert(allowed, i)
            end
        end

    elseif can_down then
        -- at bottom alignment: allow JUMP_UP to grab ladder and DOWN to attach (and a small number of movement tries)
        for i,a in ipairs(ACTIONS) do
            if a.name=="DOWN" or a.name=="JUMP_UP" or a.name=="LEFT" or a.name=="RIGHT" or a.name=="NONE" then
                table.insert(allowed, i)
            end
        end

    else
        -- normal ground behavior: forbid ladder-only inputs to avoid accidental presses
        for i,a in ipairs(ACTIONS) do
            if a.name~="UP" and a.name~="DOWN" and a.name~="UP_RIGHT" and a.name~="UP_LEFT" and a.name~="JUMP_UP" then
                table.insert(allowed, i)
            end
        end
    end

    -- safety fallback: allow everything if somehow empty
    if #allowed == 0 then
        for i=1,#ACTIONS do table.insert(allowed, i) end
    end

    local Q = Q_rooms[room]
    local tab = Q and Q[key] or nil

    -- exploration
    if math.random() < eps or tab == nil then
        return allowed[math.random(#allowed)], true
    end

    -- exploitation among allowed actions
    local best_ai = allowed[1]
    local best_q = -1e9
    for _, ai in ipairs(allowed) do
        local qv = tab[ai] or 0
        if qv > best_q then
            best_q = qv
            best_ai = ai
        end
    end
    return best_ai, false
end

local function update_q(s_key, ai, reward, s2_key, room)
    local Q = Q_rooms[room]
    Q[s_key] = Q[s_key] or {}
    local q = Q[s_key][ai] or 0
    local target = reward + GAMMA * max_q(s2_key, room)
    local newq = q + ALPHA * (target - q)
    Q[s_key][ai] = newq
end

------------------------------------------------
-- LOG FILE
------------------------------------------------
local logf = io.open(LOG_FILE,"a")
if logf then
    logf:write("frame,xb,yb,hpb,scr,ladder,action,reward,eps,episode,room,explore\n")
end

------------------------------------------------
-- STATE EXTRACTION
------------------------------------------------
local function get_state_key()
    local x = r(ADDR_X)
    local y = r(ADDR_Y)
    local hp = r(ADDR_HP)
    local sc = r(ADDR_SCROLL)
    local xb  = get_bucket(x,255,X_BUCKETS)
    local yb  = get_bucket(y,255,Y_BUCKETS)
    local hpb = get_bucket(hp,255,HP_BUCKETS)
    return state_to_key(xb,yb,hpb,sc), xb, yb, hpb, sc, x, y, hp
end

------------------------------------------------
-- BLOCKED DETECTION
------------------------------------------------
local last_positions = {}

local function is_blocked(nx, ny)
    table.insert(last_positions, {x=nx, y=ny})
    if #last_positions > BLOCKED_CHECK_FRAMES then
        table.remove(last_positions, 1)
    end

    local minx, maxx = last_positions[1].x, last_positions[1].x
    local miny, maxy = last_positions[1].y, last_positions[1].y
    for _, p in ipairs(last_positions) do
        if p.x < minx then minx = p.x end
        if p.x > maxx then maxx = p.x end
        if p.y < miny then miny = p.y end
        if p.y > maxy then maxy = p.y end
    end

    local dx = maxx - minx
    local dy = maxy - miny
    return (dx <= BLOCKED_DELTA_THRESHOLD and dy <= BLOCKED_DELTA_THRESHOLD)
end

------------------------------------------------
-- INITIAL STATE
------------------------------------------------
local frame = 0
local eps = EPSILON
local s_key, xb, yb, hpb, sstat, cur_x, cur_y, cur_hp = get_state_key()
local cur_room = r(ADDR_SCROLL)

Q_rooms[cur_room] = Q_rooms[cur_room] or {}
load_qtable_room(cur_room)

-- select initial action and start cooldown
local current_action, explore = choose_action(s_key, eps, cur_room, cur_x, cur_y)
last_action = current_action
action_timer = ACTION_COOLDOWN

------------------------------------------------
-- MAIN LOOP
------------------------------------------------
while true do
    frame = frame + 1

    -- room change
    local new_room = r(ADDR_SCROLL)
    if new_room ~= cur_room then
        cur_room = new_room
        Q_rooms[cur_room] = Q_rooms[cur_room] or {}
        load_qtable_room(cur_room)
        room_start_state = savestate.create()
        savestate.save(room_start_state)
        last_positions = {}
        print(string.format(">>> New room detected. Room ID: %d", cur_room))
    end

    -- death handling
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
            current_action, explore = choose_action(s_key, eps, cur_room, cur_x, cur_y)
            last_action = current_action
            action_timer = ACTION_COOLDOWN
            last_positions = {}
        end

    else
        death_wait_frames = 0

        -- ACTION SELECTION WITH COOLDOWN
        action_timer = action_timer - 1
        if action_timer <= 0 then
            -- pick a new action based on current observed state
            current_action, explore = choose_action(s_key, eps, cur_room, cur_x, cur_y)
            last_action = current_action
            action_timer = ACTION_COOLDOWN
        else
            -- keep applying last_action while timer > 0
            current_action = last_action
        end

        -- execute action (applied every frame while held)
        joypad.set(1, ACTIONS[current_action].map)
        emu.frameadvance()

        -- next state
        local s2_key, xb2, yb2, hpb2, sstat2, nx, ny, nhp = get_state_key()
        local ladder_flag = is_on_ladder() and 1 or 0

        -- reward
        local dx = nx - cur_x
        if dx < -128 then dx = dx + 256 end
        local reward = PROGRESS_SCALE * dx - TIME_PENALTY
        if nhp < cur_hp then reward = reward + DEATH_PENALTY end

        -- blocked check
        if is_blocked(nx, ny) then
            reward = reward - 1.0
            -- force exploration and reset cooldown so new exploratory input actually holds
            current_action, explore = choose_action(s_key, 1.0, cur_room, nx, ny)
            last_action = current_action
            action_timer = ACTION_COOLDOWN
        else
            -- if not blocked, consider next action when timer expires (we already scheduled it above)
        end

        -- update Q-table
        update_q(s_key, current_action, reward, s2_key, cur_room)

        -- decay epsilon
        eps = math.max(EPS_MIN, eps * EPS_DECAY)

        -- periodic logging
        if frame % LOG_EVERY == 0 and logf then
            logf:write(string.format("%d,%d,%d,%d,%d,%d,%s,%.4f,%.4f,%d,%d,%d\n",
                frame, xb2, yb2, hpb2, sstat2, ladder_flag,
                ACTIONS[current_action].name, reward, eps, episode, cur_room, explore and 1 or 0))
            logf:flush()
        end

        -- periodic save
        if frame % SAVE_EVERY == 0 then
            for id,_ in pairs(Q_rooms) do
                save_qtable_room(id)
            end
        end

        -- prepare next iter
        s_key = s2_key
        cur_x = nx; cur_y = ny; cur_hp = nhp

        -- HUD
        gui.text(6,10,string.format("MM2 Q-learning | frame:%d eps:%.3f", frame, eps))
        gui.text(6,24,string.format("Episode: %d", episode))
        gui.text(6,38,string.format("Room ID: %d", cur_room))
        gui.text(6,52,string.format("Action: %s (%d)", ACTIONS[current_action].name, current_action))
        gui.text(6,66,string.format("State xb:%d yb:%d hpb:%d scr:%d", xb2, yb2, hpb2, sstat2))
        gui.text(6,82,string.format("Reward: %.3f", reward))
    end
end

-- save on exit
for room_id,_ in pairs(Q_rooms) do
    save_qtable_room(room_id)
end
if logf then logf:close() end
