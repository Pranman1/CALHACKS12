# 🛹 Skateboard Balancing - Constraints & Observations

## 🆕 What We Just Added

You asked where to define more constraints - **here's everything we added!**

---

## 📊 **1. NEW OBSERVATIONS (Robot now knows about skateboard!)**

### **Added to `mdp/observations.py`:**

#### `robot_skateboard_relative_position()`
- **What**: 3D position of skateboard relative to robot (in robot's frame)
- **Output**: `[x, y, z]` vector
- **Why**: Robot needs to know where skateboard is beneath it
- **Used in**: Policy & Critic observations

#### `robot_skateboard_xy_distance()`
- **What**: Horizontal distance from robot to skateboard center
- **Output**: Scalar distance (meters)
- **Why**: Simple metric to stay centered
- **Used in**: Policy & Critic observations

#### `skateboard_orientation_body_frame()`
- **What**: Skateboard's gravity projection (tells if tilting)
- **Output**: `[x, y, z]` projected gravity
- **Why**: For moving skateboard version later
- **Used in**: Not yet (ready for motion version)

#### `feet_skateboard_relative_height()`
- **What**: Vertical distance of each foot from skateboard
- **Output**: `[left_foot_height, right_foot_height]`
- **Why**: Keep feet on skateboard surface
- **Used in**: Critic observations (privileged)

---

## 🚫 **2. NEW TERMINATION CONSTRAINTS (When to stop episode)**

### **Added to `mdp/terminations.py`:**

#### `robot_off_skateboard()`
- **Constraint**: Robot center must stay within `max_distance` of skateboard
- **Default**: 0.4m (40cm) horizontal distance
- **Terminates when**: Robot body drifts too far from skateboard center
- **Purpose**: Prevent robot from stepping off skateboard

#### `feet_off_skateboard()`
- **Constraint**: Both feet must stay near skateboard
- **Vertical limit**: 0.3m (30cm) from skateboard surface
- **Horizontal limit**: 0.6m (60cm) from skateboard center
- **Terminates when**: ANY foot violates either constraint
- **Purpose**: Ensure feet stay on/near skateboard deck

#### `skateboard_tilted()` 
- **Constraint**: Skateboard can't tilt more than `max_tilt_angle`
- **Default**: 0.5 rad (≈28°)
- **Terminates when**: Skateboard tilts too much
- **Purpose**: For moving skateboard scenarios (not used yet for stationary)

---

## 📋 **3. UPDATED ENVIRONMENT CONFIG**

### **Updated `skate_env_cfg.py`:**

#### **Policy Observations (What robot sees):**
```python
# Original observations
base_ang_vel              # Robot angular velocity
projected_gravity         # Robot gravity direction
joint_pos_rel            # Joint positions
joint_vel_rel            # Joint velocities
last_action              # Previous action

# NEW skateboard observations ✨
skateboard_relative_pos  # 3D position of skateboard (x,y,z)
skateboard_xy_distance   # Distance from skateboard center
```

**Total policy input**: ~150+ dimensions with history

#### **Critic Observations (What critic sees - privileged):**
```python
# Everything policy sees, PLUS:
base_lin_vel                  # Linear velocity (privileged)
feet_skateboard_height        # Foot heights (privileged)
```

**Total critic input**: ~180+ dimensions with history

#### **Terminations (When episode ends):**
```python
# Original terminations
time_out                     # 20 seconds
base_height < 0.4m          # Robot fell
bad_orientation > 0.8 rad   # Robot tilted too much

# NEW skateboard constraints ✨
robot_off_skateboard > 0.4m     # Center too far from board
feet_off_skateboard:            # Feet off board
  - vertical > 0.3m
  - horizontal > 0.6m
```

---

## 🎯 **HOW CONSTRAINTS WORK**

### **Soft Constraints (Rewards)**
These **guide** the robot toward desired behavior:
- `robot_on_skateboard`: Reward for staying centered (weight: 5.0)
- `feet_on_skateboard`: Reward for feet on surface (weight: 2.0)
- `stay_upright`: Penalty for tilting (weight: -10.0)

➡️ Robot **learns** to satisfy these over time

### **Hard Constraints (Terminations)**
These **end the episode** immediately when violated:
- `robot_off_skateboard`: Can't drift more than 40cm
- `feet_off_skateboard`: Feet can't leave skateboard
- `base_height`: Can't fall below 0.4m

➡️ Robot **must** satisfy these to continue episode

---

## 📐 **CONSTRAINT VALUES (You can tune these!)**

### **Termination Thresholds:**

| **Constraint** | **Default** | **What it means** | **Where to adjust** |
|----------------|-------------|-------------------|---------------------|
| `max_distance` | 0.4m | Robot center → skateboard | `skate_env_cfg.py` line 330 |
| `max_height_diff` | 0.3m | Feet height from board | `skate_env_cfg.py` line 338 |
| `max_xy_distance` | 0.6m | Feet → skateboard center | `skate_env_cfg.py` line 339 |
| `minimum_height` | 0.4m | Robot fell (from ground) | `skate_env_cfg.py` line 321 |
| `limit_angle` | 0.8 rad | Robot tilt (≈46°) | `skate_env_cfg.py` line 322 |

### **Observation Noise:**

| **Observation** | **Noise** | **Purpose** |
|-----------------|-----------|-------------|
| `skateboard_relative_pos` | ±0.02m | Sim2real robustness |
| `base_ang_vel` | ±0.2 | Sensor noise simulation |
| `projected_gravity` | ±0.05 | IMU noise simulation |

---

## 🔧 **HOW TO ADJUST CONSTRAINTS**

### **Make Constraints Tighter (Harder task):**
```python
# In skate_env_cfg.py, TerminationsCfg
robot_off_skateboard = DoneTerm(
    func=mdp.robot_off_skateboard,
    params={
        "max_distance": 0.2,  # Stricter! (was 0.4)
    },
)
```

### **Make Constraints Looser (Easier task):**
```python
feet_off_skateboard = DoneTerm(
    func=mdp.feet_off_skateboard,
    params={
        "max_height_diff": 0.5,    # More lenient (was 0.3)
        "max_xy_distance": 1.0,    # More lenient (was 0.6)
    },
)
```

### **Disable a Constraint:**
```python
# Comment out the constraint
# feet_off_skateboard = DoneTerm(...)
```

---

## 🎓 **CURRICULUM LEARNING (Progressive Constraints)**

You can make constraints **gradually tighter** during training:

```python
# Example (not implemented yet, but you could add):
class CurriculumCfg:
    skateboard_distance_curriculum = CurrTerm(
        func=mdp.modify_skateboard_distance_threshold,
        # Start with 0.6m, end with 0.3m
    )
```

This helps robot learn easier version first!

---

## 📊 **OBSERVATION SPACE SIZE**

### **Without skateboard observations:**
- Policy: ~140 dims × 5 history = 700 dims
- Critic: ~160 dims × 5 history = 800 dims

### **With skateboard observations:**
- Policy: ~145 dims × 5 history = **725 dims** 
- Critic: ~165 dims × 5 history = **825 dims**

**Impact**: Minimal increase (+3%), but robot now knows where skateboard is!

---

## 🧪 **TESTING CONSTRAINTS**

### **Verify observations work:**
```python
# After training starts, check tensorboard:
# - Look for "obs/skateboard_relative_pos"
# - Look for "obs/skateboard_xy_distance"
```

### **Verify terminations trigger:**
```python
# During training, watch for:
# - "terminations/robot_off_skateboard" count
# - "terminations/feet_off_skateboard" count
# Should be > 0 and decreasing over time
```

---

## 🎯 **SUMMARY**

### **Before (Original):**
❌ Robot didn't know where skateboard was
❌ No constraints on staying on skateboard
❌ Only basic height/orientation terminations

### **After (With Constraints):**
✅ Robot observes skateboard position (3D + distance)
✅ Hard constraints: must stay within 40cm
✅ Hard constraints: feet must stay on board
✅ Soft constraints: rewarded for alignment
✅ 3 new termination conditions
✅ 4 new observation functions

---

## 🚀 **FILES MODIFIED:**

1. ✅ `mdp/observations.py` - 4 new observation functions
2. ✅ `mdp/terminations.py` - 3 new termination functions (NEW FILE)
3. ✅ `mdp/__init__.py` - Import terminations
4. ✅ `skate_env_cfg.py` - Added observations & terminations to config

---

## 💡 **WHAT THIS MEANS FOR TRAINING:**

1. **Robot knows skateboard location** → Can learn to stay centered
2. **Episodes end if robot falls off** → Learns to avoid edge
3. **Rewards for staying on board** → Positive reinforcement
4. **Tighter constraints** → More challenging, but better sim2real

**Expected result**: Robot will learn to actively balance ON the skateboard, not just stand upright!

---

Love you! 🤖🛹 These constraints make the task properly defined!

