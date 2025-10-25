# 🛹 Skateboard Balancing Task - Setup Guide

## ✅ Files Created

### 1. **Skateboard Asset Configuration**
- **File**: `source/unitree_rl_lab/unitree_rl_lab/assets/objects/skateboard.py`
- **Purpose**: Defines the skateboard as a stationary rigid object

### 2. **Skateboard Balancing Task**
- **File**: `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/skate_env_cfg.py`
- **Purpose**: Complete environment configuration for balancing on a stationary skateboard
- **Features**:
  - Flat ground terrain
  - Stationary skateboard (fix_base=True)
  - Robot spawns on skateboard at height 0.90m
  - Rewards focused on staying upright and balanced
  - No velocity commands (just balancing)

### 3. **Custom Reward Functions**
- **File**: `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py`
- **Purpose**: Skateboard-specific reward functions
- **Functions added**:
  - `robot_skateboard_alignment()`: Keeps robot centered on skateboard (XY alignment)
  - `skateboard_orientation()`: Keeps skateboard level (for moving skateboard later)
  - `feet_on_skateboard()`: Rewards feet being on skateboard surface
  - `robot_skateboard_contact()`: Rewards maintaining foot contact with skateboard

### 4. **Task Registration**
- **File**: `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/__init__.py`
- **Purpose**: Registers the new task: `Unitree-G1-29dof-Skateboard`

---

## 📍 WHERE TO PUT YOUR SKATEBOARD URDF

### **Place your skateboard URDF file here:**
```
/home/prantheman/CALHACK/CALHACKS12/unitree_rl_lab/assets/urdf/skateboard/skateboard.urdf
```

### **URDF Requirements:**
1. **File name**: Must be named `skateboard.urdf`
2. **Meshes**: If your URDF references mesh files (STL, OBJ, DAE), place them in the same folder:
   ```
   assets/urdf/skateboard/
   ├── skateboard.urdf
   ├── deck.stl          # (example mesh files)
   ├── wheel_front.stl
   └── wheel_back.stl
   ```
3. **Height**: The skateboard deck should be approximately 0.12m high (adjust in `skateboard.py` if needed)
4. **Size**: Typical skateboard dimensions (length ~80cm, width ~20cm)

---

## 🎯 How the Task Works

### **Scenario**
- **Robot**: G1 29-DOF humanoid
- **Skateboard**: Stationary (fixed base, not moving)
- **Goal**: Robot learns to balance on top of the skateboard without falling

### **Key Configuration Details**

#### **Robot Initial Position**
```python
pos=(0.0, 0.0, 0.90)  # Height adjusted to be on skateboard
```
⚠️ **You may need to adjust this** based on your skateboard's actual height!

#### **Skateboard Position**
```python
pos=(0.0, 0.0, 0.12)  # Slightly above ground
fix_base=True         # Stationary (not moving)
```

#### **Rewards (Simplified for Balancing)**

**Primary Task Rewards:**
- ✅ `stay_upright`: Keep robot upright (-10.0 weight)
- ✅ `base_height`: Maintain correct height (-10.0 weight)
- ✅ `alive`: Bonus for staying balanced (1.0 weight)

**Skateboard-Specific Rewards:**
- ✅ `robot_on_skateboard`: Stay centered on skateboard (5.0 weight)
- ✅ `feet_on_skateboard`: Keep feet on skateboard surface (2.0 weight)

**Movement Penalties:**
- ❌ `base_xy_velocity`: Penalize horizontal movement (-1.0 weight)
- ❌ `base_linear_velocity`: Penalize z-axis movement (-2.0 weight)
- ❌ `base_angular_velocity`: Penalize roll/pitch rotation (-0.5 weight)

**Smoothness:**
- ❌ `energy`, `action_rate`, `joint_vel`, `joint_acc`: Smooth, energy-efficient movements

#### **Termination Conditions**
- Episode ends after 20 seconds
- Robot falls below 0.4m height
- Robot tilts more than 0.8 radians

---

## 🚀 How to Run

### **1. Verify Task is Registered**
```bash
./unitree_rl_lab.sh -l
```
Should show: `Unitree-G1-29dof-Skateboard`

### **2. Train the Task**
```bash
# Train with GUI (for debugging)
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Skateboard

# Train headless (faster, recommended)
python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Skateboard --num_envs 4096

# Quick test with fewer environments
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Skateboard --num_envs 128 --max_iterations 100
```

### **3. Test/Play Trained Policy**
```bash
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Skateboard

# Record video
python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Skateboard --video --video_length 500
```

---

## 🔧 Adjustments You Might Need

### **If Robot Spawns Too High/Low**
Edit `skate_env_cfg.py`:
```python
# Line ~52
robot: ArticulationCfg = ROBOT_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.90),  # ← Adjust this height
```

### **If Skateboard is Wrong Height**
Edit `source/unitree_rl_lab/unitree_rl_lab/assets/objects/skateboard.py`:
```python
# Line ~39
init_state = RigidObjectCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.12),  # ← Adjust this height
```

### **If You Want Different Skateboard Friction**
Edit `skateboard.py`:
```python
# Add physics_material to spawn config
spawn = sim_utils.UrdfFileCfg(
    asset_path=SKATEBOARD_URDF_PATH,
    rigid_props=...,
    collision_props=...,
)
```

---

## 📊 Expected Training Results

### **During Training**
- **Iterations**: ~5,000 - 20,000 for good balancing
- **Episode Length**: Should increase from ~50 steps to ~1000+ steps
- **Mean Reward**: Should increase from negative to positive values
- **FPS**: ~100,000-200,000 (depends on GPU)

### **Success Indicators**
✅ Robot stays upright (roll/pitch near 0)
✅ Robot maintains height around 0.90m
✅ Episode length reaches max (20 seconds = 1000 steps)
✅ Minimal base movement (staying centered on skateboard)

---

## 🎮 Next Steps (Moving Skateboard)

This version is for **stationary skateboard balancing only**.

When ready to add skateboard motion:
1. Create a new file: `skate_motion_env_cfg.py`
2. Set `fix_base=False` in skateboard config
3. Add skateboard velocity observations
4. Add skateboard velocity commands
5. Adjust rewards for forward motion

---

## 📝 Summary

### **What You Have Now**
✅ Skateboard asset definition  
✅ Stationary balancing task  
✅ **4 custom skateboard-specific reward functions**  
✅ Task registered and ready to train  
✅ Minimal code changes (focused on balancing only)

### **What You Need to Provide**
📍 Skateboard URDF file at: `assets/urdf/skateboard/skateboard.urdf`

### **Quick Test Command**
```bash
# After adding skateboard URDF
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Skateboard --num_envs 128 --max_iterations 100
```

---

## 🐛 Troubleshooting

### **Error: "Cannot find skateboard.urdf"**
- Make sure file exists at: `assets/urdf/skateboard/skateboard.urdf`
- Check the path in `skateboard.py` is correct

### **Error: "Robot falls through skateboard"**
- Adjust collision properties in skateboard URDF
- Check skateboard height vs robot height
- Ensure skateboard has collision geometry

### **Robot spawns away from skateboard**
- Check robot spawn position matches skateboard position (both at x=0, y=0)
- Adjust robot height to be on skateboard deck

---

Love you too! 🤖🛹 Let me know if you need help with the skateboard URDF or any adjustments!

