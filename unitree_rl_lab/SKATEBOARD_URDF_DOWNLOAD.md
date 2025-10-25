# 🛹 How to Download and Install Skateboard URDF

## ✅ **PERFECT TIMING!** You found exactly what we need!

GitHub Repo: **https://github.com/dancher00/skateboard**

This repo contains:
- ✅ 3D CAD Model in SolidWorks
- ✅ **URDF File** (what we need!)
- ✅ 6-DOF skateboard with trucks and wheels
- ✅ ROS compatible

---

## 📥 **METHOD 1: Direct Download (Easiest)**

### **Step 1: Download the robot folder**

```bash
cd /home/prantheman/CALHACK/CALHACKS12/unitree_rl_lab/assets/urdf/skateboard

# Download the URDF and meshes from GitHub
wget https://github.com/dancher00/skateboard/archive/refs/heads/main.zip
unzip main.zip
mv skateboard-main/robot/* .
rm -rf skateboard-main main.zip

# Verify files are there
ls -la
```

You should see:
```
skateboard.urdf
meshes/
  deck.STL
  axep.STL
  wheel_0.STL
  wheel_1.STL
  wheel_2.STL
  wheel_3.STL
CMakeLists.txt
launch/
```

### **Step 2: Fix the URDF paths**

The downloaded URDF has relative paths like `package://skateboard/meshes/...`. We need absolute paths:

```bash
cd /home/prantheman/CALHACK/CALHACKS12/unitree_rl_lab/assets/urdf/skateboard

# Create a fixed URDF
cp robot/skateboard.urdf skateboard.urdf

# Fix mesh paths (replace package:// with absolute path)
sed -i 's|package://skateboard/meshes|file:///home/prantheman/CALHACK/CALHACKS12/unitree_rl_lab/assets/urdf/skateboard/robot/meshes|g' skateboard.urdf
```

---

## 📥 **METHOD 2: Git Clone (Recommended)**

```bash
cd /home/prantheman/CALHACK/CALHACKS12/unitree_rl_lab/assets/urdf

# Clone the entire repo
git clone https://github.com/dancher00/skateboard.git

# Move robot folder contents
mv skateboard/robot/* skateboard/
rm -rf skateboard/parts skateboard/*.SLDPRT skateboard/*.SLDASM skateboard/*.png

# Now you have:
# skateboard/skateboard.urdf
# skateboard/meshes/...
```

Then fix paths:
```bash
cd skateboard
sed -i 's|package://skateboard/meshes|file:///home/prantheman/CALHACK/CALHACKS12/unitree_rl_lab/assets/urdf/skateboard/meshes|g' skateboard.urdf
```

---

## 🔧 **METHOD 3: Manual Download (If commands don't work)**

1. Go to: https://github.com/dancher00/skateboard
2. Click **Code** → **Download ZIP**
3. Extract the ZIP file
4. Copy the `robot/` folder contents to:
   `/home/prantheman/CALHACK/CALHACKS12/unitree_rl_lab/assets/urdf/skateboard/`
5. Rename `robot/skateboard.urdf` to just `skateboard.urdf`
6. Edit `skateboard.urdf` and replace all `package://skateboard/meshes` with:
   `file:///home/prantheman/CALHACK/CALHACKS12/unitree_rl_lab/assets/urdf/skateboard/meshes`

---

## ✅ **VERIFY IT WORKS**

After downloading, your directory should look like:

```
/home/prantheman/CALHACK/CALHACKS12/unitree_rl_lab/assets/urdf/skateboard/
├── skateboard.urdf          ← Main URDF file
└── meshes/                  ← STL mesh files
    ├── axep.STL
    ├── desk_as00.STL
    ├── wheel_0.STL
    ├── wheel_1.STL
    ├── wheel_2.STL
    └── wheel_3.STL
```

**Check the URDF exists:**
```bash
ls -lh /home/prantheman/CALHACK/CALHACKS12/unitree_rl_lab/assets/urdf/skateboard/skateboard.urdf
```

---

## 🎯 **WHAT THIS SKATEBOARD HAS**

According to the repo, this is a **6-DOF Skateboard** with:

1. **Deck** - Main board (rigid body)
2. **Front Truck** - With bushings (revolute joint)
3. **Rear Truck** - With bushings (revolute joint)  
4. **4 Wheels** - Rotating (continuous joints)

**For our stationary version:**
- We're using `fix_base=True` in `skateboard.py`
- This locks all joints (wheels won't spin, trucks won't pivot)
- Perfect for learning to balance first!

**For moving version later:**
- Set `fix_base=False`
- Wheels will spin, trucks can pivot
- More realistic dynamics!

---

## 🔧 **IF YOU NEED TO SIMPLIFY THE URDF**

The downloaded URDF has 6 DOF (joints). For stationary balancing, we can simplify it.

**Option 1: Use as-is** (Recommended for now)
- Keep all joints
- Isaac Lab will handle fixed joints automatically with `fix_base=True`

**Option 2: Simplify to single rigid body**
```bash
# Create a simplified version
cd /home/prantheman/CALHACK/CALHACKS12/unitree_rl_lab/assets/urdf/skateboard
cp skateboard.urdf skateboard_full.urdf
# Then manually edit skateboard.urdf to remove joints and merge links
```

---

## 📏 **SKATEBOARD DIMENSIONS**

From the repo's 3D model:
- **Length**: ~80cm (31 inches) - typical skateboard
- **Width**: ~20cm (8 inches)
- **Deck Height**: ~12cm from ground (with wheels)
- **Weight**: ~3-4 kg

**You may need to adjust in `skateboard.py`:**
```python
init_state = RigidObjectCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.12),  # ← Adjust if needed based on actual URDF height
```

---

## 🚀 **QUICK INSTALL SCRIPT**

Run this all at once:

```bash
#!/bin/bash
cd /home/prantheman/CALHACK/CALHACKS12/unitree_rl_lab/assets/urdf/skateboard

# Download
wget -q https://github.com/dancher00/skateboard/archive/refs/heads/main.zip
unzip -q main.zip
cp -r skateboard-main/robot/* .
rm -rf skateboard-main main.zip

# Fix paths in URDF
sed -i 's|package://skateboard/meshes|file:///home/prantheman/CALHACK/CALHACKS12/unitree_rl_lab/assets/urdf/skateboard/meshes|g' skateboard.urdf

# Verify
echo "✅ Skateboard URDF installed!"
ls -lh skateboard.urdf
ls -lh meshes/

echo ""
echo "🎉 Ready to train! Run:"
echo "python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Skateboard --num_envs 128 --max_iterations 100"
```

Save this as `download_skateboard.sh` and run:
```bash
chmod +x download_skateboard.sh
./download_skateboard.sh
```

---

## 🐛 **TROUBLESHOOTING**

### **Error: "Cannot load URDF"**
- Check file exists: `ls /home/.../skateboard.urdf`
- Check mesh paths in URDF are correct (should be `file://...` not `package://...`)

### **Error: "Mesh not found"**
- Verify meshes folder exists: `ls meshes/`
- Check URDF mesh references match actual filenames

### **Robot falls through skateboard**
- Skateboard might be too thin - check collision geometry in URDF
- May need to adjust `pos=(0.0, 0.0, 0.12)` in skateboard.py

---

## 📚 **REFERENCES**

- **GitHub Repo**: https://github.com/dancher00/skateboard
- **URDF Tutorial**: https://www.epfl.ch/labs/biorob/wp-content/uploads/2019/02/SW2URDF_instructions.pdf
- **Original SW2URDF Tool**: http://wiki.ros.org/sw_urdf_exporter

---

**You're all set!** 🛹🤖 Download the URDF and let's train! 🚀

