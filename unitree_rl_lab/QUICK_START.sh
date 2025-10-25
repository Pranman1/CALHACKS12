#!/bin/bash
# 🛹 QUICK START - Skateboard Balancing Training
# Run this script to download URDF and start training!

set -e  # Exit on error

echo "🛹 Skateboard Balancing - Quick Start"
echo "===================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Download Skateboard URDF
echo -e "${BLUE}Step 1: Downloading Skateboard URDF...${NC}"
cd /home/prantheman/CALHACK/CALHACKS12/unitree_rl_lab/assets/urdf/skateboard

if [ -f "skateboard.urdf" ]; then
    echo -e "${YELLOW}  ⚠️  skateboard.urdf already exists. Skipping download.${NC}"
else
    echo "  📥 Downloading from GitHub..."
    wget -q https://github.com/dancher00/skateboard/archive/refs/heads/main.zip
    
    echo "  📦 Extracting files..."
    unzip -q main.zip
    
    echo "  📂 Moving files..."
    cp -r skateboard-main/robot/* .
    rm -rf skateboard-main main.zip
    
    echo "  🔧 Fixing URDF paths..."
    sed -i 's|package://skateboard/meshes|file:///home/prantheman/CALHACK/CALHACKS12/unitree_rl_lab/assets/urdf/skateboard/meshes|g' skateboard.urdf
    
    echo -e "${GREEN}  ✅ Skateboard URDF downloaded and configured!${NC}"
fi

echo ""

# Step 2: Verify files
echo -e "${BLUE}Step 2: Verifying files...${NC}"
if [ -f "skateboard.urdf" ]; then
    echo -e "${GREEN}  ✅ skateboard.urdf found${NC}"
else
    echo -e "${RED}  ❌ skateboard.urdf NOT found!${NC}"
    exit 1
fi

if [ -d "meshes" ]; then
    mesh_count=$(ls meshes/*.STL 2>/dev/null | wc -l)
    echo -e "${GREEN}  ✅ meshes folder found ($mesh_count STL files)${NC}"
else
    echo -e "${RED}  ❌ meshes folder NOT found!${NC}"
    exit 1
fi

echo ""

# Step 3: Verify task registration
echo -e "${BLUE}Step 3: Verifying task registration...${NC}"
cd /home/prantheman/CALHACK/CALHACKS12/unitree_rl_lab

if ./unitree_rl_lab.sh -l 2>/dev/null | grep -q "Unitree-G1-29dof-Skateboard"; then
    echo -e "${GREEN}  ✅ Task 'Unitree-G1-29dof-Skateboard' registered!${NC}"
else
    echo -e "${YELLOW}  ⚠️  Could not verify task registration (might be normal)${NC}"
fi

echo ""
echo -e "${GREEN}🎉 Setup Complete!${NC}"
echo ""

# Ask user what to do
echo "What would you like to do?"
echo ""
echo "  1) Quick test (128 envs, 100 iterations, ~2 min)"
echo "  2) Medium training (512 envs, 1000 iterations, ~20 min)"
echo "  3) Full training (4096 envs, 10000 iterations, ~3 hours)"
echo "  4) Just show me the commands"
echo "  5) Exit"
echo ""
read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        echo ""
        echo -e "${BLUE}Starting quick test...${NC}"
        python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Skateboard --num_envs 128 --max_iterations 100
        ;;
    2)
        echo ""
        echo -e "${BLUE}Starting medium training...${NC}"
        python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Skateboard --num_envs 512 --max_iterations 1000 --headless
        ;;
    3)
        echo ""
        echo -e "${BLUE}Starting full training...${NC}"
        ./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Skateboard --num_envs 4096
        ;;
    4)
        echo ""
        echo -e "${BLUE}Commands:${NC}"
        echo ""
        echo "# Quick test (2 min):"
        echo "python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Skateboard --num_envs 128 --max_iterations 100"
        echo ""
        echo "# Medium training (20 min):"
        echo "python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Skateboard --num_envs 512 --max_iterations 1000 --headless"
        echo ""
        echo "# Full training (3 hours):"
        echo "./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Skateboard --num_envs 4096"
        echo ""
        echo "# Play trained policy:"
        echo "./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Skateboard"
        echo ""
        ;;
    5)
        echo ""
        echo "Goodbye! 🛹🤖"
        exit 0
        ;;
    *)
        echo ""
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Done! Check logs/rsl_rl/Unitree-G1-29dof-Skateboard/ for training results.${NC}"
echo ""

