#!/bin/bash
# 本地开发环境 - 一键启动（后端 + 前端）

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "========================================"
echo "  MinerU Tianshu - 本地开发启动"
echo "========================================"
echo ""

# 检查是否在项目根目录
if [ ! -f "pyproject.toml" ]; then
    echo "❌ 请在项目根目录运行此脚本"
    exit 1
fi

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "${BLUE}选择启动模式:${NC}"
echo "  1) 仅启动后端 (FastAPI)"
echo "  2) 仅启动前端 (Vue)"
echo "  3) 同时启动后端和前端"
echo ""
read -p "请选择 (1-3): " choice

case $choice in
    1)
        echo ""
        echo "${GREEN}🚀 启动后端服务...${NC}"
        bash "$SCRIPT_DIR/dev-backend.sh"
        ;;
    2)
        echo ""
        echo "${GREEN}🚀 启动前端服务...${NC}"
        bash "$SCRIPT_DIR/dev-frontend.sh"
        ;;
    3)
        echo ""
        echo "${GREEN}🚀 同时启动后端和前端...${NC}"
        echo ""

        # 在新终端窗口启动后端
        if command -v gnome-terminal &> /dev/null; then
            gnome-terminal -- bash -c "cd \"$SCRIPT_DIR\"; bash dev-backend.sh; exec bash"
        elif command -v xterm &> /dev/null; then
            xterm -e "bash \"$SCRIPT_DIR/dev-backend.sh\"" &
        else
            # 后台启动后端
            bash "$SCRIPT_DIR/dev-backend.sh" &
            BACKEND_PID=$!
            echo "${BLUE}后端 PID: $BACKEND_PID${NC}"
        fi

        sleep 3

        # 启动前端
        bash "$SCRIPT_DIR/dev-frontend.sh"
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac
