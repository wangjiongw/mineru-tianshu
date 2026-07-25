#!/bin/bash
# 本地开发环境 - 启动前端 Vue 服务

set -e

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "========================================"
echo "  MinerU Tianshu - 本地前端启动"
echo "========================================"
echo ""

# 进入 frontend 目录
cd "$(dirname "$0")/.." || exit 1
PROJECT_ROOT=$(pwd)
FRONTEND_DIR="$PROJECT_ROOT/frontend"

echo "${BLUE}📂 项目目录:${NC} $PROJECT_ROOT"
echo "${BLUE}📂 前端目录:${NC} $FRONTEND_DIR"
echo ""

# 检查 Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js 未安装，请先安装 Node.js 18+"
    exit 1
fi

NODE_VERSION=$(node --version)
echo "${GREEN}✅ Node.js 版本:${NC} $NODE_VERSION"

# 检查 npm
if ! command -v npm &> /dev/null; then
    echo "❌ npm 未安装"
    exit 1
fi

NPM_VERSION=$(npm --version)
echo "${GREEN}✅ npm 版本:${NC} $NPM_VERSION"

# 进入前端目录
cd "$FRONTEND_DIR"

# 检查依赖
if [ ! -d "node_modules" ]; then
    echo ""
    echo "${YELLOW}📦 安装前端依赖...${NC}"
    npm install
    echo "${GREEN}✅ 依赖安装完成${NC}"
fi

# 设置 API 地址（如果后端不在 8000 端口）
# export VITE_API_BASE_URL=http://localhost:8000

# 可选：指定构建输出目录（用于 preview/static）
# export VITE_OUT_DIR=dist-pod2

# 可选：运行模式（dev|preview），默认 dev
FRONTEND_MODE=${FRONTEND_MODE:-dev}

if [ -n "$VITE_OUT_DIR" ]; then
    echo "${BLUE}📦 使用构建输出目录:${NC} $VITE_OUT_DIR"
fi

# 启动开发服务器
echo ""
echo "${GREEN}========================================"
echo "  🚀 启动 Vue 前端服务"
echo "========================================${NC}"
echo ""

echo "前端地址: http://localhost:3000"
echo "API 地址: http://localhost:8000 (通过代理)"
echo "测试账户: admin / admin123"
echo "模式: $FRONTEND_MODE"
echo ""
echo "${BLUE}按 Ctrl+C 停止服务${NC}"
echo ""

if [ "$FRONTEND_MODE" = "preview" ]; then
    npm run build
    npm run preview
else
    npm run dev
fi
