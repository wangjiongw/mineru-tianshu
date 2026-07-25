#!/bin/bash
# 本地开发环境 - 启动后端 FastAPI 服务

set -e

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "========================================"
echo "  MinerU Tianshu - 本地后端启动"
echo "========================================"
echo ""

# 进入 backend 目录
cd "$(dirname "$0")/.." || exit 1
PROJECT_ROOT=$(pwd)
BACKEND_DIR="$PROJECT_ROOT/backend"

echo "${BLUE}📂 项目目录:${NC} $PROJECT_ROOT"
echo "${BLUE}📂 后端目录:${NC} $BACKEND_DIR"
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 未安装，请先安装 Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "${GREEN}✅ Python 版本:${NC} $PYTHON_VERSION"

# 检查是否在虚拟环境中
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo ""
    echo "${YELLOW}⚠️  建议使用虚拟环境${NC}"
    echo "创建虚拟环境:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo ""
    read -p "是否继续? (y/N): " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        exit 0
    fi
fi

# 设置环境变量
export DATABASE_PATH="$PROJECT_ROOT/data/db/mineru_tianshu.db"
export OUTPUT_PATH="$PROJECT_ROOT/data/output"
export UPLOAD_PATH="$PROJECT_ROOT/data/uploads"
export MODEL_PATH="$PROJECT_ROOT/models"
export LOG_LEVEL="INFO"

# 创建必要的目录
mkdir -p "$PROJECT_ROOT/data/db"
mkdir -p "$PROJECT_ROOT/data/output"
mkdir -p "$PROJECT_ROOT/data/uploads"
mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$PROJECT_ROOT/models"

echo "${GREEN}✅ 数据目录已创建${NC}"

# 初始化测试用户
echo ""
echo "${BLUE}🔑 初始化测试用户...${NC}"
python3 scripts/init_dev_user.py

# 检查依赖
echo ""
echo "${BLUE}📦 检查依赖...${NC}"
cd "$BACKEND_DIR"

if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "${YELLOW}⚠️  依赖未安装，正在安装...${NC}"
    pip install -r requirements.txt
fi

echo "${GREEN}✅ 依赖检查完成${NC}"

# 启动 API 服务
echo ""
echo "${GREEN}========================================"
echo "  🚀 启动 FastAPI 服务"
echo "========================================${NC}"
echo ""
echo "API 服务: http://localhost:8000"
echo "API 文档: http://localhost:8000/docs"
echo "测试账户: admin / admin123"
echo ""
echo "${BLUE}按 Ctrl+C 停止服务${NC}"
echo ""

# 启动服务
python3 api_server.py
