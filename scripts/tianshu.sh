#!/bin/bash
# ============================================================================
# MinerU Tianshu - 统一启动脚本
# ============================================================================
#
# 8卡NPU裸机部署完整生命周期管理
# 启动顺序: VLLM(8实例) → API Server → Workers(8个) → Frontend
#
# 使用方式:
#   bash scripts/tianshu.sh start [vllm|api|worker|frontend]  # 启动服务
#   bash scripts/tianshu.sh stop  [vllm|api|worker|frontend]  # 停止服务
#   bash scripts/tianshu.sh restart [vllm|api|worker|frontend|mcp|all]  # 重启服务
#   bash scripts/tianshu.sh status                            # 查看状态
#   bash scripts/tianshu.sh logs  [vllm|worker|api|frontend]  # 查看日志
#   bash scripts/tianshu.sh test                              # 端到端验证
#
# ============================================================================

set -o pipefail

# ============================================================================
# 配置区 - 修改此处即可适配不同环境
# ============================================================================

PROJECT_ROOT="/data/projects/mineru/mineru-tianshu"

# VLLM 服务配置
VLLM_MODEL_PATH="/share/wangjiong/model_zoo/modelscope/models/OpenDataLab/MinerU2___5-2509-1___2B"
VLLM_BASE_PORT=30025
VLLM_NUM_INSTANCES=8
VLLM_MAX_MODEL_LEN=8192

# Worker 配置
WORKER_BASE_PORT=8101
WORKER_NUM_INSTANCES=8
WORKER_ACCELERATOR="cpu"

# API Server 配置
API_PORT=8000

# MCP Server 配置
MCP_PORT=8002

# 前端配置
FRONTEND_PORT=3000

# 路径配置
DATABASE_PATH="/share/wangjiong/databases/mineru/mineru_tianshu.db"
OUTPUT_PATH="/share/wangjiong/databases/mineru/mineru_outputs"
UPLOAD_PATH="/share/wangjiong/databases/mineru/mineru_uploads"
LOG_DIR="/share/wangjiong/databases/mineru/mineru_logs"

# 日志子目录
VLLM_LOG_DIR="${LOG_DIR}/vllm"
WORKER_LOG_DIR="${LOG_DIR}/worker"
API_LOG_DIR="${LOG_DIR}/api"

# Redis 队列配置
REDIS_QUEUE_ENABLED="true"
REDIS_HOST="localhost"
REDIS_PORT="6379"
REDIS_DB="0"
REDIS_PASSWORD="redis123"
REDIS_QUEUE_KEY="tianshu:task_queue"
REDIS_PROCESSING_KEY="tianshu:processing"
REDIS_TASK_TIMEOUT="3600"

# ============================================================================
# 内部变量
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="${PROJECT_ROOT}/backend"
FRONTEND_DIR="${PROJECT_ROOT}/frontend"

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ============================================================================
# 工具函数
# ============================================================================

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "\n${BLUE}━━━ $* ━━━${NC}\n"; }

separator() {
    echo -e "${BLUE}══════════════════════════════════════════════════════════════${NC}"
}

# ============================================================================
# 目录初始化
# ============================================================================

init_dirs() {
    mkdir -p "$VLLM_LOG_DIR" "$WORKER_LOG_DIR" "$API_LOG_DIR" \
             "$OUTPUT_PATH" "$UPLOAD_PATH" \
             "${PROJECT_ROOT}/data/db" "${PROJECT_ROOT}/models"
}

# ============================================================================
# VLLM 服务管理
# ============================================================================

start_vllm() {
    log_step "启动 VLLM 服务 (${VLLM_NUM_INSTANCES} 个实例)"

    local started=0
    for i in $(seq 0 $((VLLM_NUM_INSTANCES - 1))); do
        local port=$((VLLM_BASE_PORT + i))
        local log_file="${VLLM_LOG_DIR}/vllm_npu${i}_port${port}.log"
        local pid_file="${VLLM_LOG_DIR}/vllm_npu${i}.pid"

        # 跳过已运行的实例
        if [ -f "$pid_file" ] && ps -p "$(cat "$pid_file")" > /dev/null 2>&1; then
            log_info "VLLM #${i} (NPU=${i}, Port=${port}) 已在运行"
            started=$((started + 1))
            continue
        fi

        log_info "启动 VLLM #${i}: NPU=${i}, Port=${port}"

        nohup bash -lc "
            export ASCEND_VISIBLE_DEVICES='${i}'
            export ASCEND_RT_VISIBLE_DEVICES='${i}'
            export DEVICE_ID=0
            export ASCEND_DEVICE_ID=0
            exec vllm serve '${VLLM_MODEL_PATH}' \
                --host 0.0.0.0 \
                --tensor-parallel-size 1 \
                --port ${port} \
                --max-model-len ${VLLM_MAX_MODEL_LEN} \
                --dtype float16 \
                --trust-remote-code
        " > "$log_file" 2>&1 &

        local pid=$!
        echo "$pid" > "$pid_file"
        sleep 2

        if ps -p "$pid" > /dev/null 2>&1; then
            log_info "VLLM #${i} 已启动 (PID: $pid)"
            started=$((started + 1))
        else
            log_error "VLLM #${i} 启动失败，查看: $log_file"
        fi
    done

    log_info "VLLM 启动完成: ${started}/${VLLM_NUM_INSTANCES}"

    if [ "$started" -lt "$VLLM_NUM_INSTANCES" ]; then
        log_warn "部分 VLLM 实例未启动成功，等待初始化..."
    fi

    # 健康检查
    log_info "等待 VLLM 健康检查..."
    local healthy=0
    for round in $(seq 1 24); do
        healthy=0
        for i in $(seq 0 $((VLLM_NUM_INSTANCES - 1))); do
            local port=$((VLLM_BASE_PORT + i))
            if curl -s "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
                healthy=$((healthy + 1))
            fi
        done
        if [ "$healthy" -eq "$VLLM_NUM_INSTANCES" ]; then
            log_info "所有 ${VLLM_NUM_INSTANCES} 个 VLLM 实例就绪"
            return 0
        fi
        echo -n "."
        sleep 10
    done
    echo ""
    log_warn "VLLM 健康检查: ${healthy}/${VLLM_NUM_INSTANCES} 就绪 (部分可能仍在初始化)"
}

stop_vllm() {
    log_step "停止 VLLM 服务"

    for i in $(seq 0 $((VLLM_NUM_INSTANCES - 1))); do
        local pid_file="${VLLM_LOG_DIR}/vllm_npu${i}.pid"
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            if ps -p "$pid" > /dev/null 2>&1; then
                kill "$pid" 2>/dev/null
                sleep 1
                ps -p "$pid" > /dev/null 2>&1 && kill -9 "$pid" 2>/dev/null
                log_info "VLLM #${i} (PID: $pid) 已停止"
            fi
            rm -f "$pid_file"
        fi
    done

    # 清理残留进程
    pkill -f "vllm serve.*${VLLM_MODEL_PATH}" 2>/dev/null || true
    log_info "VLLM 服务已全部停止"
}

status_vllm() {
    echo -e "${CYAN}VLLM 服务 (${VLLM_NUM_INSTANCES} 个实例)${NC}"
    local running=0
    for i in $(seq 0 $((VLLM_NUM_INSTANCES - 1))); do
        local port=$((VLLM_BASE_PORT + i))
        local pid_file="${VLLM_LOG_DIR}/vllm_npu${i}.pid"
        if [ -f "$pid_file" ] && ps -p "$(cat "$pid_file")" > /dev/null 2>&1; then
            if curl -s "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
                echo "  ✅ NPU ${i} (Port ${port}) - 运行中"
                running=$((running + 1))
            else
                echo "  ⏳ NPU ${i} (Port ${port}) - 初始化中"
            fi
        else
            echo "  ❌ NPU ${i} (Port ${port}) - 已停止"
        fi
    done
    echo "  运行中: ${running}/${VLLM_NUM_INSTANCES}"
}

# ============================================================================
# API Server 管理
# ============================================================================

start_api() {
    log_step "启动 API Server (端口 ${API_PORT})"

    local pid_file="${API_LOG_DIR}/api.pid"
    local log_file="${API_LOG_DIR}/api.log"

    # 跳过已运行的实例
    if [ -f "$pid_file" ] && ps -p "$(cat "$pid_file")" > /dev/null 2>&1; then
        log_info "API Server 已在运行 (PID: $(cat "$pid_file"))"
        return 0
    fi

    cd "$BACKEND_DIR"

    DATABASE_PATH="$DATABASE_PATH" \
    OUTPUT_PATH="$OUTPUT_PATH" \
    UPLOAD_PATH="$UPLOAD_PATH" \
    API_PORT="$API_PORT" \
    REDIS_QUEUE_ENABLED="$REDIS_QUEUE_ENABLED" \
    REDIS_HOST="$REDIS_HOST" \
    REDIS_PORT="$REDIS_PORT" \
    REDIS_DB="$REDIS_DB" \
    REDIS_PASSWORD="$REDIS_PASSWORD" \
    REDIS_QUEUE_KEY="$REDIS_QUEUE_KEY" \
    REDIS_PROCESSING_KEY="$REDIS_PROCESSING_KEY" \
    REDIS_TASK_TIMEOUT="$REDIS_TASK_TIMEOUT" \
    nohup python api_server.py > "$log_file" 2>&1 &

    local pid=$!
    echo "$pid" > "$pid_file"
    log_info "API Server 启动中 (PID: $pid)..."

    # 健康检查
    for i in $(seq 1 30); do
        if curl -s "http://localhost:${API_PORT}/docs" > /dev/null 2>&1; then
            log_info "API Server 就绪 (端口 ${API_PORT})"
            return 0
        fi
        echo -n "."
        sleep 2
    done
    echo ""
    log_error "API Server 启动超时，查看: $log_file"
    return 1
}

stop_api() {
    log_step "停止 API Server"

    local pid_file="${API_LOG_DIR}/api.pid"
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        kill "$pid" 2>/dev/null || true
        rm -f "$pid_file"
    fi
    pkill -f "python api_server.py" 2>/dev/null || true
    log_info "API Server 已停止"
}

status_api() {
    echo -e "${CYAN}API Server (端口 ${API_PORT})${NC}"
    if curl -s "http://localhost:${API_PORT}/docs" > /dev/null 2>&1; then
        echo "  ✅ 运行中 - http://localhost:${API_PORT}/docs"
    else
        echo "  ❌ 未运行"
    fi
}

# ============================================================================
# MCP Server 管理
# ============================================================================

start_mcp() {
    log_step "启动 MCP Server (端口 ${MCP_PORT})"

    local pid_file="${API_LOG_DIR}/mcp.pid"
    local log_file="${API_LOG_DIR}/mcp.log"

    # 跳过已运行的实例
    if [ -f "$pid_file" ] && ps -p "$(cat "$pid_file")" > /dev/null 2>&1; then
        log_info "MCP Server 已在运行 (PID: $(cat "$pid_file"))"
        return 0
    fi

    cd "$BACKEND_DIR"

    DATABASE_PATH="$DATABASE_PATH" \
    OUTPUT_PATH="$OUTPUT_PATH" \
    API_BASE_URL="http://localhost:${API_PORT}" \
    MCP_PORT="$MCP_PORT" \
    MCP_HOST="0.0.0.0" \
    nohup python mcp_server.py > "$log_file" 2>&1 &

    local pid=$!
    echo "$pid" > "$pid_file"
    log_info "MCP Server 启动中 (PID: $pid)..."

    sleep 3

    if ps -p "$pid" > /dev/null 2>&1; then
        log_info "MCP Server 就绪 - http://0.0.0.0:${MCP_PORT}/mcp"
    else
        log_error "MCP Server 启动失败，查看: $log_file"
        return 1
    fi
}

stop_mcp() {
    log_step "停止 MCP Server"

    local pid_file="${API_LOG_DIR}/mcp.pid"
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        kill "$pid" 2>/dev/null || true
        rm -f "$pid_file"
    fi
    pkill -f "python mcp_server.py" 2>/dev/null || true
    log_info "MCP Server 已停止"
}

status_mcp() {
    echo -e "${CYAN}MCP Server (端口 ${MCP_PORT})${NC}"
    if curl -s "http://localhost:${MCP_PORT}/mcp" > /dev/null 2>&1; then
        echo "  ✅ 运行中 - http://0.0.0.0:${MCP_PORT}/mcp"
    else
        echo "  ❌ 未运行"
    fi
}

# ============================================================================
# Worker 管理
# ============================================================================

start_workers() {
    log_step "启动 Workers (${WORKER_NUM_INSTANCES} 个独立进程)"

    # 预检查: VLLM 和 API
    local vllm_ok=0
    for i in $(seq 0 $((VLLM_NUM_INSTANCES - 1))); do
        local port=$((VLLM_BASE_PORT + i))
        curl -s "http://localhost:${port}/v1/models" > /dev/null 2>&1 && vllm_ok=$((vllm_ok + 1))
    done
    if [ "$vllm_ok" -lt "$VLLM_NUM_INSTANCES" ]; then
        log_error "VLLM 未就绪 (${vllm_ok}/${VLLM_NUM_INSTANCES})，请先启动 VLLM"
        return 1
    fi

    if ! curl -s "http://localhost:${API_PORT}/docs" > /dev/null 2>&1; then
        log_error "API Server 未运行，请先启动 API Server"
        return 1
    fi

    # 停止旧 Worker
    pkill -f "litserve_worker.py.*81[0-9][1-9]" 2>/dev/null || true
    sleep 2

    cd "$BACKEND_DIR"

    local started=0
    for i in $(seq 0 $((WORKER_NUM_INSTANCES - 1))); do
        local port=$((WORKER_BASE_PORT + i))
        local vllm_port=$((VLLM_BASE_PORT + i))
        local vllm_api="http://localhost:${vllm_port}/v1"
        local log_file="${WORKER_LOG_DIR}/worker_${i}_port${port}.log"
        local pid_file="${WORKER_LOG_DIR}/worker_${i}.pid"

        # 跳过已运行的实例
        if [ -f "$pid_file" ] && ps -p "$(cat "$pid_file")" > /dev/null 2>&1; then
            log_info "Worker #${i} (Port ${port}) 已在运行"
            started=$((started + 1))
            continue
        fi

        log_info "启动 Worker #${i}: Port=${port}, VLLM=${vllm_api}"

        DATABASE_PATH="$DATABASE_PATH" \
        OUTPUT_PATH="$OUTPUT_PATH" \
        WORKER_PORT="$port" \
        ASCEND_VISIBLE_DEVICES="$i" \
        ASCEND_RT_VISIBLE_DEVICES="$i" \
        DEVICE_ID=0 \
        ASCEND_DEVICE_ID=0 \
        REDIS_QUEUE_ENABLED="$REDIS_QUEUE_ENABLED" \
        REDIS_HOST="$REDIS_HOST" \
        REDIS_PORT="$REDIS_PORT" \
        REDIS_DB="$REDIS_DB" \
        REDIS_PASSWORD="$REDIS_PASSWORD" \
        REDIS_QUEUE_KEY="$REDIS_QUEUE_KEY" \
        REDIS_PROCESSING_KEY="$REDIS_PROCESSING_KEY" \
        REDIS_TASK_TIMEOUT="$REDIS_TASK_TIMEOUT" \
        nohup python litserve_worker.py \
            --accelerator "$WORKER_ACCELERATOR" \
            --port "$port" \
            --workers-per-device 1 \
            --devices "$i" \
            --mineru-vllm-api-list "[\"${vllm_api}\"]" \
            > "$log_file" 2>&1 &

        local pid=$!
        echo "$pid" > "$pid_file"
        sleep 3

        if ps -p "$pid" > /dev/null 2>&1; then
            log_info "Worker #${i} 已启动 (PID: $pid)"
            started=$((started + 1))
        else
            log_error "Worker #${i} 启动失败，查看: $log_file"
        fi
    done

    log_info "Worker 启动完成: ${started}/${WORKER_NUM_INSTANCES}"

    # 等待初始化
    log_info "等待 Worker 初始化 (10秒)..."
    sleep 10
}

stop_workers() {
    log_step "停止 Workers"

    for i in $(seq 0 $((WORKER_NUM_INSTANCES - 1))); do
        local pid_file="${WORKER_LOG_DIR}/worker_${i}.pid"
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            kill "$pid" 2>/dev/null || true
            rm -f "$pid_file"
        fi
    done
    pkill -f "litserve_worker.py.*81[0-9][1-9]" 2>/dev/null || true
    sleep 2
    log_info "Workers 已全部停止"
}

status_workers() {
    echo -e "${CYAN}Workers (${WORKER_NUM_INSTANCES} 个独立进程)${NC}"
    local running=0
    for i in $(seq 0 $((WORKER_NUM_INSTANCES - 1))); do
        local port=$((WORKER_BASE_PORT + i))
        local pid_file="${WORKER_LOG_DIR}/worker_${i}.pid"
        if [ -f "$pid_file" ] && ps -p "$(cat "$pid_file")" > /dev/null 2>&1; then
            echo "  ✅ Worker #${i} (Port ${port}) - 运行中"
            running=$((running + 1))
        else
            echo "  ❌ Worker #${i} (Port ${port}) - 已停止"
        fi
    done
    echo "  运行中: ${running}/${WORKER_NUM_INSTANCES}"
}

# ============================================================================
# 前端管理
# ============================================================================

start_frontend() {
    log_step "启动 Frontend (端口 ${FRONTEND_PORT})"

    local pid_file="${LOG_DIR}/frontend.pid"

    # 检查是否已运行
    if [ -f "$pid_file" ] && ps -p "$(cat "$pid_file")" > /dev/null 2>&1; then
        log_info "Frontend 已在运行 (PID: $(cat "$pid_file"))"
        return 0
    fi

    # 检查 node_modules
    if [ ! -d "${FRONTEND_DIR}/node_modules" ]; then
        log_info "安装前端依赖..."
        cd "$FRONTEND_DIR" && npm install
    fi

    cd "$FRONTEND_DIR"
    nohup npm run dev > "${LOG_DIR}/frontend.log" 2>&1 &
    local pid=$!
    echo "$pid" > "$pid_file"
    log_info "Frontend 启动中 (PID: $pid)..."

    sleep 3
    if ps -p "$pid" > /dev/null 2>&1; then
        log_info "Frontend 就绪 - http://localhost:${FRONTEND_PORT}"
    else
        log_error "Frontend 启动失败，查看: ${LOG_DIR}/frontend.log"
        return 1
    fi
}

stop_frontend() {
    log_step "停止 Frontend"

    local pid_file="${LOG_DIR}/frontend.pid"
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        # npm run dev 可能产生子进程
        pkill -P "$pid" 2>/dev/null || true
        kill "$pid" 2>/dev/null || true
        rm -f "$pid_file"
    fi
    # 清理所有 vite 相关进程
    pkill -f "vite" 2>/dev/null || true
    log_info "Frontend 已停止"
}

status_frontend() {
    echo -e "${CYAN}Frontend (端口 ${FRONTEND_PORT})${NC}"
    if curl -s "http://localhost:${FRONTEND_PORT}" > /dev/null 2>&1; then
        echo "  ✅ 运行中 - http://localhost:${FRONTEND_PORT}"
    else
        # vite 进程可能在但端口未就绪
        if pgrep -f "vite" > /dev/null 2>&1; then
            echo "  ⏳ 启动中"
        else
            echo "  ❌ 未运行"
        fi
    fi
}

# ============================================================================
# 组合命令
# ============================================================================

cmd_start() {
    local target="${1:-all}"

    separator
    echo -e "${CYAN}  MinerU Tianshu - 启动服务${NC}"
    echo -e "  项目路径: ${PROJECT_ROOT}"
    echo -e "  目标: ${target}"
    separator

    init_dirs

    case "$target" in
        vllm)    start_vllm ;;
        api)     start_api ;;
        mcp)     start_mcp ;;
        worker)  start_workers ;;
        frontend) start_frontend ;;
        all)
            start_vllm
            start_api
            start_mcp
            start_workers
            start_frontend
            ;;
        *) log_error "未知服务: $target (可选: vllm|api|mcp|worker|frontend|all)"; exit 1 ;;
    esac

    echo ""
    separator
    log_info "启动完成"
    separator
}

cmd_stop() {
    local target="${1:-all}"

    case "$target" in
        vllm)    stop_vllm ;;
        api)     stop_api ;;
        mcp)     stop_mcp ;;
        worker)  stop_workers ;;
        frontend) stop_frontend ;;
        all)
            stop_frontend
            stop_workers
            stop_mcp
            stop_api
            stop_vllm
            ;;
        *) log_error "未知服务: $target (可选: vllm|api|mcp|worker|frontend|all)"; exit 1 ;;
    esac
}

cmd_restart() {
    local target="${1:-all}"
    cmd_stop "$target"
    sleep 3
    cmd_start "$target"
}

cmd_status() {
    separator
    echo -e "${CYAN}  MinerU Tianshu - 服务状态${NC}"
    separator
    echo ""
    status_vllm
    echo ""
    status_api
    echo ""
    status_mcp
    echo ""
    status_workers
    echo ""
    status_frontend
    echo ""
    separator
    echo -e "  日志目录: ${LOG_DIR}"
    separator
}

cmd_logs() {
    local target="${1:-all}"

    case "$target" in
        vllm)    tail -f "${VLLM_LOG_DIR}"/*.log ;;
        worker)  tail -f "${WORKER_LOG_DIR}"/*.log ;;
        api)     tail -f "${API_LOG_DIR}"/*.log ;;
        mcp)     tail -f "${API_LOG_DIR}/mcp.log" ;;
        frontend) tail -f "${LOG_DIR}/frontend.log" ;;
        all)     tail -f "${LOG_DIR}"/*/*.log "${LOG_DIR}"/frontend.log ;;
        *) log_error "未知服务: $target (可选: vllm|worker|api|mcp|frontend|all)"; exit 1 ;;
    esac
}

cmd_test() {
    separator
    echo -e "${CYAN}  MinerU Tianshu - 端到端验证${NC}"
    separator

    local pass=0
    local fail=0

    # 1. NPU 设备
    echo ""
    log_info "[1/6] 检查 NPU 设备..."
    if npu-smi info > /dev/null 2>&1; then
        local npu_count=$(npu-smi info -t board -count 2>/dev/null | head -1 || echo "unknown")
        log_info "NPU 设备: ${npu_count}"
        pass=$((pass + 1))
    else
        log_error "npu-smi 不可用"
        fail=$((fail + 1))
    fi

    # 2. VLLM 服务
    echo ""
    log_info "[2/6] 检查 VLLM 服务..."
    local vllm_ok=0
    for i in $(seq 0 $((VLLM_NUM_INSTANCES - 1))); do
        local port=$((VLLM_BASE_PORT + i))
        if curl -s "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            vllm_ok=$((vllm_ok + 1))
        fi
    done
    if [ "$vllm_ok" -eq "$VLLM_NUM_INSTANCES" ]; then
        log_info "VLLM: ${vllm_ok}/${VLLM_NUM_INSTANCES} 就绪"
        pass=$((pass + 1))
    else
        log_error "VLLM: ${vllm_ok}/${VLLM_NUM_INSTANCES} 就绪"
        fail=$((fail + 1))
    fi

    # 3. Worker 进程
    echo ""
    log_info "[3/6] 检查 Worker 进程..."
    local worker_count=$(ps aux | grep "litserve_worker.py" | grep -v grep | wc -l)
    if [ "$worker_count" -ge "$WORKER_NUM_INSTANCES" ]; then
        log_info "Worker: ${worker_count} 个进程运行中"
        pass=$((pass + 1))
    else
        log_error "Worker: 仅 ${worker_count} 个进程 (期望 ${WORKER_NUM_INSTANCES})"
        fail=$((fail + 1))
    fi

    # 4. API Server
    echo ""
    log_info "[4/6] 检查 API Server..."
    if curl -s "http://localhost:${API_PORT}/docs" > /dev/null 2>&1; then
        log_info "API Server: 运行中"
        pass=$((pass + 1))
    else
        log_error "API Server: 未运行"
        fail=$((fail + 1))
    fi

    # 5. 前端
    echo ""
    log_info "[5/6] 检查 Frontend..."
    if curl -s "http://localhost:${FRONTEND_PORT}" > /dev/null 2>&1; then
        log_info "Frontend: 运行中"
        pass=$((pass + 1))
    else
        log_warn "Frontend: 未运行 (可选)"
    fi

    # 6. 数据库
    echo ""
    log_info "[6/6] 检查数据库..."
    if [ -f "$DATABASE_PATH" ]; then
        local db_size=$(du -h "$DATABASE_PATH" | cut -f1)
        log_info "数据库: ${DATABASE_PATH} (${db_size})"
        pass=$((pass + 1))
    else
        log_error "数据库不存在: ${DATABASE_PATH}"
        fail=$((fail + 1))
    fi

    echo ""
    separator
    echo -e "  通过: ${GREEN}${pass}${NC}  失败: ${RED}${fail}${NC}"
    separator
}

cmd_help() {
    cat <<EOF
MinerU Tianshu - 统一启动脚本

使用方式:
  bash scripts/tianshu.sh <命令> [服务]

命令:
  start [服务]   启动服务 (默认: all)
  stop [服务]    停止服务 (默认: all)
  restart        重启所有服务
  status         查看所有服务状态
  logs [服务]    实时查看日志 (默认: all)
  test           端到端验证测试
  help           显示帮助

服务 (可选，不指定则操作全部):
  vllm           VLLM 推理服务 (端口 ${VLLM_BASE_PORT}-${VLLM_BASE_PORT}$((VLLM_NUM_INSTANCES-1)))
  api            API Server (端口 ${API_PORT})
  mcp            MCP Server (端口 ${MCP_PORT})
  worker         Workers (端口 ${WORKER_BASE_PORT}-${WORKER_BASE_PORT}$((WORKER_NUM_INSTANCES-1)))
  frontend       前端界面 (端口 ${FRONTEND_PORT})
  all            所有服务

示例:
  bash scripts/tianshu.sh start           # 启动所有服务
  bash scripts/tianshu.sh start vllm      # 仅启动 VLLM
  bash scripts/tianshu.sh stop worker     # 仅停止 Workers
  bash scripts/tianshu.sh restart         # 重启所有
  bash scripts/tianshu.sh status          # 查看状态
  bash scripts/tianshu.sh logs worker     # 查看 Worker 日志
  bash scripts/tianshu.sh test            # 运行验证测试

配置:
  脚本顶部可修改以下配置:
    VLLM_MODEL_PATH      模型路径
    VLLM_BASE_PORT       VLLM 起始端口
    VLLM_NUM_INSTANCES   VLLM 实例数量
    WORKER_BASE_PORT     Worker 起始端口
    WORKER_NUM_INSTANCES Worker 数量
    DATABASE_PATH        数据库路径
    OUTPUT_PATH          输出路径
EOF
}

# ============================================================================
# 入口
# ============================================================================

case "${1:-help}" in
    start)    cmd_start "$2" ;;
    stop)     cmd_stop "$2" ;;
    restart)  cmd_restart "$2" ;;
    status)   cmd_status ;;
    logs)     cmd_logs "$2" ;;
    test)     cmd_test ;;
    help|--help|-h) cmd_help ;;
    *) log_error "未知命令: $1"; echo ""; cmd_help; exit 1 ;;
esac
