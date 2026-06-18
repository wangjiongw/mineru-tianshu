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
VLLM_READY_TIMEOUT="${VLLM_READY_TIMEOUT:-900}"   # vllm 健康检查总超时(秒);首次含 NPU kernel 编译,默认 15min,可调大

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

# 路径配置 —— 按实例隔离(集群多实例共享 /share 时,各实例在 DATA_ROOT/<实例名>/ 下独立存放)
DATA_ROOT="/share/wangjiong/databases/mineru_database"
INSTANCE_ID="${INSTANCE_ID:-$(hostname)}"               # 实例标识,默认主机名/pod 名;可 export 覆盖
INSTANCE_DATA_DIR="${DATA_ROOT}/${INSTANCE_ID}"         # 本实例的独立数据目录

# 可写数据路径(按实例隔离)
DATABASE_PATH="${INSTANCE_DATA_DIR}/mineru_tianshu.db"
OUTPUT_PATH="${INSTANCE_DATA_DIR}/mineru_outputs"
UPLOAD_PATH="${INSTANCE_DATA_DIR}/mineru_uploads"
LOG_DIR="${INSTANCE_DATA_DIR}/mineru_logs"

# 只读资产(多实例共享,不隔离)
MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/share/wangjiong/model_zoo/modelscope}"

# JWT 认证配置
JWT_EXPIRE_MINUTES="${JWT_EXPIRE_MINUTES:-43200}"  # 默认 30 天（30*24*60）

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
REDIS_QUEUE_KEY="tianshu:task_queue:${INSTANCE_ID}"
REDIS_PROCESSING_KEY="tianshu:processing:${INSTANCE_ID}"
REDIS_TASK_TIMEOUT="3600"

# Redis 进程启动选项（由 start_redis 使用）
REDIS_BIND="${REDIS_BIND:-127.0.0.1}"        # 监听地址（多机部署改 0.0.0.0）
REDIS_APPENDONLY="${REDIS_APPENDONLY:-no}"   # 是否开启 AOF 持久化（no=纯内存，重启丢队列；SQLite 仍是事实源）

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
# 实例冲突检测(多实例共享 DATA_ROOT 时,防止不同实例写同一数据目录)
# ============================================================================

check_instance_conflict() {
    local dir="$INSTANCE_DATA_DIR"
    local lockfile="${dir}/.instance.lock"
    mkdir -p "$dir"
    if [ -f "$lockfile" ]; then
        local owner
        owner=$(head -1 "$lockfile" 2>/dev/null)
        if [ -n "$owner" ] && [ "$owner" != "$INSTANCE_ID" ]; then
            log_error "数据目录已被其他实例占用,中止启动"
            log_error "  目录:       $dir"
            log_error "  占用实例:   $owner"
            log_error "  本实例:     $INSTANCE_ID"
            log_error "解决方法:"
            log_error "  1) 本实例使用唯一 INSTANCE_ID(默认即 hostname),无需额外设置"
            log_error "  2) 若 '$owner' 已确认停止,删除残留锁:  rm $lockfile"
            return 1
        fi
    fi
    echo "$INSTANCE_ID" > "$lockfile"     # 写入/更新本实例锁
    return 0
}

release_instance_lock() {
    local lockfile="${INSTANCE_DATA_DIR}/.instance.lock"
    if [ -f "$lockfile" ] && [ "$(head -1 "$lockfile" 2>/dev/null)" = "$INSTANCE_ID" ]; then
        rm -f "$lockfile"
    fi
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

        # 跳过已运行的实例(端口检测优先:实例隔离使 pid 文件分目录存放,
        # 仅靠 pid 文件会误判"未运行"而重复启动 vllm;端口已就绪即视为在运行)
        if curl -s "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            log_info "VLLM #${i} (NPU=${i}, Port=${port}) 已在运行"
            started=$((started + 1))
            continue
        fi
        if [ -f "$pid_file" ] && ps -p "$(cat "$pid_file")" > /dev/null 2>&1; then
            log_info "VLLM #${i} (NPU=${i}, Port=${port}) 进程存在但端口未就绪,等待中"
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
    for round in $(seq 1 $((VLLM_READY_TIMEOUT / 10))); do
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
    log_warn "VLLM 健康检查超时(${VLLM_READY_TIMEOUT}s): ${healthy}/${VLLM_NUM_INSTANCES} 就绪 (部分可能仍在初始化;可调大 VLLM_READY_TIMEOUT)"
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
    JWT_EXPIRE_MINUTES="$JWT_EXPIRE_MINUTES" \
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

    # 预检查: 等待 VLLM 全部就绪(vllm 加载慢,带超时轮询而非一次探测)
    local vllm_ok=0
    for round in $(seq 1 $((VLLM_READY_TIMEOUT / 10))); do
        vllm_ok=0
        for i in $(seq 0 $((VLLM_NUM_INSTANCES - 1))); do
            local port=$((VLLM_BASE_PORT + i))
            curl -s "http://localhost:${port}/v1/models" > /dev/null 2>&1 && vllm_ok=$((vllm_ok + 1))
        done
        [ "$vllm_ok" -eq "$VLLM_NUM_INSTANCES" ] && break
        log_info "等待 VLLM 就绪 (${vllm_ok}/${VLLM_NUM_INSTANCES})..."
        sleep 10
    done
    if [ "$vllm_ok" -lt "$VLLM_NUM_INSTANCES" ]; then
        log_error "VLLM 未就绪 (${vllm_ok}/${VLLM_NUM_INSTANCES}),请先启动 VLLM 或调大 VLLM_READY_TIMEOUT"
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
        MODELSCOPE_CACHE="$MODELSCOPE_CACHE" \
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
# Redis 管理
# ============================================================================

start_redis() {
    log_step "启动 Redis (端口 ${REDIS_PORT})"

    local pid_file="${LOG_DIR}/redis.pid"
    local log_file="${LOG_DIR}/redis.log"

    # 已运行则跳过
    if [ -f "$pid_file" ] && ps -p "$(cat "$pid_file")" > /dev/null 2>&1; then
        log_info "Redis 已在运行 (PID: $(cat "$pid_file"))"
        return 0
    fi

    # 端口已被监听则视为就绪（兼容外部/手动启动的实例）
    if (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -q ":${REDIS_PORT} "; then
        log_info "Redis 端口 ${REDIS_PORT} 已被监听，视为运行中"
        return 0
    fi

    # 定位 redis-server / redis-cli
    local redis_bin="${REDIS_BIN:-$(command -v redis-server || true)}"
    local redis_cli_bin="${REDIS_CLI_BIN:-$(command -v redis-cli || true)}"
    if [ -z "$redis_bin" ]; then
        log_error "未找到 redis-server，请先安装： mamba install -c conda-forge redis-server"
        return 1
    fi

    mkdir -p "$LOG_DIR"
    log_info "启动 Redis: ${redis_bin} (bind=${REDIS_BIND}, port=${REDIS_PORT}, appendonly=${REDIS_APPENDONLY})"

    "$redis_bin" \
        --daemonize yes \
        --port "$REDIS_PORT" \
        --bind "$REDIS_BIND" \
        --requirepass "$REDIS_PASSWORD" \
        --save "" \
        --appendonly "$REDIS_APPENDONLY" \
        --loglevel warning \
        --pidfile "$pid_file" \
        --logfile "$log_file"

    # 健康检查（鉴权 PING）
    local ok=0
    if [ -n "$redis_cli_bin" ]; then
        for _ in $(seq 1 15); do
            if "$redis_cli_bin" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning ping 2>/dev/null | grep -q "PONG"; then
                ok=1
                break
            fi
            sleep 1
        done
    fi

    if [ "$ok" -eq 1 ]; then
        log_info "Redis 就绪 (端口 ${REDIS_PORT})"
    else
        log_error "Redis 启动后鉴权失败，查看: $log_file"
        return 1
    fi
}

stop_redis() {
    log_step "停止 Redis"

    local pid_file="${LOG_DIR}/redis.pid"
    local redis_cli_bin="${REDIS_CLI_BIN:-$(command -v redis-cli || true)}"

    # 优先 redis-cli 优雅关闭
    if [ -n "$redis_cli_bin" ]; then
        "$redis_cli_bin" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning shutdown nosave 2>/dev/null || true
    fi
    sleep 1

    # 兜底：pidfile kill
    if [ -f "$pid_file" ]; then
        local pid
        pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            kill "$pid" 2>/dev/null || true
            sleep 1
            ps -p "$pid" > /dev/null 2>&1 && kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$pid_file"
    fi

    pkill -f "redis-server.*:${REDIS_PORT}" 2>/dev/null || true
    log_info "Redis 已停止"
}

status_redis() {
    echo -e "${CYAN}Redis (端口 ${REDIS_PORT})${NC}"
    local redis_cli_bin="${REDIS_CLI_BIN:-$(command -v redis-cli || true)}"
    if [ -n "$redis_cli_bin" ] && \
       "$redis_cli_bin" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning ping 2>/dev/null | grep -q "PONG"; then
        local clients
        clients=$("$redis_cli_bin" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning info clients 2>/dev/null | grep "^connected_clients:" | cut -d: -f2 | tr -d '\r')
        echo "  ✅ 运行中 - localhost:${REDIS_PORT} (connected_clients: ${clients:-?})"
    else
        echo "  ❌ 未运行"
    fi
}

# ============================================================================
# 组合命令
# ============================================================================

cmd_start() {
    local target="${1:-all}"
    local rc=0

    separator
    echo -e "${CYAN}  MinerU Tianshu - 启动服务${NC}"
    echo -e "  项目路径: ${PROJECT_ROOT}"
    echo -e "  目标: ${target}"
    separator

    init_dirs

    # 多实例冲突检测:确保本数据目录未被别的活跃实例占用
    check_instance_conflict || return 1

    case "$target" in
        redis)   start_redis    || rc=1 ;;
        vllm)    start_vllm     || rc=1 ;;
        api)     start_api      || rc=1 ;;
        mcp)     start_mcp      || rc=1 ;;
        worker)  start_workers  || rc=1 ;;
        frontend) start_frontend || rc=1 ;;
        all)
            start_redis    || rc=1
            start_vllm     || rc=1
            start_api      || rc=1
            start_mcp      || rc=1
            start_workers  || rc=1
            start_frontend || rc=1
            ;;
        *) log_error "未知服务: $target (可选: vllm|api|redis|mcp|worker|frontend|all)"; exit 1 ;;
    esac

    echo ""
    separator
    if [ "$rc" -eq 0 ]; then
        log_info "启动完成"
    else
        log_error "启动完成,但有服务失败 (见上方日志;用 'logs <服务>' 排查)"
    fi
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
        redis)   stop_redis ;;
        all)
            stop_frontend
            stop_workers
            stop_mcp
            stop_api
            stop_vllm
            stop_redis
            ;;
        *) log_error "未知服务: $target (可选: vllm|api|redis|mcp|worker|frontend|all)"; exit 1 ;;
    esac

    # 仅在停止整个实例(all)时释放数据目录锁;单服务停止不释放(api 仍占用目录)
    if [ "$target" = "all" ]; then
        release_instance_lock
    fi
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
    echo -e "  实例: ${INSTANCE_ID}    数据目录: ${INSTANCE_DATA_DIR}"
    echo ""
    status_redis
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
        redis)   tail -f "${LOG_DIR}/redis.log" ;;
        vllm)    tail -f "${VLLM_LOG_DIR}"/*.log ;;
        worker)  tail -f "${WORKER_LOG_DIR}"/*.log ;;
        api)     tail -f "${API_LOG_DIR}"/*.log ;;
        mcp)     tail -f "${API_LOG_DIR}/mcp.log" ;;
        frontend) tail -f "${LOG_DIR}/frontend.log" ;;
        all)     tail -f "${LOG_DIR}"/*/*.log "${LOG_DIR}"/frontend.log "${LOG_DIR}"/redis.log ;;
        *) log_error "未知服务: $target (可选: vllm|worker|api|mcp|redis|frontend|all)"; exit 1 ;;
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
  redis          Redis 队列服务 (端口 ${REDIS_PORT})
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
    DATA_ROOT            数据根(所有实例在 <DATA_ROOT>/<INSTANCE_ID>/ 下隔离存放)
    INSTANCE_ID          实例标识(默认 hostname;export 覆盖可指定特定实例)
    VLLM_MODEL_PATH      模型路径
    VLLM_BASE_PORT       VLLM 起始端口
    VLLM_NUM_INSTANCES   VLLM 实例数量
    VLLM_READY_TIMEOUT   VLLM 健康检查超时秒数(默认 900;首次加载/kernel 编译慢可调大)
    WORKER_BASE_PORT     Worker 起始端口
    WORKER_NUM_INSTANCES Worker 数量
    DATABASE_PATH        数据库路径(自动派生自 INSTANCE_DATA_DIR)
    OUTPUT_PATH          输出路径(自动派生)
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
