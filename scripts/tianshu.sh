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
VLLM_MODEL_PATH="/share/wangjiong/model_zoo/modelscope/models/OpenDataLab/MinerU2___5-Pro-2605-1___2B"
VLLM_BASE_PORT=30025
VLLM_NUM_INSTANCES=8
VLLM_MAX_MODEL_LEN=8192
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.60}" # 实测 c8 峰值 HBM < 86%，为 8 个任务进程保留安全余量
VLLM_PERFORMANCE_MODE="${VLLM_PERFORMANCE_MODE:-throughput}" # 吞吐模式实测比 balanced 提升约 11%；可用 NPU<n> 覆盖
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-4096}" # 8192 实测降低 PDF 吞吐，保留 4096
#                                                                # hybrid/VLM 推理走本卡 vLLM；0.60 + c8 在 10 分钟压力下 0 OOM/0 preemption
VLLM_READY_TIMEOUT="${VLLM_READY_TIMEOUT:-900}"   # vllm 健康检查总超时(秒);首次含 NPU kernel 编译,默认 15min,可调大
WORKER_READY_TIMEOUT="${WORKER_READY_TIMEOUT:-180}" # worker cold-start /health 等待超时(秒)
SUPERVISOR_HEALTH_PROBE_TIMEOUT_SECONDS="${SUPERVISOR_HEALTH_PROBE_TIMEOUT_SECONDS:-5}"
COMPUTE_SUPERVISOR_POLL_SECONDS="${COMPUTE_SUPERVISOR_POLL_SECONDS:-5}"
COMPUTE_SUPERVISOR_HEALTH_INTERVAL_SECONDS="${COMPUTE_SUPERVISOR_HEALTH_INTERVAL_SECONDS:-15}"
COMPUTE_SUPERVISOR_HEALTH_FAILURE_THRESHOLD="${COMPUTE_SUPERVISOR_HEALTH_FAILURE_THRESHOLD:-3}"
COMPUTE_SUPERVISOR_HEARTBEAT_SECONDS="${COMPUTE_SUPERVISOR_HEARTBEAT_SECONDS:-300}"
# 锁定 vllm 可执行文件到绝对路径:其 shebang 指向 /usr/local/python3.11.15/bin/python3,
# 独立于当前 shell 的 PATH 与默认 python(默认 python 仍保持 conda mineru 环境)。
# 避免 PATH 漂移导致不同实例加载不同版本的 vllm_ascend —— 例如 conda mineru 环境里的
# vllm 0.13.0 与 CANN 9.0.0 不兼容会编译失败,而此处的 0.20.2 已验证可用。
VLLM_BIN="${VLLM_BIN:-/usr/local/python3.11.15/bin/vllm}"
# Worker / API / MCP 用的 Python:锁定到 conda mineru 环境(magic_pdf、paddleocr 等 mineru
# pipeline 依赖只装在这里)。与 VLLM_BIN 互补 —— 两套 Python 环境各司其职,均不依赖当前 shell 的 PATH。
PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/mineru/bin/python}"

# Worker 配置
WORKER_BASE_PORT=8101
WORKER_NUM_INSTANCES=8
WORKER_ACCELERATOR="cpu"
MINERU_INTRA_OP_NUM_THREADS="${MINERU_INTRA_OP_NUM_THREADS:-4}"
MINERU_INTER_OP_NUM_THREADS="${MINERU_INTER_OP_NUM_THREADS:-1}"
# 本机未配置 RUSTFS_PUBLIC_URL 时关闭无效上传；显式 RUSTFS_ENABLED=true 可恢复远端上传。
RUSTFS_ENABLED="${RUSTFS_ENABLED:-false}"
# 本机 vLLM 由本脚本直接托管，不需要每个任务探测 Docker socket。
VLLM_DOCKER_CONTROLLER_ENABLED="${VLLM_DOCKER_CONTROLLER_ENABLED:-false}"

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
RUNTIME_CONFIG_DIR="${RUNTIME_CONFIG_DIR:-${INSTANCE_DATA_DIR}/runtime_config}"
# Triton kernel JIT 编译缓存根目录(按实例隔离)。
# 每个 vLLM 实例分配独立子目录(npu0, npu1, ...)，避免多实例并发编译同一 kernel 时
# 在共享缓存目录产生临时目录清理竞态(MLIRCompilationError: [Errno 39] Directory not empty)。
# 放在 /tmp 下以缩短路径(原 INSTANCE_DATA_DIR 路径 + kernel hash 超 107 字符会触发 socket path limit)。
# /tmp 是 per-pod 的,天然跨 POD 隔离;重启后丢失只导致一次性重新编译,无数据损失。
TRITON_CACHE_ROOT="${TRITON_CACHE_ROOT:-/tmp/triton}"

# vLLM AOT 编译缓存根目录(同样按实例 + NPU 隔离)。
# 默认 vLLM 用 ~/.cache/vllm —— 但本集群 /data/persist_home/wangjiong 跨多 POD 共享,
# 导致 8 NPU × N POD 的编译缓存全堆在同一 hash 目录里并发读写,触发:
#   1) 跨 POD 加载彼此的缓存失败(CANN/load 环境细微差异) → 回退重新编译
#   2) 并发编译写同一目录 → 损坏概率高
# /tmp 是 per-POD 的,天然隔离;每个 NPU 一个子目录避免 8 实例并发写。
VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/tmp/vllm_cache}"

# 只读资产(多实例共享,不隔离)
MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/share/wangjiong/model_zoo/modelscope}"

# JWT 认证配置
JWT_EXPIRE_MINUTES="${JWT_EXPIRE_MINUTES:-43200}"  # 默认 30 天（30*24*60）

# 日志子目录
VLLM_LOG_DIR="${LOG_DIR}/vllm"
WORKER_LOG_DIR="${LOG_DIR}/worker"
API_LOG_DIR="${LOG_DIR}/api"
SCHEDULER_LOG_DIR="${LOG_DIR}/scheduler"
WORKER_RUNTIME_DIR="${WORKER_RUNTIME_DIR:-/tmp/mineru_tianshu/${INSTANCE_ID}}"
SUPERVISED_WORKER_RESTART_TIMEOUT_SECONDS="${SUPERVISED_WORKER_RESTART_TIMEOUT_SECONDS:-600}"

# Redis 队列配置
REDIS_QUEUE_ENABLED="true"
REDIS_HOST="localhost"
REDIS_PORT="6379"
REDIS_DB="0"
REDIS_PASSWORD="redis123"
REDIS_QUEUE_KEY="tianshu:task_queue:${INSTANCE_ID}"
REDIS_PROCESSING_KEY="tianshu:processing:${INSTANCE_ID}"
REDIS_CLAIM_MAINTENANCE_KEY="tianshu:claim_maintenance:${INSTANCE_ID}"
REDIS_CLAIM_PAUSE_KEY="tianshu:claim_pause:${INSTANCE_ID}"
SQLITE_QUEUE_FALLBACK="${SQLITE_QUEUE_FALLBACK:-false}"
REDIS_TASK_TIMEOUT="3600"

# Redis 进程启动选项（由 start_redis 使用）
REDIS_BIND="${REDIS_BIND:-127.0.0.1}"        # 监听地址（多机部署改 0.0.0.0）
REDIS_APPENDONLY="${REDIS_APPENDONLY:-no}"   # 是否开启 AOF 持久化（no=纯内存，重启丢队列；SQLite 仍是事实源）

# 调度器（孤儿任务恢复 + 队列监控）
SCHEDULER_MONITOR_INTERVAL="${SCHEDULER_MONITOR_INTERVAL:-300}"   # 监控周期(秒,默认 5 分钟)
SCHEDULER_HEALTH_INTERVAL="${SCHEDULER_HEALTH_INTERVAL:-900}"     # 健康检查周期(秒,默认 15 分钟)
SCHEDULER_STALE_TIMEOUT="${SCHEDULER_STALE_TIMEOUT:-10}"          # 孤儿任务判定阈值(分钟,默认 10)
SCHEDULER_CLEANUP_DAYS="${SCHEDULER_CLEANUP_DAYS:-0}"             # 旧任务文件清理(天, 0=禁用;批量处理时必须关闭否则边跑边删)
SCHEDULER_ORPHAN_RECOVERY_APPLY="${SCHEDULER_ORPHAN_RECOVERY_APPLY:-true}" # 本机批处理默认实际恢复; false 可退回审计模式
SCHEDULER_ORPHAN_RECOVERY_BATCH_SIZE="${SCHEDULER_ORPHAN_RECOVERY_BATCH_SIZE:-500}"

# PDF 自动拆分（降低单次推理峰值内存，缓解 worker 原生崩溃）
PDF_SPLIT_ENABLED="${PDF_SPLIT_ENABLED:-true}"
PDF_SPLIT_THRESHOLD_PAGES="${PDF_SPLIT_THRESHOLD_PAGES:-50}"      # 原默认 500，下调以降低峰值内存
PDF_SPLIT_CHUNK_SIZE="${PDF_SPLIT_CHUNK_SIZE:-50}"

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

pid_cmdline() {
    local pid="$1"
    local proc_root="${TIANSHU_PROC_ROOT:-/proc}"
    local cmdline_file="${proc_root}/${pid}/cmdline"
    [ -r "$cmdline_file" ] || return 1
    { tr '\0' ' ' < "$cmdline_file"; } 2>/dev/null
}

pid_state() {
    local pid="$1"
    local proc_root="${TIANSHU_PROC_ROOT:-/proc}"
    awk '/^State:/ {print $2}' "${proc_root}/${pid}/status" 2>/dev/null
}

pid_start_time() {
    local pid="$1"
    local proc_root="${TIANSHU_PROC_ROOT:-/proc}"
    local stat_file="${proc_root}/${pid}/stat"
    [ -r "$stat_file" ] || return 1
    awk '{print $22}' "$stat_file" 2>/dev/null
}

pid_identity_matches() {
    local pid="$1"
    local expected_start_time="$2"
    local current_start_time
    local state

    current_start_time="$(pid_start_time "$pid")"
    [ -n "$current_start_time" ] && [ "$current_start_time" = "$expected_start_time" ] || return 1
    state="$(pid_state "$pid")"
    [ -n "$state" ] && [ "$state" != "Z" ]
}

pid_matches() {
    local pid="$1"
    local expected="$2"
    local expected_port="${3:-}"
    local cmdline
    local state

    case "$pid" in
        ''|*[!0-9]*) return 1 ;;
    esac

    state="$(pid_state "$pid")"
    [ "$state" = "Z" ] && return 1

    cmdline="$(pid_cmdline "$pid")"
    [ -n "$cmdline" ] || return 1
    printf '%s\n' "$cmdline" | grep -Eq -- "$expected" || return 1
    if [ -n "$expected_port" ]; then
        printf '%s\n' "$cmdline" | grep -Eq -- "(^|[[:space:]])--port[=[:space:]]${expected_port}($|[[:space:]])" || return 1
    fi

    return 0
}

pid_file_matches() {
    local pid_file="$1"
    local expected="$2"
    local expected_port="${3:-}"
    [ -f "$pid_file" ] || return 1
    pid_matches "$(cat "$pid_file" 2>/dev/null)" "$expected" "$expected_port"
}

terminate_pid_file() {
    local pid_file="$1"
    local expected="$2"
    local expected_port="${3:-}"
    local timeout="${4:-5}"
    local pid

    [ -f "$pid_file" ] || return 0
    pid="$(cat "$pid_file" 2>/dev/null)"
    if pid_matches "$pid" "$expected" "$expected_port"; then
        kill "$pid" 2>/dev/null || true
        for _ in $(seq 1 "$timeout"); do
            pid_matches "$pid" "$expected" "$expected_port" || break
            sleep 1
        done
        if pid_matches "$pid" "$expected" "$expected_port"; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    fi
    rm -f "$pid_file"
}

# ============================================================================
# 目录初始化
# ============================================================================

init_dirs() {
    mkdir -p "$VLLM_LOG_DIR" "$WORKER_LOG_DIR" "$API_LOG_DIR" "$SCHEDULER_LOG_DIR" \
             "$RUNTIME_CONFIG_DIR" "$OUTPUT_PATH" "$UPLOAD_PATH" \
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
# CANN / Ascend 环境自动加载
# ============================================================================
# tianshu.sh 启动的所有服务(vllm/api/worker/mcp/scheduler)都依赖 CANN 算子库
# (libopapi.so / libccec.so / libascendcl.so),这些 .so 的路径不在默认 linker
# 搜索路径里,必须 source CANN 的 set_env.sh 让 LD_LIBRARY_PATH 包含
# /usr/local/Ascend/ascend-toolkit/latest/lib64 及 plugin 子目录。
#
# 历史上依赖登录 shell(/etc/profile.d/cann.sh 或 ~/.bashrc)隐式加载,新机器若
# 没配 profile 就会导致 vllm 启动时报:
#   RuntimeError: aclnnXxx ... not in libopapi.so, or libopapi.so not found
# 这里改为显式 source,不再依赖外部 profile 配置;nohup 启动的子进程会继承 export。

ASCEND_SET_ENV="${ASCEND_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
# vllm-ascend 的 ATB(Ascend Transformer Boost)库 env;若存在则一并 source
ASCEND_ATB_SET_ENV="${ASCEND_ATB_SET_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}"

# vllm-ascend 的 custom transformer 算子包路径(包含 aclnnAddRmsNormBias 等融合算子)。
# 这些算子不在 CANN 内置 libopapi.so 里,而是 vllm-ascend 自己编译的 libcust_opapi.so。
# vllm_ascend.__init__ 默认只 export ASCEND_CUSTOM_OPP_PATH,不加 LD_LIBRARY_PATH;
# 在 fusion pass 注册期间(torch.compiler.is_compiling()=True)兜底分支被禁用,
# 导致首次编译时报 "aclnnAddRmsNormBias not in libopapi.so"。这里预先把算子库路径
# 加到 LD_LIBRARY_PATH + ASCEND_CUSTOM_OPP_PATH,绕过兜底依赖。
#
# 候选路径按优先级(运行时选第一个普通用户也能 traverse 的):
#   1) vllm_ascend/_cann_ops_custom/vendors/custom_transformer —— 标准"已安装"位置
#      (root 属主的 .run 安装生成,目录默认 0750;非 root 用户对 [ -d ] 检测会失败)
#   2) csrc/build/_CPack_Packages/.../packages/vendors/custom_transformer —— CPack 打包暂存区
#      (构建副产物,通常 0755 全可读,含同一份 libcust_opapi.so,符号完整)
VLLM_ASCEND_HOME="${VLLM_ASCEND_HOME:-/vllm-workspace/vllm-ascend}"
VLLM_ASCEND_VENDOR_DIR=""
for _cand in \
    "${VLLM_ASCEND_HOME}/vllm_ascend/_cann_ops_custom/vendors/custom_transformer" \
    "${VLLM_ASCEND_HOME}/csrc/build/_CPack_Packages/Linux/External/cann-ops-transformer-custom_linux-aarch64.run/packages/vendors/custom_transformer"
do
    if [ -d "${_cand}/op_api/lib" ]; then
        VLLM_ASCEND_VENDOR_DIR="$_cand"
        break
    fi
done
unset _cand

source_ascend_env() {
    # 跳过重复 source(避免 PATH / LD_LIBRARY_PATH 累积污染)
    if [ -n "${TIANSHU_ASCEND_ENV_LOADED:-}" ]; then
        return 0
    fi

    local found=0
    if [ -f "$ASCEND_SET_ENV" ]; then
        # shellcheck disable=SC1090
        source "$ASCEND_SET_ENV" >/dev/null 2>&1 || true
        found=1
    fi
    # ATB env(vllm-ascend 依赖)—— 若存在则加载
    if [ -f "$ASCEND_ATB_SET_ENV" ]; then
        # shellcheck disable=SC1090
        source "$ASCEND_ATB_SET_ENV" >/dev/null 2>&1 || true
    fi

    # vllm-ascend custom 算子库 —— 让动态链接器在 fusion pass 注册阶段就能
    # 找到 libcust_opapi.so(含 aclnnAddRmsNormBias 等融合算子实现)
    if [ -n "${VLLM_ASCEND_VENDOR_DIR}" ] && [ -d "${VLLM_ASCEND_VENDOR_DIR}/op_api/lib" ]; then
        local vendor_lib_path="${VLLM_ASCEND_VENDOR_DIR}/op_api/lib"
        case ":${LD_LIBRARY_PATH:-}:" in
            *":${vendor_lib_path}:"*) ;;  # 已存在,跳过
            *) export LD_LIBRARY_PATH="${vendor_lib_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
        esac
        case ":${ASCEND_CUSTOM_OPP_PATH:-}:" in
            *":${VLLM_ASCEND_VENDOR_DIR}:"*) ;;  # 已存在,跳过
            *) export ASCEND_CUSTOM_OPP_PATH="${VLLM_ASCEND_VENDOR_DIR}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}" ;;
        esac
        log_info "vllm-ascend custom 算子库: ${VLLM_ASCEND_VENDOR_DIR}"
    else
        log_warn "vllm-ascend custom 算子库(libcust_opapi.so)未找到可读路径"
        log_warn "若 vllm 启动报 'aclnnAddRmsNormBias not in libopapi.so',请检查 vllm-ascend 安装/权限"
    fi

    if [ "$found" -eq 1 ]; then
        export TIANSHU_ASCEND_ENV_LOADED=1
        log_info "CANN 环境已加载 ($ASCEND_SET_ENV)"
    else
        log_warn "CANN set_env.sh 未找到: $ASCEND_SET_ENV"
        log_warn "若 vllm 启动报 'aclnnXxx not in libopapi.so',请检查 CANN 安装路径"
    fi
}

indexed_env_value() {
    local prefix="$1"
    local index="$2"
    local fallback="$3"
    local name="${prefix}${index}"
    local value="${!name:-}"
    if [ -n "$value" ]; then
        echo "$value"
    else
        echo "$fallback"
    fi
}

vllm_gpu_memory_utilization_for() {
    indexed_env_value "VLLM_GPU_MEMORY_UTILIZATION_NPU" "$1" "$VLLM_GPU_MEMORY_UTILIZATION"
}

vllm_performance_mode_for() {
    indexed_env_value "VLLM_PERFORMANCE_MODE_NPU" "$1" "$VLLM_PERFORMANCE_MODE"
}

vllm_max_num_batched_tokens_for() {
    indexed_env_value "VLLM_MAX_NUM_BATCHED_TOKENS_NPU" "$1" "$VLLM_MAX_NUM_BATCHED_TOKENS"
}

worker_runtime_config_file() { echo "${RUNTIME_CONFIG_DIR}/worker_${1}.env"; }

worker_config_revision_for() {
    local config_file
    config_file="$(worker_runtime_config_file "$1")"
    [ -f "$config_file" ] || { echo "none"; return 0; }
    cksum "$config_file" 2>/dev/null | awk '{print $1}'
}

worker_runtime_config_value() {
    local index="$1"
    local wanted="$2"
    local config_file name value
    config_file="$(worker_runtime_config_file "$index")"
    [ -f "$config_file" ] || return 1
    while IFS="=" read -r name value; do
        case "$name" in
            MINERU_HYBRID_BATCH_RATIO|MAX_CONCURRENT_TASKS) ;;
            *) continue ;;
        esac
        [ "$name" = "$wanted" ] || continue
        case "$name:$value" in
            MINERU_HYBRID_BATCH_RATIO:1|MINERU_HYBRID_BATCH_RATIO:2|MINERU_HYBRID_BATCH_RATIO:4|MINERU_HYBRID_BATCH_RATIO:8) echo "$value"; return 0 ;;
            MAX_CONCURRENT_TASKS:*)
                case "$value" in ""|*[!0-9]*) return 1 ;; esac
                [ "$value" -ge 1 ] && [ "$value" -le 16 ] || return 1
                echo "$value"
                return 0
                ;;
        esac
    done < "$config_file"
    return 1
}

worker_hybrid_batch_ratio_valid() {
    case "$1" in 1|2|4|8) return 0 ;; *) return 1 ;; esac
}

worker_hybrid_batch_ratio_for() {
    local index="$1"
    local value
    if value="$(worker_runtime_config_value "$index" "MINERU_HYBRID_BATCH_RATIO")"; then
        worker_hybrid_batch_ratio_valid "$value" || return 1
        echo "$value"
        return 0
    fi
    value="$(indexed_env_value "MINERU_HYBRID_BATCH_RATIO_WORKER" "$index" "${MINERU_HYBRID_BATCH_RATIO:-}")"
    [ -n "$value" ] || return 1
    worker_hybrid_batch_ratio_valid "$value" || return 1
    echo "$value"
}

worker_max_concurrent_tasks_for() {
    worker_runtime_config_value "$1" "MAX_CONCURRENT_TASKS" || indexed_env_value "MAX_CONCURRENT_TASKS_WORKER" "$1" "${MAX_CONCURRENT_TASKS:-8}"
}

configure_worker_instance() {
    local i="$1"
    shift || true
    local ratio=""
    local max_tasks=""
    worker_index_valid "$i" || { log_error "无效 Worker 编号: $i"; return 1; }
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --hybrid-batch-ratio) ratio="$2"; shift 2 ;;
            --max-concurrent-tasks) max_tasks="$2"; shift 2 ;;
            *) log_error "未知 configure worker 参数: $1"; return 1 ;;
        esac
    done
    case "$ratio" in 1|2|4|8) ;; *) log_error "--hybrid-batch-ratio 仅允许 1|2|4|8"; return 1 ;; esac
    case "$max_tasks" in ""|*[!0-9]*) log_error "--max-concurrent-tasks 需要 1..16"; return 1 ;; esac
    [ "$max_tasks" -ge 1 ] && [ "$max_tasks" -le 16 ] || { log_error "--max-concurrent-tasks 需要 1..16"; return 1; }

    mkdir -p "$RUNTIME_CONFIG_DIR"
    local config_file tmp_file
    config_file="$(worker_runtime_config_file "$i")"
    tmp_file="${config_file}.tmp.$$"
    {
        printf "MINERU_HYBRID_BATCH_RATIO=%s\n" "$ratio"
        printf "MAX_CONCURRENT_TASKS=%s\n" "$max_tasks"
    } > "$tmp_file" || return 1
    chmod 600 "$tmp_file" 2>/dev/null || true
    mv "$tmp_file" "$config_file"
    log_info "Worker #${i} runtime config saved: $config_file (revision $(worker_config_revision_for "$i"))"
}

vllm_index_valid() {
    local index="$1"
    case "$index" in
        ''|*[!0-9]*) return 1 ;;
    esac
    [ "$index" -ge 0 ] && [ "$index" -lt "$VLLM_NUM_INSTANCES" ]
}

vllm_pid_file() { echo "${VLLM_LOG_DIR}/vllm_npu${1}.pid"; }
vllm_expected_cmd() { echo "vllm.*serve.*${VLLM_MODEL_PATH}"; }

vllm_http_ready() {
    local index="$1"
    local timeout="${2:-$SUPERVISOR_HEALTH_PROBE_TIMEOUT_SECONDS}"
    vllm_index_valid "$index" || return 1
    local port=$((VLLM_BASE_PORT + index))
    curl -fsS --max-time "$timeout" "http://localhost:${port}/v1/models" > /dev/null 2>&1
}

vllm_http_healthy() {
    local index="$1"
    local timeout="${2:-$SUPERVISOR_HEALTH_PROBE_TIMEOUT_SECONDS}"
    vllm_index_valid "$index" || return 1
    local port=$((VLLM_BASE_PORT + index))
    curl -fsS --max-time "$timeout" "http://localhost:${port}/health" > /dev/null 2>&1
}

vllm_is_running() {
    local index="$1"
    local port=$((VLLM_BASE_PORT + index))
    pid_file_matches "$(vllm_pid_file "$index")" "$(vllm_expected_cmd)" "$port"
}

wait_vllm_instance_ready() {
    local index="$1"
    local timeout="${2:-$VLLM_READY_TIMEOUT}"
    local elapsed=0

    while [ "$elapsed" -lt "$timeout" ]; do
        vllm_is_running "$index" || return 1
        if vllm_http_ready "$index"; then
            return 0
        fi
        sleep 10
        elapsed=$((elapsed + 10))
    done
    return 1
}

start_vllm_instance() {
    local i="$1"
    vllm_index_valid "$i" || { log_error "无效 VLLM 编号: $i"; return 1; }

    local port=$((VLLM_BASE_PORT + i))
    local log_file="${VLLM_LOG_DIR}/vllm_npu${i}_port${port}.log"
    local pid_file
    pid_file="$(vllm_pid_file "$i")"
    local gpu_memory_utilization
    local performance_mode
    local max_num_batched_tokens
    gpu_memory_utilization="$(vllm_gpu_memory_utilization_for "$i")"
    performance_mode="$(vllm_performance_mode_for "$i")"
    max_num_batched_tokens="$(vllm_max_num_batched_tokens_for "$i")"

    mkdir -p "$VLLM_LOG_DIR"
    if vllm_http_ready "$i"; then
        log_info "VLLM #${i} (NPU=${i}, Port=${port}) 已在运行"
        return 0
    fi
    if vllm_is_running "$i"; then
        log_info "VLLM #${i} (NPU=${i}, Port=${port}) 进程存在但端口未就绪,等待中"
        wait_vllm_instance_ready "$i" "$VLLM_READY_TIMEOUT"
        return $?
    fi
    rm -f "$pid_file"

    log_info "启动 VLLM #${i}: NPU=${i}, Port=${port}"

    local triton_cache_dir="${TRITON_CACHE_ROOT}/npu${i}"
    local vllm_tmp_dir="/tmp/vllm_tmp_npu${i}"
    local vllm_cache_dir="${VLLM_CACHE_ROOT}/npu${i}"
    mkdir -p "$triton_cache_dir" "$vllm_tmp_dir" "$vllm_cache_dir"

    nohup bash -lc "
        [ -f '${ASCEND_SET_ENV}' ] && source '${ASCEND_SET_ENV}' >/dev/null 2>&1 || true
        [ -f '${ASCEND_ATB_SET_ENV}' ] && source '${ASCEND_ATB_SET_ENV}' >/dev/null 2>&1 || true
        if [ -d '${VLLM_ASCEND_VENDOR_DIR}/op_api/lib' ]; then
            case \":\$LD_LIBRARY_PATH:\" in
                *\":${VLLM_ASCEND_VENDOR_DIR}/op_api/lib:\"*) ;;
                *) export LD_LIBRARY_PATH='${VLLM_ASCEND_VENDOR_DIR}/op_api/lib':\$LD_LIBRARY_PATH ;;
            esac
            case \":\$ASCEND_CUSTOM_OPP_PATH:\" in
                *\":${VLLM_ASCEND_VENDOR_DIR}:\"*) ;;
                *) export ASCEND_CUSTOM_OPP_PATH='${VLLM_ASCEND_VENDOR_DIR}':\$ASCEND_CUSTOM_OPP_PATH ;;
            esac
        fi
        export ASCEND_VISIBLE_DEVICES='${i}'
        export ASCEND_RT_VISIBLE_DEVICES='${i}'
        export DEVICE_ID=0
        export ASCEND_DEVICE_ID=0
        export TRITON_CACHE_DIR='${triton_cache_dir}'
        export VLLM_CACHE_ROOT='${vllm_cache_dir}'
        export TMPDIR='${vllm_tmp_dir}'
        exec ${VLLM_BIN} serve '${VLLM_MODEL_PATH}' \
            --host 0.0.0.0 \
            --tensor-parallel-size 1 \
            --port ${port} \
            --max-model-len ${VLLM_MAX_MODEL_LEN} \
            --gpu-memory-utilization ${gpu_memory_utilization} \
            --performance-mode ${performance_mode} \
            --max-num-batched-tokens ${max_num_batched_tokens} \
            --dtype float16 \
            --trust-remote-code
    " > "$log_file" 2>&1 &

    local pid=$!
    echo "$pid" > "$pid_file"
    sleep 2

    if ! pid_matches "$pid" "$(vllm_expected_cmd)" "$port"; then
        log_error "VLLM #${i} 启动失败，查看: $log_file"
        rm -f "$pid_file"
        return 1
    fi

    log_info "VLLM #${i} 已启动 (PID: $pid)，等待健康检查"
    if wait_vllm_instance_ready "$i" "$VLLM_READY_TIMEOUT"; then
        log_info "VLLM #${i} 就绪"
        return 0
    fi
    log_warn "VLLM #${i} 健康检查超时(${VLLM_READY_TIMEOUT}s)，可能仍在初始化"
    return 1
}

stop_vllm_process_tree() {
    local pid_file="$1"
    local port="$2"
    local timeout="${3:-5}"
    local parent_pid=""
    local descendant_pids=""
    local descendant_identities=""
    local identity start_time pid survivors

    if [ -f "$pid_file" ]; then
        parent_pid="$(cat "$pid_file" 2>/dev/null)"
    fi

    if pid_matches "$parent_pid" "$(vllm_expected_cmd)" "$port"; then
        descendant_pids="$(worker_descendant_pids "$parent_pid")"
        for pid in $descendant_pids; do
            start_time="$(pid_start_time "$pid")"
            [ -n "$start_time" ] || continue
            descendant_identities="$descendant_identities $pid:$start_time"
        done
        kill "$parent_pid" 2>/dev/null || true

        for _ in $(seq 1 "$timeout"); do
            pid_matches "$parent_pid" "$(vllm_expected_cmd)" "$port" || break
            sleep 1
        done
        if pid_matches "$parent_pid" "$(vllm_expected_cmd)" "$port"; then
            kill -9 "$parent_pid" 2>/dev/null || true
        fi

        for identity in $descendant_identities; do
            pid="${identity%%:*}"
            start_time="${identity#*:}"
            pid_identity_matches "$pid" "$start_time" && kill "$pid" 2>/dev/null || true
        done
        for _ in $(seq 1 "$timeout"); do
            survivors=""
            for identity in $descendant_identities; do
                pid="${identity%%:*}"
                start_time="${identity#*:}"
                pid_identity_matches "$pid" "$start_time" && survivors="$survivors $pid"
            done
            [ -z "$survivors" ] && break
            sleep 1
        done
        for identity in $descendant_identities; do
            pid="${identity%%:*}"
            start_time="${identity#*:}"
            pid_identity_matches "$pid" "$start_time" && kill -9 "$pid" 2>/dev/null || true
        done
        survivors=""
        for identity in $descendant_identities; do
            pid="${identity%%:*}"
            start_time="${identity#*:}"
            pid_identity_matches "$pid" "$start_time" && survivors="$survivors $pid"
        done
        if [ -n "$survivors" ]; then
            log_error "VLLM Port ${port} 停止后仍存在已捕获子孙进程: ${survivors}"
            rm -f "$pid_file"
            return 1
        fi
    fi

    rm -f "$pid_file"
    return 0
}

stop_vllm_instance() {
    local i="$1"
    vllm_index_valid "$i" || { log_error "无效 VLLM 编号: $i"; return 1; }
    local port=$((VLLM_BASE_PORT + i))
    local pid_file
    pid_file="$(vllm_pid_file "$i")"
    mkdir -p "$VLLM_LOG_DIR"
    if ! stop_vllm_process_tree "$pid_file" "$port"; then
        log_error "VLLM #${i} (Port ${port}) 停止失败；仍有已捕获子孙进程存活"
        return 1
    fi
    log_info "VLLM #${i} (Port ${port}) 已停止"
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
        local gpu_memory_utilization
        local performance_mode
        local max_num_batched_tokens
        gpu_memory_utilization="$(vllm_gpu_memory_utilization_for "$i")"
        performance_mode="$(vllm_performance_mode_for "$i")"
        max_num_batched_tokens="$(vllm_max_num_batched_tokens_for "$i")"

        # 跳过已运行的实例(端口检测优先:实例隔离使 pid 文件分目录存放,
        # 仅靠 pid 文件会误判"未运行"而重复启动 vllm;端口已就绪即视为在运行)
        if curl -s "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            log_info "VLLM #${i} (NPU=${i}, Port=${port}) 已在运行"
            started=$((started + 1))
            continue
        fi
        if pid_file_matches "$pid_file" "vllm.*serve.*${VLLM_MODEL_PATH}" "$port"; then
            log_info "VLLM #${i} (NPU=${i}, Port=${port}) 进程存在但端口未就绪,等待中"
            started=$((started + 1))
            continue
        fi
        rm -f "$pid_file"

        log_info "启动 VLLM #${i}: NPU=${i}, Port=${port}"

        local triton_cache_dir="${TRITON_CACHE_ROOT}/npu${i}"
        mkdir -p "$triton_cache_dir"
        local vllm_tmp_dir="/tmp/vllm_tmp_npu${i}"
        mkdir -p "$vllm_tmp_dir"
        # vLLM AOT 编译缓存:per-NPU 子目录,避免 8 实例并发写同一 hash 目录
        local vllm_cache_dir="${VLLM_CACHE_ROOT}/npu${i}"
        mkdir -p "$vllm_cache_dir"

        nohup bash -lc "
            # 兜底:显式 source CANN 环境(防 bash -lc 新开登录 shell 时 profile 没配)
            [ -f '${ASCEND_SET_ENV}' ] && source '${ASCEND_SET_ENV}' >/dev/null 2>&1 || true
            [ -f '${ASCEND_ATB_SET_ENV}' ] && source '${ASCEND_ATB_SET_ENV}' >/dev/null 2>&1 || true
            # 兜底:vllm-ascend custom 算子库(让 LD_LIBRARY_PATH 找到 libcust_opapi.so)
            if [ -d '${VLLM_ASCEND_VENDOR_DIR}/op_api/lib' ]; then
                case \":\$LD_LIBRARY_PATH:\" in
                    *\":${VLLM_ASCEND_VENDOR_DIR}/op_api/lib:\"*) ;;
                    *) export LD_LIBRARY_PATH='${VLLM_ASCEND_VENDOR_DIR}/op_api/lib':\$LD_LIBRARY_PATH ;;
                esac
                case \":\$ASCEND_CUSTOM_OPP_PATH:\" in
                    *\":${VLLM_ASCEND_VENDOR_DIR}:\"*) ;;
                    *) export ASCEND_CUSTOM_OPP_PATH='${VLLM_ASCEND_VENDOR_DIR}':\$ASCEND_CUSTOM_OPP_PATH ;;
                esac
            fi
            export ASCEND_VISIBLE_DEVICES='${i}'
            export ASCEND_RT_VISIBLE_DEVICES='${i}'
            export DEVICE_ID=0
            export ASCEND_DEVICE_ID=0
            export TRITON_CACHE_DIR='${triton_cache_dir}'
            export VLLM_CACHE_ROOT='${vllm_cache_dir}'
            export TMPDIR='${vllm_tmp_dir}'
            exec ${VLLM_BIN} serve '${VLLM_MODEL_PATH}' \
                --host 0.0.0.0 \
                --tensor-parallel-size 1 \
                --port ${port} \
                --max-model-len ${VLLM_MAX_MODEL_LEN} \
                --gpu-memory-utilization ${gpu_memory_utilization} \
                --performance-mode ${performance_mode} \
                --max-num-batched-tokens ${max_num_batched_tokens} \
                --dtype float16 \
                --trust-remote-code
        " > "$log_file" 2>&1 &

        local pid=$!
        echo "$pid" > "$pid_file"
        sleep 2

        if pid_matches "$pid" "vllm.*serve.*${VLLM_MODEL_PATH}" "$port"; then
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
            terminate_pid_file "$pid_file" "vllm.*serve.*${VLLM_MODEL_PATH}" "$((VLLM_BASE_PORT + i))"
            log_info "VLLM #${i} (PID: $pid) 已停止"
        fi
    done

    log_info "VLLM 服务已全部停止"
}

status_vllm() {
    echo -e "${CYAN}VLLM 服务 (${VLLM_NUM_INSTANCES} 个实例)${NC}"
    local running=0
    for i in $(seq 0 $((VLLM_NUM_INSTANCES - 1))); do
        local port=$((VLLM_BASE_PORT + i))
        local pid_file="${VLLM_LOG_DIR}/vllm_npu${i}.pid"
        if pid_file_matches "$pid_file" "vllm.*serve.*${VLLM_MODEL_PATH}" "$port"; then
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
    if pid_file_matches "$pid_file" "python.*api_server.py"; then
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
    REDIS_CLAIM_MAINTENANCE_KEY="$REDIS_CLAIM_MAINTENANCE_KEY" \
    REDIS_CLAIM_PAUSE_KEY="$REDIS_CLAIM_PAUSE_KEY" \
    SQLITE_QUEUE_FALLBACK="$SQLITE_QUEUE_FALLBACK" \
    REDIS_TASK_TIMEOUT="$REDIS_TASK_TIMEOUT" \
    nohup ${PYTHON_BIN} api_server.py > "$log_file" 2>&1 &

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
        terminate_pid_file "$pid_file" "python.*api_server.py"
    fi
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
    if pid_file_matches "$pid_file" "python.*mcp_server.py"; then
        log_info "MCP Server 已在运行 (PID: $(cat "$pid_file"))"
        return 0
    fi

    cd "$BACKEND_DIR"

    DATABASE_PATH="$DATABASE_PATH" \
    OUTPUT_PATH="$OUTPUT_PATH" \
    API_BASE_URL="http://localhost:${API_PORT}" \
    MCP_PORT="$MCP_PORT" \
    MCP_HOST="0.0.0.0" \
    nohup ${PYTHON_BIN} mcp_server.py > "$log_file" 2>&1 &

    local pid=$!
    echo "$pid" > "$pid_file"
    log_info "MCP Server 启动中 (PID: $pid)..."

    sleep 3

    if pid_matches "$pid" "python.*mcp_server.py"; then
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
        terminate_pid_file "$pid_file" "python.*mcp_server.py"
    fi
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

worker_index_valid() {
    local index="$1"
    case "$index" in
        ''|*[!0-9]*) return 1 ;;
    esac
    [ "$index" -ge 0 ] && [ "$index" -lt "$WORKER_NUM_INSTANCES" ]
}

worker_pid_file() { echo "${WORKER_LOG_DIR}/worker_${1}.pid"; }
worker_drain_file() { echo "${WORKER_RUNTIME_DIR}/worker_${1}.drain"; }
worker_activity_dir() { echo "${WORKER_RUNTIME_DIR}/worker_${1}_activity"; }
worker_disabled_file() { echo "${WORKER_RUNTIME_DIR}/worker_${1}.disabled"; }
worker_expected_cmd() { echo "python.*litserve_worker.py"; }
compute_supervisor_expected_cmd() { echo "tianshu.sh[[:space:]]+supervise[[:space:]]+compute[[:space:]]+${1}"; }
compute_supervisor_pid_file() { echo "${WORKER_RUNTIME_DIR}/compute_${1}.supervisor.pid"; }
compute_supervisor_heartbeat_file() { echo "${WORKER_RUNTIME_DIR}/compute_${1}.supervisor.heartbeat"; }
worker_restart_request_file() { echo "${WORKER_RUNTIME_DIR}/worker_${1}.restart.request"; }
worker_restart_ack_file() { echo "${WORKER_RUNTIME_DIR}/worker_${1}.restart.ack"; }

compute_supervisor_is_running() {
    local i="$1"
    local pid_file pid
    pid_file="$(compute_supervisor_pid_file "$i")"
    [ -f "$pid_file" ] || return 1
    pid="$(cat "$pid_file" 2>/dev/null)"
    pid_matches "$pid" "$(compute_supervisor_expected_cmd "$i")"
}

write_compute_supervisor_state() {
    local i="$1"
    mkdir -p "$WORKER_RUNTIME_DIR"
    printf "%s\n" "$$" > "$(compute_supervisor_pid_file "$i")"
    date +%s > "$(compute_supervisor_heartbeat_file "$i")"
}

write_supervised_worker_restart_ack() {
    local i="$1"
    local revision="$2"
    local status="$3"
    local message="$4"
    local ack_file tmp_file
    ack_file="$(worker_restart_ack_file "$i")"
    tmp_file="${ack_file}.tmp.$$"
    {
        printf "%s\n" "$revision"
        printf "%s\n" "$status"
        printf "%s\n" "$(date +%s)"
        printf "%s\n" "$message"
    } > "$tmp_file" || return 1
    mv "$tmp_file" "$ack_file"
}

request_supervised_worker_restart() {
    local i="$1"
    local force="$2"
    local timeout="${SUPERVISED_WORKER_RESTART_TIMEOUT_SECONDS:-600}"
    local revision="$(date +%s)-$$"
    local request_file ack_file tmp_file elapsed ack_revision ack_status
    request_file="$(worker_restart_request_file "$i")"
    ack_file="$(worker_restart_ack_file "$i")"
    mkdir -p "$WORKER_RUNTIME_DIR"
    rm -f "$ack_file"
    tmp_file="${request_file}.tmp.$$"
    {
        printf "%s\n" "$revision"
        printf "%s\n" "$force"
        printf "%s\n" "$(date +%s)"
    } > "$tmp_file" || return 1
    mv "$tmp_file" "$request_file"
    log_info "Worker #${i} restart requested via compute supervisor (revision ${revision})"
    elapsed=0
    while [ "$elapsed" -lt "$timeout" ]; do
        if [ -f "$ack_file" ]; then
            ack_revision="$(sed -n "1p" "$ack_file" 2>/dev/null)"
            ack_status="$(sed -n "2p" "$ack_file" 2>/dev/null)"
            if [ "$ack_revision" = "$revision" ]; then
                if [ "$ack_status" = "ok" ]; then
                    log_info "Worker #${i} supervised restart acknowledged (revision ${revision})"
                    return 0
                fi
                log_error "Worker #${i} supervised restart failed (revision ${revision})"
                return 1
            fi
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    log_error "Worker #${i} supervised restart timed out after ${timeout}s; no direct fallback attempted"
    return 1
}

handle_supervised_worker_restart_request() {
    local i="$1"
    local request_file revision force ack_revision
    request_file="$(worker_restart_request_file "$i")"
    [ -f "$request_file" ] || return 0
    revision="$(sed -n "1p" "$request_file" 2>/dev/null)"
    force="$(sed -n "2p" "$request_file" 2>/dev/null)"
    [ -n "$revision" ] || return 0
    if [ -f "$(worker_restart_ack_file "$i")" ]; then
        ack_revision="$(sed -n "1p" "$(worker_restart_ack_file "$i")" 2>/dev/null)"
        [ "$ack_revision" = "$revision" ] && return 0
    fi
    log_info "Compute #${i} supervisor handling Worker restart request (revision ${revision})"
    drain_worker_instance "$i" "${DRAIN_WAIT_SECONDS:-300}"
    local drain_rc=$?
    if [ "$drain_rc" -ne 0 ] && [ "$force" != "--force" ] && [ "${DRAIN_FORCE_STOP:-false}" != "true" ]; then
        write_supervised_worker_restart_ack "$i" "$revision" "failed" "drain-not-complete" || true
        return 0
    fi
    stop_worker_instance "$i" || { write_supervised_worker_restart_ack "$i" "$revision" "failed" "stop-failed" || true; return 75; }
    start_worker_instance "$i" || { write_supervised_worker_restart_ack "$i" "$revision" "failed" "start-failed" || true; return 75; }
    worker_pid="$(cat "$(worker_pid_file "$i")" 2>/dev/null)"
    if ! wait_worker_instance_ready "$i" "$WORKER_READY_TIMEOUT"; then
        write_supervised_worker_restart_ack "$i" "$revision" "failed" "worker-not-ready" || true
        log_warn "Compute #${i} supervisor restarted Worker PID ${worker_pid}, but it did not become ready within ${WORKER_READY_TIMEOUT}s"
        return 75
    fi
    worker_health_failures=0
    write_supervised_worker_restart_ack "$i" "$revision" "ok" "worker-pid=${worker_pid}" || true
    log_info "Compute #${i} supervisor restarted ready Worker PID ${worker_pid} without replacing VLLM PID ${vllm_pid}"
}


worker_vllm_api_list() {
    local index="$1"
    if [ "${VLLM_ENDPOINT_STRATEGY:-local}" = "ring3" ]; then
        local out="["
        local sep=""
        for n in $(seq 0 $((VLLM_NUM_INSTANCES - 1))); do
            out="${out}${sep}\"http://localhost:$((VLLM_BASE_PORT + n))/v1\""
            sep=","
        done
        echo "${out}]"
    else
        echo "[\"http://localhost:$((VLLM_BASE_PORT + index))/v1\"]"
    fi
}

worker_is_running() {
    local index="$1"
    local port=$((WORKER_BASE_PORT + index))
    pid_file_matches "$(worker_pid_file "$index")" "$(worker_expected_cmd)" "$port"
}

pid_exists() {
    local pid="$1"
    case "$pid" in
        ''|*[!0-9]*) return 1 ;;
    esac
    [ -n "$(pid_state "$pid")" ]
}

worker_descendant_pids() {
    local root_pid="$1"
    ps -eo pid=,ppid= 2>/dev/null | awk -v root="$root_pid" '
        { children[$2] = children[$2] " " $1 }
        function walk(parent, parts, count, i, child) {
            count = split(children[parent], parts, " ")
            for (i = 1; i <= count; i++) {
                child = parts[i]
                if (child == "") {
                    continue
                }
                walk(child)
                print child
            }
        }
        END { walk(root) }
    '
}

worker_candidate_pids() {
    local proc_root="${TIANSHU_PROC_ROOT:-/proc}"
    local proc_dir

    if [ "$proc_root" != "/proc" ]; then
        for proc_dir in "$proc_root"/[0-9]*; do
            [ -d "$proc_dir" ] && echo "${proc_dir##*/}"
        done
        return 0
    fi

    pgrep -f '[l]itserve_worker.py' 2>/dev/null || true
}

worker_matching_pids_for_port() {
    local port="$1"
    local pid
    while read -r pid; do
        pid_matches "$pid" "$(worker_expected_cmd)" "$port" && echo "$pid"
    done < <(worker_candidate_pids)
}

stop_worker_process_tree() {
    local pid_file="$1"
    local port="$2"
    local timeout="${3:-5}"
    local parent_pid=""
    local descendant_pids=""
    local descendant_identities=""
    local identity
    local start_time
    local pid
    local survivors

    if [ -f "$pid_file" ]; then
        parent_pid="$(cat "$pid_file" 2>/dev/null)"
    fi

    if pid_matches "$parent_pid" "$(worker_expected_cmd)" "$port"; then
        descendant_pids="$(worker_descendant_pids "$parent_pid")"
        for pid in $descendant_pids; do
            start_time="$(pid_start_time "$pid")"
            [ -n "$start_time" ] || continue
            descendant_identities="$descendant_identities $pid:$start_time"
        done
        for identity in $descendant_identities; do
            pid="${identity%%:*}"
            start_time="${identity#*:}"
            pid_identity_matches "$pid" "$start_time" && kill "$pid" 2>/dev/null || true
        done
        kill "$parent_pid" 2>/dev/null || true

        for _ in $(seq 1 "$timeout"); do
            survivors=""
            pid_matches "$parent_pid" "$(worker_expected_cmd)" "$port" && survivors="$survivors $parent_pid"
            for identity in $descendant_identities; do
                pid="${identity%%:*}"
                start_time="${identity#*:}"
                pid_identity_matches "$pid" "$start_time" && survivors="$survivors $pid"
            done
            [ -z "$survivors" ] && break
            sleep 1
        done

        pid_matches "$parent_pid" "$(worker_expected_cmd)" "$port" && kill -9 "$parent_pid" 2>/dev/null || true
        for identity in $descendant_identities; do
            pid="${identity%%:*}"
            start_time="${identity#*:}"
            pid_identity_matches "$pid" "$start_time" && kill -9 "$pid" 2>/dev/null || true
        done

        survivors=""
        for identity in $descendant_identities; do
            pid="${identity%%:*}"
            start_time="${identity#*:}"
            pid_identity_matches "$pid" "$start_time" && survivors="$survivors $pid"
        done
        if [ -n "$survivors" ]; then
            log_error "Worker Port ${port} 停止后仍存在已捕获子孙进程: ${survivors}"
            rm -f "$pid_file"
            return 1
        fi
    fi

    rm -f "$pid_file"
    survivors="$(worker_matching_pids_for_port "$port")"
    if [ -n "$survivors" ]; then
        log_error "Worker Port ${port} 停止后仍存在匹配进程: ${survivors//$'\n'/ }"
        return 1
    fi
    return 0
}

worker_http_healthy() {
    local index="$1"
    local timeout="${2:-$SUPERVISOR_HEALTH_PROBE_TIMEOUT_SECONDS}"
    worker_index_valid "$index" || return 1
    local port=$((WORKER_BASE_PORT + index))
    curl -fsS --max-time "$timeout" "http://localhost:${port}/health" > /dev/null 2>&1
}

wait_worker_instance_ready() {
    local index="$1"
    local timeout="${2:-$WORKER_READY_TIMEOUT}"
    local elapsed=0
    local interval="${WORKER_READY_POLL_SECONDS:-5}"
    case "$timeout" in ""|*[!0-9]*) timeout=180 ;; esac
    case "$interval" in ""|*[!0-9]*|0) interval=5 ;; esac

    while [ "$elapsed" -le "$timeout" ]; do
        worker_is_running "$index" || return 1
        if worker_http_healthy "$index" "$SUPERVISOR_HEALTH_PROBE_TIMEOUT_SECONDS"; then
            return 0
        fi
        sleep "$interval"
        elapsed=$((elapsed + interval))
    done
    return 1
}

check_worker_dependencies() {
    local index="$1"
    local vllm_port=$((VLLM_BASE_PORT + index))

    if ! vllm_http_ready "$index"; then
        log_error "VLLM #${index} 未就绪 (端口 ${vllm_port})"
        return 1
    fi

    if ! curl -s "http://localhost:${API_PORT}/docs" > /dev/null 2>&1; then
        log_error "API Server 未运行，请先启动 API Server"
        return 1
    fi

    return 0
}

start_worker_instance() {
    local i="$1"
    worker_index_valid "$i" || { log_error "无效 Worker 编号: $i"; return 1; }
    check_worker_dependencies "$i" || return 1

    local port=$((WORKER_BASE_PORT + i))
    local vllm_port=$((VLLM_BASE_PORT + i))
    local vllm_api="http://localhost:${vllm_port}/v1"
    local vllm_api_list
    local worker_max_tasks
    local worker_hybrid_ratio=""
    local worker_hybrid_display="auto"
    local worker_config_revision
    local -a worker_hybrid_env
    vllm_api_list="$(worker_vllm_api_list "$i")"
    worker_max_tasks="$(worker_max_concurrent_tasks_for "$i")"
    if worker_hybrid_ratio="$(worker_hybrid_batch_ratio_for "$i")"; then
        worker_hybrid_display="$worker_hybrid_ratio"
        worker_hybrid_env=(env "MINERU_HYBRID_BATCH_RATIO=$worker_hybrid_ratio")
    else
        worker_hybrid_env=(env -u MINERU_HYBRID_BATCH_RATIO)
    fi
    worker_config_revision="$(worker_config_revision_for "$i")"
    local log_file="${WORKER_LOG_DIR}/worker_${i}_port${port}.log"
    local pid_file
    pid_file="$(worker_pid_file "$i")"

    if worker_is_running "$i"; then
        log_info "Worker #${i} (Port ${port}) 已在运行 (PID: $(cat "$pid_file"))"
        return 0
    fi
    mkdir -p "$WORKER_RUNTIME_DIR" "$(worker_activity_dir "$i")"
    rm -f "$(worker_drain_file "$i")" "$(worker_disabled_file "$i")"
    find "$(worker_activity_dir "$i")" -mindepth 1 -type f -delete 2>/dev/null || true
    rm -f "$pid_file"

    log_info "启动 Worker #${i}: Port=${port}, VLLM=${vllm_api}, hybrid_batch_ratio=${worker_hybrid_display}, max_concurrent_tasks=${worker_max_tasks}, config_revision=${worker_config_revision}"
    cd "$BACKEND_DIR"

    DATABASE_PATH="$DATABASE_PATH" \
    OUTPUT_PATH="$OUTPUT_PATH" \
    MODELSCOPE_CACHE="$MODELSCOPE_CACHE" \
    WORKER_PORT="$port" \
    WORKER_GROUP_INDEX="$i" \
    WORKER_DRAIN_FILE="$(worker_drain_file "$i")" \
    WORKER_ACTIVITY_DIR="$(worker_activity_dir "$i")" \
    VLLM_ENDPOINT_STRATEGY="${VLLM_ENDPOINT_STRATEGY:-local}" \
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
    REDIS_CLAIM_MAINTENANCE_KEY="$REDIS_CLAIM_MAINTENANCE_KEY" \
    REDIS_CLAIM_PAUSE_KEY="$REDIS_CLAIM_PAUSE_KEY" \
    SQLITE_QUEUE_FALLBACK="$SQLITE_QUEUE_FALLBACK" \
    REDIS_TASK_TIMEOUT="$REDIS_TASK_TIMEOUT" \
    PDF_SPLIT_ENABLED="$PDF_SPLIT_ENABLED" \
    PDF_SPLIT_THRESHOLD_PAGES="$PDF_SPLIT_THRESHOLD_PAGES" \
    PDF_SPLIT_CHUNK_SIZE="$PDF_SPLIT_CHUNK_SIZE" \
    MINERU_INTRA_OP_NUM_THREADS="$MINERU_INTRA_OP_NUM_THREADS" \
    MINERU_INTER_OP_NUM_THREADS="$MINERU_INTER_OP_NUM_THREADS" \
    RUSTFS_ENABLED="$RUSTFS_ENABLED" \
    VLLM_DOCKER_CONTROLLER_ENABLED="$VLLM_DOCKER_CONTROLLER_ENABLED" \
    MAX_CONCURRENT_TASKS="$worker_max_tasks" \
    "${worker_hybrid_env[@]}" nohup ${PYTHON_BIN} litserve_worker.py \
        --accelerator "$WORKER_ACCELERATOR" \
        --port "$port" \
        --workers-per-device 1 \
        --devices "$i" \
        --mineru-vllm-api-list "$vllm_api_list" \
        > "$log_file" 2>&1 &

    local pid=$!
    echo "$pid" > "$pid_file"
    sleep 3

    if worker_is_running "$i"; then
        log_info "Worker #${i} 已启动 (PID: $pid)"
        return 0
    fi

    log_error "Worker #${i} 启动失败，查看: $log_file"
    rm -f "$pid_file"
    return 1
}

start_workers() {
    local target_index="${1:-}"
    local started=0
    local failed=0

    if [ -n "$target_index" ]; then
        log_step "启动 Worker #${target_index}"
        start_worker_instance "$target_index"
        return $?
    fi

    log_step "启动 Workers (${WORKER_NUM_INSTANCES} 个独立进程)"
    for i in $(seq 0 $((WORKER_NUM_INSTANCES - 1))); do
        if start_worker_instance "$i"; then
            started=$((started + 1))
        else
            failed=$((failed + 1))
        fi
    done

    log_info "Worker 启动完成: ${started}/${WORKER_NUM_INSTANCES}"
    [ "$failed" -eq 0 ]
}

stop_worker_instance() {
    local i="$1"
    worker_index_valid "$i" || { log_error "无效 Worker 编号: $i"; return 1; }
    local port=$((WORKER_BASE_PORT + i))
    local pid_file
    pid_file="$(worker_pid_file "$i")"
    mkdir -p "$WORKER_LOG_DIR" "$WORKER_RUNTIME_DIR"
    echo "$(date +%s)" > "$(worker_disabled_file "$i")"
    if ! stop_worker_process_tree "$pid_file" "$port"; then
        log_error "Worker #${i} (Port ${port}) 停止失败；仍有匹配进程存活"
        return 1
    fi
    log_info "Worker #${i} (Port ${port}) 已停止并标记为 disabled；start worker ${i} 会清除标记"
}

stop_workers() {
    local target_index="${1:-}"
    local failed=0

    if [ -n "$target_index" ]; then
        log_step "停止 Worker #${target_index}"
        stop_worker_instance "$target_index"
        return $?
    fi

    log_step "停止 Workers"
    for i in $(seq 0 $((WORKER_NUM_INSTANCES - 1))); do
        stop_worker_instance "$i" || failed=$((failed + 1))
    done
    if [ "$failed" -ne 0 ]; then
        log_error "Workers 停止完成，但 ${failed} 个实例仍有存活进程"
        return 1
    fi
    log_info "Workers 已全部停止"
}

worker_drain_complete() {
    local i="$1"
    local activity_dir
    activity_dir="$(worker_activity_dir "$i")"
    [ -d "$activity_dir" ] || return 0
    ! find "$activity_dir" -mindepth 1 -type f -print -quit 2>/dev/null | grep -q .
}

wait_worker_drain() {
    local i="$1"
    local timeout="${2:-${DRAIN_WAIT_SECONDS:-300}}"
    local elapsed=0
    local grace="${DRAIN_GRACE_SECONDS:-3}"
    local stable_interval="${DRAIN_STABLE_INTERVAL_SECONDS:-2}"

    sleep "$grace"
    elapsed=$((elapsed + grace))
    while [ "$elapsed" -le "$timeout" ]; do
        if worker_drain_complete "$i"; then
            sleep "$stable_interval"
            elapsed=$((elapsed + stable_interval))
            if worker_drain_complete "$i"; then
                return 0
            fi
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    return 2
}

drain_worker_instance() {
    local i="$1"
    local wait_seconds="${2:-${DRAIN_WAIT_SECONDS:-300}}"
    worker_index_valid "$i" || { log_error "无效 Worker 编号: $i"; return 1; }
    mkdir -p "$WORKER_RUNTIME_DIR" "$(worker_activity_dir "$i")"
    echo "$(date +%s)" > "$(worker_drain_file "$i")"
    echo "$(date +%s)" > "$(worker_disabled_file "$i")"
    log_info "Worker #${i} 已标记 drain/disabled；watchdog 将跳过该实例，start worker ${i} 会清除标记"
    if wait_worker_drain "$i" "$wait_seconds"; then
        log_info "Worker #${i} drain 完成（进程已退出或不存在）"
        return 0
    fi
    log_warn "Worker #${i} drain 未完成或无法验证 (${wait_seconds}s)；默认不会停止该实例"
    return 2
}

restart_worker_instance() {
    local i="$1"
    local force="${2:-}"
    worker_index_valid "$i" || { log_error "无效 Worker 编号: $i"; return 1; }
    if [ "${SUPERVISOR_RESTART_WORKER_IN_PLACE:-false}" != "true" ] && compute_supervisor_is_running "$i"; then
        request_supervised_worker_restart "$i" "$force"
        return $?
    fi
    drain_worker_instance "$i" "${DRAIN_WAIT_SECONDS:-300}"
    local drain_rc=$?
    if [ "$drain_rc" -ne 0 ]; then
        if [ "$force" != "--force" ] && [ "${DRAIN_FORCE_STOP:-false}" != "true" ]; then
            log_warn "Worker #${i} restart pending: drain 未验证完成；如确认可中断，使用 restart worker ${i} --force"
            return 2
        fi
        log_warn "Worker #${i} 使用 force 重启；可能中断 in-flight task"
    fi
    stop_worker_instance "$i" || return 1
    start_worker_instance "$i"
}


restart_compute_instance() {
    local i="$1"
    local force="${2:-}"

    worker_index_valid "$i" || { log_error "无效 Worker 编号: $i"; return 1; }
    vllm_index_valid "$i" || { log_error "无效 VLLM 编号: $i"; return 1; }
    if [ "${VLLM_ENDPOINT_STRATEGY:-local}" = "ring3" ]; then
        log_error "restart compute ${i} 不支持 VLLM_ENDPOINT_STRATEGY=ring3；跨端点 worker 可能仍在使用该 vLLM"
        return 1
    fi

    init_dirs
    source_ascend_env
    check_instance_conflict || return 1

    drain_worker_instance "$i" "${DRAIN_WAIT_SECONDS:-300}"
    local drain_rc=$?
    if [ "$drain_rc" -ne 0 ]; then
        if [ "$force" != "--force" ] && [ "${DRAIN_FORCE_STOP:-false}" != "true" ]; then
            log_warn "Compute #${i} restart pending: drain 未验证完成；如确认可中断，使用 restart compute ${i} --force"
            return 2
        fi
        log_warn "Compute #${i} 使用 force 重启；可能中断 in-flight task"
    fi

    stop_worker_instance "$i" || return 1
    stop_vllm_instance "$i" || return 1
    start_vllm_instance "$i" || return 1
    start_worker_instance "$i"
}

status_workers() {
    echo -e "${CYAN}Workers (${WORKER_NUM_INSTANCES} 个独立进程)${NC}"
    local running=0
    for i in $(seq 0 $((WORKER_NUM_INSTANCES - 1))); do
        local port=$((WORKER_BASE_PORT + i))
        if [ -f "$(worker_drain_file "$i")" ]; then
            echo "  ⏸️  Worker #${i} (Port ${port}) - Drain 中"
            worker_is_running "$i" && running=$((running + 1))
        elif [ -f "$(worker_disabled_file "$i")" ]; then
            echo "  ⏹️  Worker #${i} (Port ${port}) - Disabled"
        elif worker_is_running "$i"; then
            echo "  ✅ Worker #${i} (Port ${port}) - 运行中"
            running=$((running + 1))
        else
            echo "  ❌ Worker #${i} (Port ${port}) - 已停止"
        fi
    done
    echo "  运行中: ${running}/${WORKER_NUM_INSTANCES}"
}

# ============================================================================
# Watchdog（Worker 单实例补齐 + API 端口探活自动恢复）
# ============================================================================

start_watchdog() {
    log_step "启动 Watchdog (Worker + API)"

    local pid_file="${WORKER_LOG_DIR}/watchdog.pid"
    local log_file="${WORKER_LOG_DIR}/watchdog.log"

    if pid_file_matches "$pid_file" "tianshu-watchdog"; then
        log_info "Watchdog 已在运行 (PID: $(cat "$pid_file"))"
        return 0
    fi
    rm -f "$pid_file"

    # 后台子 shell,每 60s 巡检:
    #   1) Worker 单实例:只补齐缺失/僵尸/错误 cmdline 的具体实例
    #   2) API 端口探活:失败时通过 tianshu.sh stop/start 走受控 PID 校验
    nohup bash -c '
        TIANSHU_SCRIPT="'"${PROJECT_ROOT}"'/scripts/tianshu.sh"
        WORKER_LOG_DIR="'"${WORKER_LOG_DIR}"'"
        WORKER_RUNTIME_DIR="'"${WORKER_RUNTIME_DIR}"'"
        WORKER_BASE_PORT='"${WORKER_BASE_PORT}"'
        EXPECTED='"${WORKER_NUM_INSTANCES}"'
        API_PORT='"${API_PORT}"'
        INTERVAL=60
        API_RESTART_COOLDOWN=300
        API_LAST_RESTART=0
        proc_state() { awk '\''/^State:/ {print $2}'\'' "/proc/$1/status" 2>/dev/null; }
        proc_cmd() { tr '\''\0'\'' '\'' '\'' < "/proc/$1/cmdline" 2>/dev/null; }
        worker_ok() {
            local idx="$1" port pid_file pid state cmd
            port=$((WORKER_BASE_PORT + idx))
            if [ -f "$WORKER_RUNTIME_DIR/worker_${idx}.disabled" ] || [ -f "$WORKER_RUNTIME_DIR/worker_${idx}.drain" ]; then
                return 0
            fi
            pid_file="$WORKER_LOG_DIR/worker_${idx}.pid"
            [ -f "$pid_file" ] || return 1
            pid=$(cat "$pid_file" 2>/dev/null)
            case "$pid" in ""|*[!0-9]*) return 1 ;; esac
            state=$(proc_state "$pid")
            [ "$state" = Z ] && return 1
            cmd=$(proc_cmd "$pid")
            printf "%s\n" "$cmd" | grep -Eq "python.*litserve_worker.py" || return 1
            printf "%s\n" "$cmd" | grep -Eq "(^|[[:space:]])--port[=[:space:]]${port}($|[[:space:]])" || return 1
        }
        while true; do
            sleep "$INTERVAL"
            NOW=$(date +%s)

            idx=0
            while [ "$idx" -lt "$EXPECTED" ]; do
                if ! worker_ok "$idx"; then
                    echo "[$(date "+%F %T")] watchdog: worker #$idx missing/stale, starting only this instance" >> "'"${log_file}"'"
                    bash "$TIANSHU_SCRIPT" start worker "$idx" >> "'"${log_file}"'" 2>&1
                fi
                idx=$((idx + 1))
            done

            if ! curl -fsS --max-time 10 "http://localhost:${API_PORT}/docs" > /dev/null 2>&1; then
                ELAPSED=$((NOW - API_LAST_RESTART))
                if [ "$ELAPSED" -lt "$API_RESTART_COOLDOWN" ]; then
                    echo "[$(date "+%F %T")] watchdog: API restart cooldown (${ELAPSED}s/${API_RESTART_COOLDOWN}s), skip" >> "'"${log_file}"'"
                else
                    echo "[$(date "+%F %T")] watchdog: restarting API server via controlled lifecycle" >> "'"${log_file}"'"
                    bash "$TIANSHU_SCRIPT" stop api >> "'"${log_file}"'" 2>&1
                    bash "$TIANSHU_SCRIPT" start api >> "'"${log_file}"'" 2>&1
                    API_LAST_RESTART=$NOW
                fi
            fi
        done
    ' tianshu-watchdog > /dev/null 2>&1 &

    local pid=$!
    echo "$pid" > "$pid_file"
    log_info "Watchdog 已启动 (PID: $pid, 每 60s 巡检 Worker 单实例 + API 端口探活)"
}

stop_watchdog() {
    local pid_file="${WORKER_LOG_DIR}/watchdog.pid"
    terminate_pid_file "$pid_file" "tianshu-watchdog"
    log_info "Watchdog 已停止"
}

status_watchdog() {
    echo -e "${CYAN}Watchdog (Worker + API)${NC}"
    local pid_file="${WORKER_LOG_DIR}/watchdog.pid"
    if pid_file_matches "$pid_file" "tianshu-watchdog"; then
        echo "  ✅ 运行中 (PID: $(cat "$pid_file")), 每 60s 巡检 Worker 单实例 + API"
    else
        echo "  ❌ 未运行"
    fi
}

# ============================================================================
# 调度器管理（孤儿任务恢复 + 队列监控 + 旧文件清理）
# ============================================================================

start_scheduler() {
    log_step "启动 Task Scheduler"

    mkdir -p "$SCHEDULER_LOG_DIR"

    local pid_file="${SCHEDULER_LOG_DIR}/scheduler.pid"
    local log_file="${SCHEDULER_LOG_DIR}/scheduler.log"

    if pid_file_matches "$pid_file" "python.*task_scheduler.py"; then
        log_info "Task Scheduler 已在运行 (PID: $(cat "$pid_file"))"
        return 0
    fi

    # 预检查: worker 至少有一个通过 PID/cmdline/port 校验
    local running_workers=0
    for i in $(seq 0 $((WORKER_NUM_INSTANCES - 1))); do
        worker_is_running "$i" && running_workers=$((running_workers + 1))
    done
    if [ "$running_workers" -eq 0 ]; then
        log_error "无运行中的 Worker，请先启动 Worker (scheduler 依赖 worker 健康检查)"
        return 1
    fi

    cd "$BACKEND_DIR"

    local scheduler_args=(
        task_scheduler.py
        --litserve-url "http://localhost:${WORKER_BASE_PORT}/predict"
        --monitor-interval "$SCHEDULER_MONITOR_INTERVAL"
        --health-check-interval "$SCHEDULER_HEALTH_INTERVAL"
        --stale-task-timeout "$SCHEDULER_STALE_TIMEOUT"
        --orphan-recovery-batch-size "$SCHEDULER_ORPHAN_RECOVERY_BATCH_SIZE"
        --cleanup-old-files-days "$SCHEDULER_CLEANUP_DAYS"
        --wait-for-workers
    )
    case "${SCHEDULER_ORPHAN_RECOVERY_APPLY,,}" in
        1|true|yes|on) scheduler_args+=(--orphan-recovery-apply) ;;
        0|false|no|off) ;;
        *)
            log_error "SCHEDULER_ORPHAN_RECOVERY_APPLY 必须是 true/false"
            return 1
            ;;
    esac

    DATABASE_PATH="$DATABASE_PATH" \
    OUTPUT_PATH="$OUTPUT_PATH" \
    REDIS_QUEUE_ENABLED="$REDIS_QUEUE_ENABLED" \
    REDIS_HOST="$REDIS_HOST" \
    REDIS_PORT="$REDIS_PORT" \
    REDIS_DB="$REDIS_DB" \
    REDIS_PASSWORD="$REDIS_PASSWORD" \
    REDIS_QUEUE_KEY="$REDIS_QUEUE_KEY" \
    REDIS_PROCESSING_KEY="$REDIS_PROCESSING_KEY" \
    REDIS_CLAIM_MAINTENANCE_KEY="$REDIS_CLAIM_MAINTENANCE_KEY" \
    REDIS_CLAIM_PAUSE_KEY="$REDIS_CLAIM_PAUSE_KEY" \
    SQLITE_QUEUE_FALLBACK="$SQLITE_QUEUE_FALLBACK" \
    REDIS_TASK_TIMEOUT="$REDIS_TASK_TIMEOUT" \
    nohup "$PYTHON_BIN" "${scheduler_args[@]}" > "$log_file" 2>&1 &

    local pid=$!
    echo "$pid" > "$pid_file"
    log_info "Task Scheduler 启动中 (PID: $pid)..."

    sleep 3

    if pid_matches "$pid" "python.*task_scheduler.py"; then
        log_info "Task Scheduler 就绪 - 孤儿恢复 apply=${SCHEDULER_ORPHAN_RECOVERY_APPLY}, 阈值 ${SCHEDULER_STALE_TIMEOUT}m, 监控周期 ${SCHEDULER_MONITOR_INTERVAL}s"
    else
        log_error "Task Scheduler 启动失败，查看: $log_file"
        return 1
    fi
}

stop_scheduler() {
    log_step "停止 Task Scheduler"

    local pid_file="${SCHEDULER_LOG_DIR}/scheduler.pid"
    if [ -f "$pid_file" ]; then
        terminate_pid_file "$pid_file" "python.*task_scheduler.py"
    fi
    log_info "Task Scheduler 已停止"
}

status_scheduler() {
    echo -e "${CYAN}Task Scheduler (孤儿恢复阈值 ${SCHEDULER_STALE_TIMEOUT}m)${NC}"
    local pid_file="${SCHEDULER_LOG_DIR}/scheduler.pid"
    if pid_file_matches "$pid_file" "python.*task_scheduler.py"; then
        echo "  ✅ 运行中 (PID: $(cat "$pid_file"))"
        # 显示最近一次 orphan recovery 日志
        if [ -f "${SCHEDULER_LOG_DIR}/scheduler.log" ]; then
            local last=$(grep "recover_orphans\|Orphan recovery" "${SCHEDULER_LOG_DIR}/scheduler.log" 2>/dev/null | tail -1)
            [ -n "$last" ] && echo "  最近恢复: $last"
        fi
    else
        echo "  ❌ 未运行"
    fi
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
    local redis_bin="${REDIS_BIN:-/data/miniconda3/envs/mineru/bin/redis-server}"
    local redis_cli_bin="${REDIS_CLI_BIN:-/data/miniconda3/envs/mineru/bin/redis-cli}"
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
    local redis_cli_bin="${REDIS_CLI_BIN:-/data/miniconda3/envs/mineru/bin/redis-cli}"

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
    local redis_cli_bin="${REDIS_CLI_BIN:-/data/miniconda3/envs/mineru/bin/redis-cli}"
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

run_parent_merge_reconciler() {
    local apply="${PARENT_MERGE_RECONCILE_APPLY:-false}"
    local reconciler="${SCRIPT_DIR}/reconcile_parent_merges.py"
    local args=(--report-json)
    [ -f "$reconciler" ] || return 0
    if [ "$apply" = "true" ]; then
        args+=(--apply)
    fi
    "$PYTHON_BIN" "$reconciler" "${args[@]}" || \
        log_warn "parent merge reconciler failed"
}

cmd_configure() {
    local target="${1:-}"
    local instance_index="${2:-}"
    shift 2 || true
    case "$target" in
        worker)
            [ -n "$instance_index" ] || { log_error "configure worker 需要指定实例编号"; return 1; }
            configure_worker_instance "$instance_index" "$@"
            ;;
        *) log_error "未知 configure 目标: $target (可选: worker)"; return 1 ;;
    esac
}

cmd_start() {
    local target="${1:-all}"
    local instance_index="${2:-}"
    local rc=0

    separator
    echo -e "${CYAN}  MinerU Tianshu - 启动服务${NC}"
    echo -e "  项目路径: ${PROJECT_ROOT}"
    echo -e "  目标: ${target}"
    separator

    init_dirs

    # 加载 CANN / Ascend 环境(LD_LIBRARY_PATH 等),供所有后续 nohup 子进程继承
    source_ascend_env

    # 多实例冲突检测:确保本数据目录未被别的活跃实例占用
    check_instance_conflict || return 1

    case "$target" in
        redis)    start_redis    || rc=1 ;;
        vllm)     start_vllm     || rc=1 ;;
        api)      start_api      || rc=1 ;;
        mcp)      start_mcp      || rc=1 ;;
        worker)   start_workers "$instance_index" || rc=1 ;;
        watchdog) start_watchdog || rc=1 ;;
        scheduler) start_scheduler || rc=1 ;;
        frontend) start_frontend || rc=1 ;;
        all)
            start_redis    || rc=1
            start_vllm     || rc=1
            start_api      || rc=1
            start_mcp      || rc=1
            start_workers  || rc=1
            start_watchdog || rc=1
            start_scheduler || rc=1
            start_frontend || rc=1
            ;;
        *) log_error "未知服务: $target (可选: vllm|api|redis|mcp|worker|watchdog|scheduler|frontend|all)"; exit 1 ;;
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
    local instance_index="${2:-}"

    case "$target" in
        vllm)     stop_vllm ;;
        api)      stop_api ;;
        mcp)      stop_mcp ;;
        worker)   stop_workers "$instance_index" ;;
        watchdog) stop_watchdog ;;
        scheduler) stop_scheduler ;;
        frontend) stop_frontend ;;
        redis)    stop_redis ;;
        all)
            stop_frontend
            stop_scheduler
            stop_watchdog
            stop_workers
            stop_mcp
            stop_api
            stop_vllm
            stop_redis
            ;;
        *) log_error "未知服务: $target (可选: vllm|api|redis|mcp|worker|watchdog|scheduler|frontend|all)"; exit 1 ;;
    esac

    # 仅在停止整个实例(all)时释放数据目录锁;单服务停止不释放(api 仍占用目录)
    if [ "$target" = "all" ]; then
        release_instance_lock
    fi
}

cmd_restart() {
    local target="${1:-all}"
    local instance_index="${2:-}"
    local force_flag="${3:-}"
    if [ -n "$instance_index" ]; then
        case "$target" in
            worker)
                log_step "重启 Worker #${instance_index}"
                restart_worker_instance "$instance_index" "$force_flag"
                return $?
                ;;
            compute)
                log_step "重启 Compute #${instance_index} (Worker + VLLM)"
                restart_compute_instance "$instance_index" "$force_flag"
                return $?
                ;;
        esac
    fi
    cmd_stop "$target" "$instance_index"
    sleep 3
    cmd_start "$target" "$instance_index"
}

cmd_drain() {
    local target="${1:-worker}"
    local instance_index="${2:-}"
    case "$target" in
        worker)
            [ -n "$instance_index" ] || { log_error "drain worker 需要指定实例编号"; exit 1; }
            drain_worker_instance "$instance_index"
            ;;
        *) log_error "未知 drain 目标: $target (可选: worker)"; exit 1 ;;
    esac
}

supervise_child() {
    local name="$1"
    local start_fn="$2"
    local pid_file="$3"
    local expected="$4"
    local backoff=1
    local max_backoff="${SUPERVISOR_MAX_BACKOFF:-60}"

    while true; do
        "$start_fn" || true
        if pid_file_matches "$pid_file" "$expected"; then
            local pid
            pid="$(cat "$pid_file")"
            log_info "Supervisor watching ${name} (PID: ${pid})"
            if ! wait "$pid" 2>/dev/null; then
                while pid_file_matches "$pid_file" "$expected"; do
                    sleep 5
                done
            fi
            rm -f "$pid_file"
            log_warn "Supervisor detected ${name} exit; restarting after ${backoff}s"
        else
            log_warn "Supervisor could not start ${name}; retrying after ${backoff}s"
        fi
        sleep "$backoff"
        if [ "$backoff" -lt "$max_backoff" ]; then
            backoff=$((backoff * 2))
            [ "$backoff" -gt "$max_backoff" ] && backoff="$max_backoff"
        fi
    done
}

supervise_node() {
    separator
    echo -e "${CYAN}  MinerU Tianshu - 监督完整节点${NC}"
    separator
    init_dirs
    source_ascend_env
    check_instance_conflict || return 1

    start_redis || return 1
    start_api || return 1
    start_mcp || log_warn "Node supervisor: MCP 启动失败，将继续监督核心计算链路"
    start_frontend || log_warn "Node supervisor: Frontend 启动失败，将继续监督核心计算链路"

    local supervisor_pids=()
    local cleanup_started=0
    cleanup_node_supervisor() {
        [ "$cleanup_started" -eq 0 ] || return 0
        cleanup_started=1
        if [ "${#supervisor_pids[@]}" -gt 0 ]; then
            kill "${supervisor_pids[@]}" 2>/dev/null || true
            wait "${supervisor_pids[@]}" 2>/dev/null || true
        fi
    }
    trap 'cleanup_node_supervisor; exit 0' INT TERM
    trap cleanup_node_supervisor EXIT

    cmd_supervise control &
    supervisor_pids+=("$!")
    local i=0
    while [ "$i" -lt "$VLLM_NUM_INSTANCES" ]; do
        supervise_compute_instance "$i" &
        supervisor_pids+=("$!")
        i=$((i + 1))
    done
    log_info "Node supervisor watching control + ${VLLM_NUM_INSTANCES} compute supervisors"

    wait -n "${supervisor_pids[@]}"
    local rc=$?
    [ "$rc" -ne 0 ] || rc=1
    log_error "Node supervisor detected a child supervisor exit (rc=${rc}); terminating owned supervisors"
    cleanup_node_supervisor
    return "$rc"
}

supervise_compute_instance() {
    local i="$1"
    worker_index_valid "$i" || { log_error "无效 Compute 编号: $i"; return 1; }
    vllm_index_valid "$i" || { log_error "无效 Compute 编号: $i"; return 1; }

    separator
    echo -e "${CYAN}  MinerU Tianshu - 监督 Compute #${i}${NC}"
    separator
    init_dirs
    source_ascend_env
    check_instance_conflict || return 1

    if worker_is_running "$i" || vllm_is_running "$i"; then
        if [ "${SUPERVISE_COMPUTE_REPLACE:-false}" != "true" ]; then
            log_error "Compute #${i} 已有运行进程；先安全停止，或显式设置 SUPERVISE_COMPUTE_REPLACE=true"
            return 2
        fi
        if [ "${VLLM_ENDPOINT_STRATEGY:-local}" = "ring3" ]; then
            log_error "supervise compute ${i} 不支持替换 ring3 worker"
            return 1
        fi
        if worker_is_running "$i"; then
            drain_worker_instance "$i" "${DRAIN_WAIT_SECONDS:-300}" || return 2
            stop_worker_instance "$i" || return 1
        fi
        if vllm_is_running "$i"; then
            stop_vllm_instance "$i" || return 1
        fi
    fi

    local vllm_pid=""
    local worker_pid=""
    local cleanup_started=0
    local backoff="${COMPUTE_SUPERVISOR_BACKOFF_SECONDS:-1}"
    local max_backoff="${SUPERVISOR_MAX_BACKOFF:-60}"
    local poll_seconds="$COMPUTE_SUPERVISOR_POLL_SECONDS"
    local health_interval="$COMPUTE_SUPERVISOR_HEALTH_INTERVAL_SECONDS"
    local health_failure_threshold="$COMPUTE_SUPERVISOR_HEALTH_FAILURE_THRESHOLD"
    local heartbeat_seconds="$COMPUTE_SUPERVISOR_HEARTBEAT_SECONDS"
    local health_probe_timeout="$SUPERVISOR_HEALTH_PROBE_TIMEOUT_SECONDS"
    case "$poll_seconds" in ""|*[!0-9]*|0) poll_seconds=5 ;; esac
    case "$health_interval" in ""|*[!0-9]*|0) health_interval=15 ;; esac
    case "$health_failure_threshold" in ""|*[!0-9]*|0) health_failure_threshold=3 ;; esac
    case "$heartbeat_seconds" in ""|*[!0-9]*|0) heartbeat_seconds=300 ;; esac
    case "$health_probe_timeout" in ""|*[!0-9]*|0) health_probe_timeout=5 ;; esac

    stop_owned_compute() {
        stop_worker_instance "$i" || true
        stop_vllm_instance "$i" || true
        [ -z "$worker_pid" ] || wait "$worker_pid" 2>/dev/null || true
        [ -z "$vllm_pid" ] || wait "$vllm_pid" 2>/dev/null || true
        worker_pid=""
        vllm_pid=""
    }
    cleanup_compute_supervisor() {
        [ "$cleanup_started" -eq 0 ] || return 0
        cleanup_started=1
        rm -f "$(compute_supervisor_pid_file "$i")" "$(compute_supervisor_heartbeat_file "$i")"
        log_info "Compute #${i} supervisor 正在停止子进程"
        stop_owned_compute
    }
    wait_compute_backoff() {
        sleep "$backoff"
        if [ "$backoff" -lt "$max_backoff" ]; then
            backoff=$((backoff * 2))
            [ "$backoff" -gt "$max_backoff" ] && backoff="$max_backoff"
        fi
    }
    trap 'cleanup_compute_supervisor; exit 0' INT TERM
    trap cleanup_compute_supervisor EXIT
    write_compute_supervisor_state "$i"

    while true; do
        if ! start_vllm_instance "$i"; then
            log_warn "Compute #${i} supervisor could not start VLLM; retrying after ${backoff}s"
            stop_owned_compute
            wait_compute_backoff
            continue
        fi
        vllm_pid="$(cat "$(vllm_pid_file "$i")" 2>/dev/null)"

        if ! start_worker_instance "$i"; then
            log_warn "Compute #${i} supervisor could not start Worker; retrying pair after ${backoff}s"
            stop_owned_compute
            wait_compute_backoff
            continue
        fi
        worker_pid="$(cat "$(worker_pid_file "$i")" 2>/dev/null)"
        log_info "Compute #${i} supervisor watching VLLM PID ${vllm_pid}, Worker PID ${worker_pid}"
        write_compute_supervisor_state "$i"
        if ! wait_worker_instance_ready "$i" "$WORKER_READY_TIMEOUT"; then
            log_warn "Compute #${i} Worker did not become ready within ${WORKER_READY_TIMEOUT}s; restarting pair after ${backoff}s"
            stop_owned_compute
            wait_compute_backoff
            continue
        fi

        local failed_component=""
        local failure_reason=""
        local vllm_health_failures=0
        local worker_health_failures=0
        local now
        local last_health_check=0
        local last_heartbeat
        last_heartbeat="$(date +%s)"
        while true; do
            handle_supervised_worker_restart_request "$i"
            local restart_request_rc=$?
            if [ "$restart_request_rc" -ne 0 ]; then
                failed_component="Worker"
                failure_reason="supervised restart did not reach ready"
                break
            fi
            write_compute_supervisor_state "$i"

            if ! pid_matches "$vllm_pid" "$(vllm_expected_cmd)" "$((VLLM_BASE_PORT + i))"; then
                failed_component="VLLM"
                failure_reason="process exited"
                break
            fi
            if ! pid_matches "$worker_pid" "$(worker_expected_cmd)" "$((WORKER_BASE_PORT + i))"; then
                failed_component="Worker"
                failure_reason="process exited"
                break
            fi

            now="$(date +%s)"
            if [ $((now - last_health_check)) -ge "$health_interval" ]; then
                if vllm_http_healthy "$i" "$health_probe_timeout"; then
                    vllm_health_failures=0
                else
                    vllm_health_failures=$((vllm_health_failures + 1))
                    log_warn "Compute #${i} VLLM health probe failed (${vllm_health_failures}/${health_failure_threshold})"
                fi
                if worker_http_healthy "$i" "$health_probe_timeout"; then
                    worker_health_failures=0
                else
                    worker_health_failures=$((worker_health_failures + 1))
                    log_warn "Compute #${i} Worker health probe failed (${worker_health_failures}/${health_failure_threshold})"
                fi
                last_health_check="$now"
                if [ "$vllm_health_failures" -ge "$health_failure_threshold" ]; then
                    failed_component="VLLM"
                    failure_reason="${vllm_health_failures} consecutive health probe failures"
                    break
                fi
                if [ "$worker_health_failures" -ge "$health_failure_threshold" ]; then
                    failed_component="Worker"
                    failure_reason="${worker_health_failures} consecutive health probe failures"
                    break
                fi
            fi

            if [ $((now - last_heartbeat)) -ge "$heartbeat_seconds" ]; then
                log_info "Compute #${i} supervisor heartbeat: VLLM PID ${vllm_pid}, Worker PID ${worker_pid}, health=ok"
                last_heartbeat="$now"
            fi
            sleep "$poll_seconds"
        done

        log_error "Compute #${i} supervisor detected ${failed_component} failure: ${failure_reason}"
        stop_owned_compute
        log_warn "Compute #${i} supervisor restarting owned pair after ${backoff}s"
        wait_compute_backoff
    done
}
cmd_supervise() {
    local target="${1:-control}"
    local instance_index="${2:-}"
    if [ "$target" = "compute" ]; then
        [ -n "$instance_index" ] || { log_error "supervise compute 需要指定实例编号"; return 1; }
        supervise_compute_instance "$instance_index"
        return $?
    fi
    if [ "$target" = "node" ]; then
        supervise_node
        return $?
    fi
    if [ "$target" != "control" ]; then
        log_error "未知 supervise 目标: $target (可选: control|compute|node)"
        return 1
    fi

    run_parent_merge_reconciler
    separator
    echo -e "${CYAN}  MinerU Tianshu - 监督 Watchdog/Scheduler${NC}"
    separator
    init_dirs
    source_ascend_env
    check_instance_conflict || return 1

    supervise_child "watchdog" start_watchdog "${WORKER_LOG_DIR}/watchdog.pid" "tianshu-watchdog" &
    local watchdog_supervisor=$!
    supervise_child "scheduler" start_scheduler "${SCHEDULER_LOG_DIR}/scheduler.pid" "python.*task_scheduler.py" &
    local scheduler_supervisor=$!

    trap 'kill "$watchdog_supervisor" "$scheduler_supervisor" 2>/dev/null || true; wait "$watchdog_supervisor" "$scheduler_supervisor" 2>/dev/null || true; stop_watchdog; stop_scheduler; exit 0' INT TERM EXIT
    wait -n "$watchdog_supervisor" "$scheduler_supervisor"
    local rc=$?
    kill "$watchdog_supervisor" "$scheduler_supervisor" 2>/dev/null || true
    wait "$watchdog_supervisor" "$scheduler_supervisor" 2>/dev/null || true
    return "$rc"
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
    status_watchdog
    echo ""
    status_scheduler
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
        redis)     tail -f "${LOG_DIR}/redis.log" ;;
        vllm)      tail -f "${VLLM_LOG_DIR}"/*.log ;;
        worker)    tail -f "${WORKER_LOG_DIR}"/worker_*.log ;;
        watchdog)  tail -f "${WORKER_LOG_DIR}/watchdog.log" ;;
        scheduler) tail -f "${SCHEDULER_LOG_DIR}"/*.log ;;
        api)       tail -f "${API_LOG_DIR}"/*.log ;;
        mcp)       tail -f "${API_LOG_DIR}/mcp.log" ;;
        frontend)  tail -f "${LOG_DIR}/frontend.log" ;;
        all)       tail -f "${LOG_DIR}"/*/*.log "${LOG_DIR}"/frontend.log "${LOG_DIR}"/redis.log ;;
        *) log_error "未知服务: $target (可选: vllm|worker|watchdog|scheduler|api|mcp|redis|frontend|all)"; exit 1 ;;
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
  bash scripts/tianshu.sh <命令> [服务] [实例编号]

命令:
  start [服务]   启动服务 (默认: all)
  stop [服务]    停止服务 (默认: all)
  restart        重启所有服务
  status         查看所有服务状态
  logs [服务]    实时查看日志 (默认: all)
  drain worker N 将指定 Worker 置为 drain 模式
  supervise      前台监督 control、指定 compute 或完整 node，适配容器 PID1
  test           端到端验证测试
  help           显示帮助

服务 (可选，不指定则操作全部):
  redis          Redis 队列服务 (端口 ${REDIS_PORT})
  vllm           VLLM 推理服务 (端口 ${VLLM_BASE_PORT}-${VLLM_BASE_PORT}$((VLLM_NUM_INSTANCES-1)))
  api            API Server (端口 ${API_PORT})
  mcp            MCP Server (端口 ${MCP_PORT})
  worker         Workers (端口 ${WORKER_BASE_PORT}-${WORKER_BASE_PORT}$((WORKER_NUM_INSTANCES-1)))
  watchdog       看门狗 (Worker 崩溃补齐 + API 端口探活自动恢复,每 60s)
  scheduler      任务调度器 (孤儿恢复+队列监控,默认 10 分钟判定孤儿)
  frontend       前端界面 (端口 ${FRONTEND_PORT})
  all            所有服务

示例:
  bash scripts/tianshu.sh start           # 启动所有服务
  bash scripts/tianshu.sh start vllm      # 仅启动 VLLM
  bash scripts/tianshu.sh stop worker     # 停止全部 Workers
  bash scripts/tianshu.sh stop worker 3   # 仅停止 Worker #3
  bash scripts/tianshu.sh restart worker 3 # drain 验证完成后重启 Worker #3
  bash scripts/tianshu.sh restart worker 3 --force # 明确允许中断后强制重启
  bash scripts/tianshu.sh drain worker 3  # 将 Worker #3 置为 drain
  bash scripts/tianshu.sh supervise       # 前台监督 watchdog/scheduler
  bash scripts/tianshu.sh supervise compute 0 # 前台拥有并回收 Compute #0 子进程
  bash scripts/tianshu.sh supervise node  # Kubernetes PID1:监督 control + 8 组 compute
  bash scripts/tianshu.sh restart         # 重启所有
  bash scripts/tianshu.sh status          # 查看状态
  bash scripts/tianshu.sh logs worker     # 查看 Worker 日志
  bash scripts/tianshu.sh logs scheduler  # 查看调度器日志
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
    SCHEDULER_ORPHAN_RECOVERY_APPLY  是否实际恢复孤儿任务(默认 true)
    OUTPUT_PATH          输出路径(自动派生)
EOF
}

# ============================================================================
# 入口
# ============================================================================

if [ "${TIANSHU_SH_SOURCE_ONLY:-0}" = "1" ]; then
    return 0 2>/dev/null || exit 0
fi

case "${1:-help}" in
    start)    cmd_start "$2" "$3" ;;
    stop)     cmd_stop "$2" "$3" ;;
    restart)  cmd_restart "$2" "$3" "$4" ;;
    drain)    cmd_drain "$2" "$3" ;;
    configure) cmd_configure "$2" "$3" "$4" "$5" "$6" "$7" ;;
    supervise) cmd_supervise "$2" "$3" ;;
    status)   cmd_status ;;
    logs)     cmd_logs "$2" ;;
    test)     cmd_test ;;
    help|--help|-h) cmd_help ;;
    *) log_error "未知命令: $1"; echo ""; cmd_help; exit 1 ;;
esac
