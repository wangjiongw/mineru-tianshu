#!/bin/bash
# ============================================================================
# MinerU Tianshu 日志管理脚本
# ============================================================================
#
# 功能：统一管理所有系统日志
# - 查看日志
# - 清理日志
# - 归档日志
# - 日志轮转
#
# 使用方式：
#   bash scripts/log_manager.sh list         # 列出所有日志
#   bash scripts/log_manager.sh view vllm    # 查看VLLM日志
#   bash scripts/log_manager.sh tail worker  # 实时查看Worker日志
#   bash scripts/log_manager.sh archive      # 归档旧日志
#   bash scripts/log_manager.sh clean        # 清理归档日志
#   bash scripts/log_manager.sh rotate       # 手动日志轮转
#
# ============================================================================

set -e

# 项目路径
PROJECT_ROOT="/data/projects/mineru/mineru-tianshu"
LOG_DIR="$PROJECT_ROOT/data/logs"

# 日志保留策略
LOG_RETENTION_DAYS=30  # 日志保留天数
ARCHIVE_AFTER_DAYS=7   # 多少天后归档
MAX_ARCHIVE_SIZE=1G     # 归档目录最大大小

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# 函数定义
# ============================================================================

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    cat << EOF
${BLUE}MinerU Tianshu 日志管理工具${NC}

使用方式:
  $0 <命令> [参数]

命令:
  ${GREEN}list${NC}                    列出所有日志文件及大小
  ${GREEN}view${NC} <类型> [文件]     查看指定类型的日志
  ${GREEN}tail${NC} <类型> [文件]     实时查看日志（类似tail -f）
  ${GREEN}search${NC} <类型> <关键词>  在日志中搜索关键词
  ${GREEN}archive${NC}                归档旧日志
  ${GREEN}clean${NC}                  清理归档日志
  ${GREEN}rotate${NC}                 手动执行日志轮转
  ${GREEN}stats${NC}                  显示日志统计信息
  ${GREEN}help${NC}                   显示此帮助信息

日志类型:
  vllm        - VLLM服务日志
  worker      - Worker进程日志
  api         - API Server日志
  all         - 所有日志

示例:
  $0 list                          # 列出所有日志
  $0 view vllm                     # 查看VLLM日志列表
  $0 view vllm vllm_npu0_port30025.log  # 查看指定日志文件
  $0 tail worker                   # 实时查看Worker日志
  $0 search worker "error"         # 在Worker日志中搜索error
  $0 archive                       # 归档7天前的日志
  $0 clean                         # 清理归档目录
  $0 stats                         # 显示统计信息

配置:
  日志目录: $LOG_DIR
  保留天数: $LOG_RETENTION_DAYS
  归档天数: $ARCHIVE_AFTER_DAYS

EOF
}

list_logs() {
    local log_type=${1:-"all"}

    echo "=========================================="
    echo "日志文件列表"
    echo "=========================================="
    echo ""

    case "$log_type" in
        vllm)
            echo "【VLLM服务日志】"
            echo ""
            if [ -d "$LOG_DIR/vllm" ] && [ "$(ls -A $LOG_DIR/vllm)" ]; then
                ls -lh "$LOG_DIR/vllm" | awk 'NR>1 {printf "  %-40s %10s\n", $9, $5}'
                echo ""
                echo "总大小: $(du -sh $LOG_DIR/vllm | awk '{print $1}')"
            else
                echo "  (无日志文件)"
            fi
            ;;
        worker)
            echo "【Worker日志】"
            echo ""
            if [ -d "$LOG_DIR/worker" ] && [ "$(ls -A $LOG_DIR/worker)" ]; then
                ls -lh "$LOG_DIR/worker" | awk 'NR>1 {printf "  %-40s %10s\n", $9, $5}'
                echo ""
                echo "总大小: $(du -sh $LOG_DIR/worker | awk '{print $1}')"
            else
                echo "  (无日志文件)"
            fi
            ;;
        api)
            echo "【API Server日志】"
            echo ""
            if [ -d "$LOG_DIR/api" ] && [ "$(ls -A $LOG_DIR/api)" ]; then
                ls -lh "$LOG_DIR/api" | awk 'NR>1 {printf "  %-40s %10s\n", $9, $5}'
                echo ""
                echo "总大小: $(du -sh $LOG_DIR/api | awk '{print $1}')"
            else
                echo "  (无日志文件)"
            fi
            ;;
        all)
            list_logs "vllm"
            echo ""
            list_logs "worker"
            echo ""
            list_logs "api"
            echo ""
            echo "【归档日志】"
            if [ -d "$LOG_DIR/archive" ] && [ "$(ls -A $LOG_DIR/archive)" ]; then
                echo ""
                ls -lh "$LOG_DIR/archive" | awk 'NR>1 {printf "  %-40s %10s\n", $9, $5}'
                echo ""
                echo "归档总大小: $(du -sh $LOG_DIR/archive | awk '{print $1}')"
            else
                echo "  (无归档日志)"
            fi
            ;;
        *)
            log_error "未知的日志类型: $log_type"
            return 1
            ;;
    esac

    echo ""
}

view_log() {
    local log_type=$1
    local log_file=$2

    local target_dir=""

    case "$log_type" in
        vllm)
            target_dir="$LOG_DIR/vllm"
            ;;
        worker)
            target_dir="$LOG_DIR/worker"
            ;;
        api)
            target_dir="$LOG_DIR/api"
            ;;
        *)
            log_error "未知的日志类型: $log_type"
            return 1
            ;;
    esac

    if [ ! -d "$target_dir" ]; then
        log_error "日志目录不存在: $target_dir"
        return 1
    fi

    if [ -z "$log_file" ]; then
        # 列出该类型所有日志
        echo "=========================================="
        echo "可用的日志文件 ($log_type)"
        echo "=========================================="
        echo ""
        ls -lh "$target_dir" | awk 'NR>1 {printf "  %-40s %10s %s\n", $9, $5, $6}'
        echo ""
        echo "使用方式: $0 view $log_type <文件名>"
        echo ""
        return 0
    fi

    local full_path="$target_dir/$log_file"

    if [ ! -f "$full_path" ]; then
        log_error "日志文件不存在: $full_path"
        return 1
    fi

    # 查看日志（使用less，支持翻页）
    less -R "+/?" "$full_path"
}

tail_log() {
    local log_type=$1
    local log_file=$2

    local target_dir=""

    case "$log_type" in
        vllm)
            target_dir="$LOG_DIR/vllm"
            ;;
        worker)
            target_dir="$LOG_DIR/worker"
            ;;
        api)
            target_dir="$LOG_DIR/api"
            ;;
        all)
            # 同时监控多个日志文件
            log_info "监控所有日志文件 (Ctrl+C 退出)"
            echo ""
            tail -f $LOG_DIR/vllm/*.log 2>/dev/null &
            tail -f $LOG_DIR/worker/*.log 2>/dev/null &
            wait
            return 0
            ;;
        *)
            log_error "未知的日志类型: $log_type"
            return 1
            ;;
    esac

    if [ ! -d "$target_dir" ]; then
        log_error "日志目录不存在: $target_dir"
        return 1
    fi

    if [ -z "$log_file" ]; then
        # 监控该类型所有日志
        log_info "监控 $log_type 日志 (Ctrl+C 退出)"
        echo ""
        tail -f $target_dir/*.log 2>/dev/null
    else
        local full_path="$target_dir/$log_file"

        if [ ! -f "$full_path" ]; then
            log_error "日志文件不存在: $full_path"
            return 1
        fi

        log_info "监控 $full_path (Ctrl+C 退出)"
        echo ""
        tail -f "$full_path"
    fi
}

search_log() {
    local log_type=$1
    local keyword=$2

    if [ -z "$keyword" ]; then
        log_error "请提供搜索关键词"
        return 1
    fi

    local target_dir=""

    case "$log_type" in
        vllm)
            target_dir="$LOG_DIR/vllm"
            ;;
        worker)
            target_dir="$LOG_DIR/worker"
            ;;
        api)
            target_dir="$LOG_DIR/api"
            ;;
        all)
            target_dir="$LOG_DIR"
            ;;
        *)
            log_error "未知的日志类型: $log_type"
            return 1
            ;;
    esac

    if [ ! -d "$target_dir" ]; then
        log_error "日志目录不存在: $target_dir"
        return 1
    fi

    echo "=========================================="
    echo "在 $log_type 日志中搜索: $keyword"
    echo "=========================================="
    echo ""

    if [ "$log_type" = "all" ]; then
        grep -r -n --color=always "$keyword" "$target_dir"/{vllm,worker,api} 2>/dev/null | head -50
    else
        grep -r -n --color=always "$keyword" "$target_dir" 2>/dev/null | head -50
    fi

    echo ""
}

archive_logs() {
    log_info "归档旧日志..."

    local archive_dir="$LOG_DIR/archive/$(date +%Y%m%d)"
    mkdir -p "$archive_dir"

    # 查找需要归档的日志文件
    local archived_count=0

    for log_type in vllm worker api; do
        local type_dir="$LOG_DIR/$log_type"

        if [ ! -d "$type_dir" ]; then
            continue
        fi

        # 归档$ARCHIVE_AFTER_DAYS天前修改的日志
        while IFS= read -r -d '' log_file; do
            local filename=$(basename "$log_file")
            local archived_file="$archive_dir/${log_type}_${filename}"

            mv "$log_file" "$archived_file"
            archived_count=$((archived_count + 1))

            log_info "  归档: $filename"
        done < <(find "$type_dir" -type f -name "*.log" -mtime +$ARCHIVE_AFTER_DAYS -print0 2>/dev/null)
    done

    if [ $archived_count -eq 0 ]; then
        log_warn "没有需要归档的日志文件"
    else
        log_info "归档完成，共 $archived_count 个文件"
        log_info "归档位置: $archive_dir"
    fi

    echo ""
}

clean_archives() {
    log_info "清理归档日志..."

    # 删除$LOG_RETENTION_DAYS天前的归档
    local deleted_count=0

    while IFS= read -r -d '' archive_dir; do
        log_info "  删除: $archive_dir"
        rm -rf "$archive_dir"
        deleted_count=$((deleted_count + 1))
    done < <(find "$LOG_DIR/archive" -type d -mtime +$LOG_RETENTION_DAYS -print0 2>/dev/null)

    if [ $deleted_count -eq 0 ]; then
        log_warn "没有需要清理的归档"
    else
        log_info "清理完成，共删除 $deleted_count 个归档目录"
    fi

    # 检查归档目录大小
    local archive_size=$(du -sh "$LOG_DIR/archive" 2>/dev/null | awk '{print $1}')
    log_info "当前归录大小: $archive_size"

    echo ""
}

rotate_logs() {
    log_info "执行日志轮转..."

    for log_type in vllm worker api; do
        local type_dir="$LOG_DIR/$log_type"

        if [ ! -d "$type_dir" ]; then
            continue
        fi

        # 重命名当前日志文件
        while IFS= read -r -d '' log_file; do
            local filename=$(basename "$log_file")
            local timestamp=$(date +%Y%m%d_%H%M%S)
            local rotated_file="${log_file}.${timestamp}"

            # 重命名
            mv "$log_file" "$rotated_file"

            # 如果文件过大，压缩
            local file_size=$(stat -f%z "$rotated_file" 2>/dev/null || stat -c%s "$rotated_file" 2>/dev/null)
            if [ "$file_size" -gt 104857600 ]; then  # 大于100MB
                gzip "$rotated_file"
                log_info "  轮转并压缩: $filename"
            else
                log_info "  轮转: $filename"
            fi
        done < <(find "$type_dir" -type f -name "*.log" -size +10M -print0 2>/dev/null)
    done

    log_info "日志轮转完成"
    echo ""
}

show_stats() {
    echo "=========================================="
    echo "日志统计信息"
    echo "=========================================="
    echo ""

    # 各类型日志大小
    echo "【日志大小】"
    for log_type in vllm worker api archive; do
        local type_dir="$LOG_DIR/$log_type"
        if [ -d "$type_dir" ]; then
            local size=$(du -sh "$type_dir" 2>/dev/null | awk '{print $1}')
            local count=$(find "$type_dir" -type f -name "*.log*" 2>/dev/null | wc -l)
            printf "  %-10s %10s  (%4d 文件)\n" "$log_type:" "$size" "$count"
        fi
    done
    echo ""

    # 总大小
    local total_size=$(du -sh "$LOG_DIR" 2>/dev/null | awk '{print $1}')
    echo "  总计:      $total_size"
    echo ""

    # 日志文件数量
    echo "【文件数量】"
    local total_files=$(find "$LOG_DIR" -type f -name "*.log*" 2>/dev/null | wc -l)
    echo "  总日志文件: $total_files"
    echo ""

    # 最近的日志活动
    echo "【最近活动】"
    echo "  最新VLLM日志:"
    ls -lt "$LOG_DIR/vllm"/*.log 2>/dev/null | head -1 | awk '{print "    " $9 " (" $6 " " $7 " " $8 ")"}'
    echo "  最新Worker日志:"
    ls -lt "$LOG_DIR/worker"/*.log 2>/dev/null | head -1 | awk '{print "    " $9 " (" $6 " " $7 " " $8 ")"}'
    echo ""
}

# ============================================================================
# 主命令
# ============================================================================

case "$1" in
    list)
        list_logs "$2"
        ;;

    view)
        if [ -z "$2" ]; then
            log_error "请指定日志类型"
            show_help
            exit 1
        fi
        view_log "$2" "$3"
        ;;

    tail)
        if [ -z "$2" ]; then
            log_error "请指定日志类型"
            show_help
            exit 1
        fi
        tail_log "$2" "$3"
        ;;

    search)
        if [ -z "$2" ] || [ -z "$3" ]; then
            log_error "请指定日志类型和搜索关键词"
            show_help
            exit 1
        fi
        search_log "$2" "$3"
        ;;

    archive)
        archive_logs
        ;;

    clean)
        clean_archives
        ;;

    rotate)
        rotate_logs
        ;;

    stats)
        show_stats
        ;;

    help|--help|-h)
        show_help
        ;;

    *)
        log_error "未知命令: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
