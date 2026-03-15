#!/bin/bash
# QSAR项目后台运行脚本（带CPU监控）

PROJECT_DIR="/public/home/zhw/cptac/projects/experiment/qsar_project"
LOG_DIR="$PROJECT_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/run_$TIMESTAMP.log"
PID_FILE="$PROJECT_DIR/.run.pid"

# 创建日志目录
mkdir -p "$LOG_DIR"

# CPU监控函数
monitor_cpu() {
    local pid=$1
    local log=$2
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] CPU监控已启动，PID: $pid" >> "$log"

    while kill -0 $pid 2>/dev/null; do
        # 获取进程CPU使用率和内存使用情况
        cpu_usage=$(ps -p $pid -o %cpu --no-headers 2>/dev/null || echo "0.0")
        mem_usage=$(ps -p $pid -o %mem --no-headers 2>/dev/null || echo "0.0")

        # 获取系统整体CPU负载
        load_avg=$(cat /proc/loadavg | awk '{print $1}')

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] CPU: ${cpu_usage}%, Mem: ${mem_usage}%, Load: ${load_avg}" >> "$log"

        # CPU使用率超过80%时发出警告
        if (( $(echo "$cpu_usage > 80" | bc -l) )); then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] 警告: CPU使用率过高 (${cpu_usage}%)" >> "$log"
        fi

        # 系统负载过高时暂停并等待
        if (( $(echo "$load_avg > $(nproc)" | bc -l) )); then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] 系统负载过高，暂停30秒" >> "$log"
            sleep 30
        else
            sleep 5
        fi
    done

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 进程已结束，监控停止" >> "$log"
}

# 检查是否已有运行中的进程
if [ -f "$PID_FILE" ]; then
    old_pid=$(cat "$PID_FILE")
    if kill -0 $old_pid 2>/dev/null; then
        echo "错误: 已有运行中的进程 (PID: $old_pid)"
        echo "如需重新运行，请先停止旧进程: kill $old_pid"
        exit 1
    fi
fi

# 设置Python环境
source ~/.bashrc

# 切换到项目目录
cd "$PROJECT_DIR" || exit 1

# 运行主程序并记录PID
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始运行QSAR项目..." | tee -a "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 日志文件: $LOG_FILE" | tee -a "$LOG_FILE"

python run.py >> "$LOG_FILE" 2>&1 &
MAIN_PID=$!
echo $MAIN_PID > "$PID_FILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 主程序PID: $MAIN_PID" >> "$LOG_FILE"

# 启动CPU监控
monitor_cpu $MAIN_PID "$LOG_FILE"

# 等待主程序结束
wait $MAIN_PID
exit_code=$?

# 清理PID文件
rm -f "$PID_FILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 程序已退出，退出码: $exit_code" >> "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 完整日志: $LOG_FILE"

exit $exit_code
