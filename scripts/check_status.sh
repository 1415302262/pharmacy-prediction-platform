#!/bin/bash
# 检查QSAR项目运行状态

PROJECT_DIR="/public/home/zhw/cptac/projects/experiment/qsar_project"
PID_FILE="$PROJECT_DIR/.run.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "没有运行中的进程"
    exit 0
fi

PID=$(cat "$PID_FILE")

if ! kill -0 $PID 2>/dev/null; then
    echo "PID文件存在但进程已停止"
    rm -f "$PID_FILE"
    exit 0
fi

echo "正在运行中"
echo "PID: $PID"
echo "CPU使用率: $(ps -p $PID -o %cpu --no-headers)%"
echo "内存使用率: $(ps -p $PID -o %mem --no-headers)%"
echo "运行时长: $(ps -p $PID -o etime --no-headers)"

# 显示最新的日志
if [ -f "$PROJECT_DIR/logs"/*.log ]; then
    LATEST_LOG=$(ls -t $PROJECT_DIR/logs/*.log | head -1)
    echo ""
    echo "=== 最新日志（最后10行）==="
    tail -10 "$LATEST_LOG"
fi
