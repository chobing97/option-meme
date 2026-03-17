#!/usr/bin/env bash
# Databento 원본 데이터(dbn.zst)를 Google Drive 로 백업한다.
#
# 사전 설정:
#   1) brew install rclone
#   2) rclone config  →  remote 이름: "gdrive", type: Google Drive
#
# 사용법:
#   bash src/collector/databento/backup_databento.sh          # 전체 백업
#   bash src/collector/databento/backup_databento.sh --dry-run # 변경분 확인만

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

REMOTE="gdrive:option-meme-data"
STOCK_SRC="${PROJECT_DIR}/data/raw/stock/databento"
OPTIONS_SRC="${PROJECT_DIR}/data/raw/options/databento"

EXTRA_ARGS="${*:-}"

# rclone 설치 확인
if ! command -v rclone &>/dev/null; then
    echo "Error: rclone 이 설치되어 있지 않습니다."
    echo "  brew install rclone && rclone config"
    exit 1
fi

# rclone remote 확인
if ! rclone listremotes | grep -q "^gdrive:"; then
    echo "Error: rclone remote 'gdrive' 가 설정되어 있지 않습니다."
    echo "  rclone config  →  이름: gdrive, type: Google Drive"
    exit 1
fi

echo "=== Databento 백업 → Google Drive ==="

if [ -d "${STOCK_SRC}" ]; then
    echo ""
    echo "  [1/2] 주식 데이터: ${STOCK_SRC}"
    rclone sync "${STOCK_SRC}/" "${REMOTE}/raw/stock/databento/" --progress ${EXTRA_ARGS}
else
    echo "  [1/2] 주식 데이터: 없음 (skip)"
fi

if [ -d "${OPTIONS_SRC}" ]; then
    echo ""
    echo "  [2/2] 옵션 데이터: ${OPTIONS_SRC}"
    rclone sync "${OPTIONS_SRC}/" "${REMOTE}/raw/options/databento/" --progress ${EXTRA_ARGS}
else
    echo "  [2/2] 옵션 데이터: 없음 (skip)"
fi

echo ""
echo "=== 백업 완료 ==="
