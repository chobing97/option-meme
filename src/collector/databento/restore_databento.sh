#!/usr/bin/env bash
# Google Drive 에서 Databento 원본 데이터(dbn.zst)를 복원한다.
#
# 사전 설정:
#   1) brew install rclone
#   2) rclone config  →  remote 이름: "gdrive", type: Google Drive
#
# 사용법:
#   bash src/collector/databento/restore_databento.sh          # 전체 복원
#   bash src/collector/databento/restore_databento.sh --dry-run # 변경분 확인만
#
# run_pipeline.py 의 collector 단계에서 데이터가 없으면 자동 호출됨.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

REMOTE="gdrive:option-meme-data"
STOCK_DST="${PROJECT_DIR}/data/raw/stock/databento"
OPTIONS_DST="${PROJECT_DIR}/data/raw/options/databento"

EXTRA_ARGS="${*:-}"

# rclone 설치 확인
if ! command -v rclone &>/dev/null; then
    echo "Warning: rclone 이 설치되어 있지 않습니다. 복원을 건너뜁니다."
    echo "  brew install rclone && rclone config"
    exit 0
fi

# rclone remote 확인
if ! rclone listremotes | grep -q "^gdrive:"; then
    echo "Warning: rclone remote 'gdrive' 가 설정되어 있지 않습니다. 복원을 건너뜁니다."
    echo "  rclone config  →  이름: gdrive, type: Google Drive"
    exit 0
fi

echo "=== Databento 복원 ← Google Drive ==="

mkdir -p "${STOCK_DST}" "${OPTIONS_DST}"

echo ""
echo "  [1/2] 주식 데이터 → ${STOCK_DST}"
rclone sync "${REMOTE}/raw/stock/databento/" "${STOCK_DST}/" --progress ${EXTRA_ARGS}

echo ""
echo "  [2/2] 옵션 데이터 → ${OPTIONS_DST}"
rclone sync "${REMOTE}/raw/options/databento/" "${OPTIONS_DST}/" --progress ${EXTRA_ARGS}

echo ""
echo "=== 복원 완료 ==="
