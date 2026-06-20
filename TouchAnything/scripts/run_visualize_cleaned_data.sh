#!/bin/bash
# TouchAnything cleaned-data visualization script
# Visualization videos are saved as visualization.mp4 under each trajectory folder

# Set paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

INPUT_DIR="$PROJECT_ROOT/datasets/TouchAnything_Datasets_opensource"

# Default parameters
WORKERS=8
FPS=30
FORCE=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        --force)
            FORCE="--force"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--workers N] [--fps N] [--force]"
            exit 1
            ;;
    esac
done

echo "=================================="
echo "TouchAnything data visualization"
echo "=================================="
echo "Dataset directory: $INPUT_DIR"
echo "Output location: visualization.mp4 under each trajectory folder"
echo "Worker processes: $WORKERS"
echo "video frame rate: $FPS"
echo "Force regeneration: ${FORCE:-No}"
echo "=================================="
echo ""

# Run visualization
cd "$PROJECT_ROOT"
python scripts/visualize_cleaned_data.py \
    --root "$INPUT_DIR" \
    --batch \
    --workers "$WORKERS" \
    --fps "$FPS" \
    $FORCE

echo ""
echo "=================================="
echo "Visualization complete"
echo "=================================="
