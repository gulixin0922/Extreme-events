export OPENAI_API_KEY=sk-LhuVXG7YK0UsoMfdmSsJqOAojWjcyXDmtrcj5b7ujTJTSZQy
export OPENAI_BASE_URL=http://35.220.164.252:3888/v1
export MODEL_NAME=gpt-4o
export MAX_WORKERS=64

# DATASET_ROOT=/mnt/shared-storage-user/intern7shared/gulixin/data/fengwu/0202/storm0201
# RAW_DATA_BASE_PATH=/mnt/shared-storage-user/intern7shared/gulixin/data/fengwu/0202/storm
# TARGET_FILE=Image_Only.json
# python evaluate.py $DATASET_ROOT $RAW_DATA_BASE_PATH $TARGET_FILE

DATASET_ROOT=/mnt/shared-storage-user/intern7shared/gulixin/data/fengwu/0202/earthquake0201
RAW_DATA_BASE_PATH=/mnt/shared-storage-user/intern7shared/gulixin/data/fengwu/0202/earthquake
TARGET_FILE=Image_Only.json
python evaluate.py $DATASET_ROOT $RAW_DATA_BASE_PATH $TARGET_FILE