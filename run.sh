eval_gpt4o() {
    export OPENAI_BASE_URL=http://35.220.164.252:3888/v1
    export OPENAI_API_KEY=sk-LhuVXG7YK0UsoMfdmSsJqOAojWjcyXDmtrcj5b7ujTJTSZQy
    python evaluate.py \
        --dataset-root /mnt/shared-storage-user/intern7shared/gulixin/data/fengwu/0202/earthquake0201 \
        --raw-data-base-path /mnt/shared-storage-user/intern7shared/gulixin/data/fengwu/0202/earthquake \
        --target-file Image_Only.json \
        --model-name gpt-4o \
        --image-max-num 500 \
        --temperature 0.1 \
        --max-tokens 300
}

eval_gemini3pro() {
    export OPENAI_BASE_URL=http://35.220.164.252:3888/v1
    export OPENAI_API_KEY=sk-NZgVNOjYoi1dLGFpdImQfwtjJJdqoD4NeVfaISTyi7FJYOEs
    python evaluate.py \
        --dataset-root /mnt/shared-storage-user/intern7shared/gulixin/data/fengwu/0202/earthquake0201 \
        --raw-data-base-path /mnt/shared-storage-user/intern7shared/gulixin/data/fengwu/0202/earthquake \
        --target-file Image_Only.json \
        --model-name gemini-3-pro-preview-thinking \
        --image-max-num 600 \
        --temperature 1.0 \
        --max-tokens 65536
}

eval_gemini3pro