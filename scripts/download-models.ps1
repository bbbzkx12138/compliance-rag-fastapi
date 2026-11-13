$ErrorActionPreference = "Stop"

$ProjectDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $ProjectDir ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$env:HF_ENDPOINT = "https://hf-mirror.com"
Set-Location $ProjectDir

& $Python -c @"
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='BAAI/bge-small-zh-v1.5',
    local_dir='models/bge-small-zh-v1.5',
)

snapshot_download(
    repo_id='BAAI/bge-reranker-base',
    local_dir='models/bge-reranker-base',
    allow_patterns=[
        'config.json',
        'model.safetensors',
        'sentencepiece.bpe.model',
        'special_tokens_map.json',
        'tokenizer.json',
        'tokenizer_config.json',
    ],
)
"@
