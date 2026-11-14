$ErrorActionPreference = "Stop"

$ProjectDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Body = @{
    modes = @("keyword", "vector", "hybrid", "rerank")
    save_report = $true
} | ConvertTo-Json

$Report = Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/api/evaluate" `
    -Method Post `
    -ContentType "application/json" `
    -Body $Body `
    -TimeoutSec 300

Write-Output "Report ID: $($Report.report_id)"
Write-Output "JSON: $($Report.report_files.json)"
Write-Output "Markdown: $($Report.report_files.markdown)"
Write-Output "Quality gate: $($Report.quality_gate.passed)"
