# MYSTIC AI TRADING SYSTEM - QUICK HEALTH CHECK
# ==============================================  
# Personal laptop version - simple and fast

Write-Host "`nMYSTIC QUICK HEALTH CHECK" -ForegroundColor Cyan
Write-Host "Personal Laptop Version" -ForegroundColor Yellow
Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor DarkGray
Write-Host ""

$results = @()

# Check Python processes
Write-Host "Checking Python processes..." -NoNewline
$pythonProcs = Get-Process python -ErrorAction SilentlyContinue
if ($pythonProcs -and $pythonProcs.Count -ge 5) {
    Write-Host " [OK] ($($pythonProcs.Count) running)" -ForegroundColor Green
    $results += "[OK] Python Processes: $($pythonProcs.Count) running"
} else {
    Write-Host " [WARN] Only $($pythonProcs.Count) found (expected 5+)" -ForegroundColor Yellow
    $results += "[WARN] Python Processes: Only $($pythonProcs.Count) running"
}

# Check Backend API
Write-Host "Checking Backend API..." -NoNewline
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 3 -ErrorAction Stop
    if ($response.StatusCode -eq 200) {
        Write-Host " [OK] (HTTP 200)" -ForegroundColor Green
        $results += "[OK] Backend API: Responding"
    } else {
        Write-Host " [FAIL] HTTP $($response.StatusCode)" -ForegroundColor Red
        $results += "[FAIL] Backend API: HTTP $($response.StatusCode)"
    }
} catch {
    Write-Host " [FAIL] Not responding" -ForegroundColor Red
    $results += "[FAIL] Backend API: Not responding"
}

# Check Redis
Write-Host "Checking Redis..." -NoNewline
$redis = Test-NetConnection -ComputerName localhost -Port 6379 -WarningAction SilentlyContinue
if ($redis.TcpTestSucceeded) {
    Write-Host " [OK] (Port 6379)" -ForegroundColor Green
    $results += "[OK] Redis: Available"
} else {
    Write-Host " [FAIL] Not responding" -ForegroundColor Red
    $results += "[FAIL] Redis: Not responding"
}

# Check Trading endpoints
Write-Host "Checking Trading endpoints..." -NoNewline
try {
    $autobuy = Invoke-WebRequest -Uri "http://localhost:8000/autobuy/status" -TimeoutSec 3 -ErrorAction SilentlyContinue
    $orchestrator = Invoke-WebRequest -Uri "http://localhost:8000/api/orchestrator/status" -TimeoutSec 3 -ErrorAction SilentlyContinue
    
    if ($autobuy -and $orchestrator) {
        Write-Host " [OK] (AutoBuy + Orchestrator)" -ForegroundColor Green
        $results += "[OK] Trading: Both endpoints responding"
    } else {
        Write-Host " [WARN] Partial (some endpoints down)" -ForegroundColor Yellow
        $results += "[WARN] Trading: Some endpoints not responding"
    }
} catch {
    Write-Host " [FAIL] Not responding" -ForegroundColor Red
    $results += "[FAIL] Trading: Endpoints not responding"
}

# Check Memory usage
Write-Host "Checking Memory usage..." -NoNewline
$highMemProcs = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.WorkingSet -gt 500MB }
if (-not $highMemProcs) {
    Write-Host " [OK] (Under 500MB per process)" -ForegroundColor Green
    $results += "[OK] Memory: Usage normal"
} else {
    $totalMB = [math]::Round(($highMemProcs | Measure-Object WorkingSet -Sum).Sum / 1MB, 1)
    Write-Host " [WARN] High usage: ${totalMB}MB" -ForegroundColor Yellow
    $results += "[WARN] Memory: High usage detected"
}

# Trading Safety Check
Write-Host "Checking Trading safety..." -NoNewline
try {
    $autobuyResponse = Invoke-WebRequest -Uri "http://localhost:8000/autobuy/status" -TimeoutSec 3 -ErrorAction SilentlyContinue
    if ($autobuyResponse) {
        $data = $autobuyResponse.Content | ConvertFrom-Json -ErrorAction SilentlyContinue
        if ($data -and ($data.mode -eq "PAPER" -or $data.live_enabled -eq $false)) {
            Write-Host " [SAFE] (Paper mode active)" -ForegroundColor Green
            $results += "[SAFE] Safety: Paper trading mode"
        } else {
            Write-Host " [WARN] CHECK MANUALLY (live mode status unclear)" -ForegroundColor Yellow
            $results += "[WARN] Safety: Verify trading mode"
        }
    } else {
        Write-Host " [UNKNOWN] Cannot verify" -ForegroundColor DarkYellow
        $results += "[UNKNOWN] Safety: Cannot verify mode"
    }
} catch {
    Write-Host " [UNKNOWN] Cannot verify" -ForegroundColor DarkYellow
    $results += "[UNKNOWN] Safety: Cannot verify mode"
}

# Summary
Write-Host "`n" + "="*50 -ForegroundColor Cyan
Write-Host "SUMMARY:" -ForegroundColor Cyan
Write-Host "="*50 -ForegroundColor Cyan

$okCount = ($results | Where-Object { $_ -like "[OK]*" }).Count
$warnCount = ($results | Where-Object { $_ -like "[WARN]*" }).Count  
$errorCount = ($results | Where-Object { $_ -like "[FAIL]*" }).Count
$total = $results.Count

foreach ($result in $results) {
    Write-Host $result
}

Write-Host ""
if ($errorCount -eq 0) {
    Write-Host "SYSTEM STATUS: HEALTHY" -ForegroundColor Green
    Write-Host "Your Mystic AI system is running well!" -ForegroundColor Green
} elseif ($errorCount -le 1) {
    Write-Host "SYSTEM STATUS: MOSTLY OK" -ForegroundColor Yellow
    Write-Host "Minor issues detected but system should work" -ForegroundColor Yellow
} else {
    Write-Host "SYSTEM STATUS: NEEDS ATTENTION" -ForegroundColor Red
    Write-Host "Multiple issues found - check the details above" -ForegroundColor Red
}

Write-Host "`nHealth Score: $okCount/$total components OK" -ForegroundColor $(if ($okCount -eq $total) { "Green" } elseif ($okCount -ge ($total * 0.8)) { "Yellow" } else { "Red" })
Write-Host "="*50 -ForegroundColor Cyan

Write-Host "`nQuick check complete! Press Enter to continue..." -ForegroundColor Yellow
Read-Host
