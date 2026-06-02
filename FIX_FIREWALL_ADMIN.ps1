# Run this script AS ADMINISTRATOR to fix async Redis connections
# Right-click PowerShell -> Run as Administrator -> cd to this folder -> .\FIX_FIREWALL_ADMIN.ps1

Write-Host "`n===================================================" -ForegroundColor Yellow
Write-Host "  FIXING WINDOWS FIREWALL FOR ASYNC REDIS        " -ForegroundColor Yellow
Write-Host "===================================================`n" -ForegroundColor Yellow

# Check if running as admin
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "ERROR: This script must be run as Administrator!" -ForegroundColor Red
    Write-Host "`nRight-click PowerShell -> Run as Administrator" -ForegroundColor Yellow
    Read-Host "`nPress Enter to exit"
    exit 1
}

Write-Host "Running as Administrator`n" -ForegroundColor Green

# Add firewall rules for both Python executables
$rules = @(
    @{
        Name = "Python venv to WSL2 Redis"
        Path = "C:\Users\lloyd\Mystic-Codebase\venv\Scripts\python.exe"
    },
    @{
        Name = "Python base to WSL2 Redis"
        Path = "C:\Users\lloyd\AppData\Local\Programs\Python\Python310\python.exe"
    }
)

foreach ($rule in $rules) {
    try {
        # Remove existing rule if present
        Remove-NetFirewallRule -DisplayName $rule.Name -ErrorAction SilentlyContinue | Out-Null
        
        # Add new rule
        New-NetFirewallRule `
            -DisplayName $rule.Name `
            -Direction Outbound `
            -Program $rule.Path `
            -Action Allow `
            -Profile Any `
            -ErrorAction Stop | Out-Null
        
        Write-Host "SUCCESS: Added firewall rule: $($rule.Name)" -ForegroundColor Green
    }
    catch {
        Write-Host "FAILED: $($rule.Name) - $_" -ForegroundColor Red
    }
}

Write-Host "`n===================================================" -ForegroundColor Green
Write-Host "  FIREWALL RULES APPLIED - NOW TEST SERVICES      " -ForegroundColor Green
Write-Host "===================================================`n" -ForegroundColor Green

Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Close this admin PowerShell" -ForegroundColor White
Write-Host "2. Go back to your regular PowerShell" -ForegroundColor White
Write-Host "3. Run: .\START_MYSTIC.ps1" -ForegroundColor White
Write-Host ""

Read-Host "Press Enter to close"

