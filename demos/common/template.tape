# VHS Tape Script: Template for Rune Infer Screencasts
# Requirements: https://github.com/charmbracelet/vhs
#
# To render this tape:
#   vhs demos/<feature>/<feature>.tape
# Or using the orchestrator:
#   .\demos\record.ps1 -Name <feature>

Output docs/demo_feature.gif

Set Shell "powershell"
Set FontSize 14
Set FontFamily "Consolas"
Set Width 1280
Set Height 760
Set LineHeight 1.15
Set Padding 20
Set Theme "Catppuccin Mocha"

# 1. Silently configure environment, launch background driver, and boot dashboard
Hide
Type "$env:RUNE_API_KEY = 'rune-demo-secure-key'"
Enter
Type "Start-Process powershell -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File .\demos\feature\driver.ps1' -WindowStyle Hidden"
Enter
Type ".\target\release\rune-infer.exe --config config/models.json"
Enter
Sleep 1500ms
Show

# 2. Frame 1: Dashboard starts live with initialized state
Sleep 2s

# 3. Action Sequence: Background driver issues requests to simulate workload
Sleep 15s

# 4. Hold triumphant final view on dashboard with completed requests, then exit cleanly
Sleep 4s
Type "q"
Sleep 1s
