# VHS Tape Script: Rune Infer 60-second Terminal Screencast
# Requirements: https://github.com/charmbracelet/vhs
#
# To render this tape into public/demo.gif:
#   vhs demos/overview/overview.tape
# Or run:
#   .\demos\record.ps1 -Name overview

Output public/demo.gif

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
Type "Start-Process powershell -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File .\demos\overview\driver.ps1' -WindowStyle Hidden"
Enter
Type ".\target\release\rune-infer.exe --config config/models.json"
Enter
Sleep 1500ms
Show

# 2. Frame 1: Dashboard starts live with 0 models loaded and green 🔒 API key required badge
Sleep 2s

# 3. Request 1 arrives: lazy-loading Qwen3.8 27B (VRAM rises to ~13.8 GB)
Sleep 20s

# 4. Request 2 arrives: LRU eviction of Qwen, Bonsai loads safely within 16GB limit
Sleep 22s

# 5. Hold triumphant final view with both completed requests, then exit cleanly
Sleep 4s
Type "q"
Sleep 1s
