# VHS Tape Script: Rune Infer Text-to-Speech (TTS) Screencast
# Requirements: https://github.com/charmbracelet/vhs
#
# To render this tape into docs/demo_tts.gif:
#   vhs demos/tts/tts.tape
# Or run:
#   .\demos\record.ps1 -Name tts

Output docs/demo_tts.gif

Set Shell "powershell"
Set FontSize 14
Set FontFamily "Consolas"
Set Width 1280
Set Height 760
Set LineHeight 1.15
Set Padding 20
Set Theme "Catppuccin Mocha"

# 1. Silently configure environment, launch background TTS driver, and boot dashboard
Hide
Type "$env:RUNE_API_KEY = 'rune-demo-secure-key'"
Enter
Type "Start-Process powershell -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File .\demos\tts\driver.ps1' -WindowStyle Hidden"
Enter
Type ".\target\release\rune-infer.exe --config config/models.json"
Enter
Sleep 1500ms
Show

# 2. Frame 1: Dashboard starts live with models initialized and green 🔒 API key badge
Sleep 2s

# 3. Request arrives: Audio8 TTS Preview 0.6B loads and synthesizes speech in real-time
Sleep 12s

# 4. Hold triumphant final view displaying completed TTS request, latency & throughput metrics
Sleep 4s
Type "q"
Sleep 1s
