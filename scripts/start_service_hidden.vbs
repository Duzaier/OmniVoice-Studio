Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "D:\ALL PROJECT\OJT PROJECT\OmniVoice"
WshShell.Run """C:\Users\Duzaier\AppData\Roaming\uv\python\cpython-3.13.15-windows-x86_64-none\python.exe"" ""D:\ALL PROJECT\OJT PROJECT\OmniVoice\workstation.py"" --port 8001 --no-browser", 0, False

