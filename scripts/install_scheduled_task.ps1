$action = New-ScheduledTaskAction -Execute "wscript.exe" -Argument "`"D:\ALL PROJECT\OJT PROJECT\OmniVoice\scripts\start_service_hidden.vbs`""
$trigger = New-ScheduledTaskTrigger -AtLogOn
Register-ScheduledTask -TaskName "OmniVoice_Service" -Action $action -Trigger $trigger -Force
Start-ScheduledTask -TaskName "OmniVoice_Service"
Write-Host "OmniVoice Windows Scheduled Task Registered and Started Successfully."

