Option Explicit

Dim shell, fso, root, pythonExe, runScript, provider, cmd, ffmpegBin, pathEnv

Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

root = fso.GetParentFolderName(WScript.ScriptFullName)
pythonExe = root & "\venv\Scripts\python.exe"
runScript = root & "\run.py"
provider = "cuda"

If WScript.Arguments.Count > 0 Then
    provider = WScript.Arguments(0)
End If

ffmpegBin = root & "\ffmpeg\bin"
pathEnv = shell.Environment("Process")("PATH")
If fso.FolderExists(ffmpegBin) Then
    pathEnv = ffmpegBin & ";" & pathEnv
End If

shell.Environment("Process")("PATH") = pathEnv
shell.Environment("Process")("DLC_GUI_LAUNCH") = "1"
shell.CurrentDirectory = root

cmd = """" & pythonExe & """ """ & runScript & """ --execution-provider " & provider

' Window style 0 = hidden console (no pythonw / py launcher needed).
shell.Run cmd, 0, False
