import subprocess
blender_path = "C:\\Program Files\\Blender Foundation\\blender.exe"
blender_script_path = "B:\\Master arbeit\\blender_script.py"
command = [blender_path,"--background","--python",blender_script_path]
subprocess.run(command)