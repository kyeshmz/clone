import subprocess
import psutil
import time

dict_pids = {
    p.info['pid']: p.info['name']
    for p in psutil.process_iter(attrs=['pid', 'name'])
}

# def run_batch_file(file_path):
#     subprocess.Popen(file_path,creationflags=subprocess.CREATE_NEW_CONSOLE)

if 'TouchDesigner.exe' in dict_pids.values():
    print('Touchdesigner is running')
    pass
else:
	print('starting 1')
	first = r'./B1_MI_AvtBridge_2.toe'
	subprocess.Popen(['start', first], shell=True)
	print('end 1')
	print('Restarted TouchDesigner.')