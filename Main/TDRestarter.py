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
	first = r'./1_MI_MainSystem3.toe'
	subprocess.Popen(['start', first], shell=True)
	print('end 1')
	time.sleep(10)
	print('starting 2')
	second = r'./2_MI_BlendMorph_P1.toe'
	subprocess.Popen(['start', second], shell=True)
	print('end 2')
	time.sleep(10)
	print('starting 3')
	third = r'./2_MI_BlendMorph_P2.toe'
	subprocess.Popen(['start', third], shell=True)
	print('end 3')
	time.sleep(10)
	print('starting 4')
	fourth = r'./3_MI_AvtBridge_1.toe'
	subprocess.Popen(['start', fourth], shell=True)
	print('end 4')
	time.sleep(10)
	print('starting 5')
	filepath = r'./4_launch_avaterify.bat'
	subprocess.Popen("4_launch_avaterify.bat", cwd=r"C:\Users\admin\Dropbox\Project\66_MorphingIdentity\1_dev\7_MorphingIdentity\MorphingIdentity\Main", shell=True)
	print('end 5')
	print('Restarted TouchDesigner.')