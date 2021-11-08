@echo off

@REM call scripts/settings_windows.bat

set CONFIG=fomm/config/vox-adv-256.yaml

set PYTHONPATH=%PYTHONPATH%;%CD%;%CD%/fomm
call python afy/MI_fomm.py --OscMainIP "192.168.10.101" --instance_id 2 --cuda_id 1 --config %CONFIG% --relative --adapt_scale --no-pad --checkpoint vox-adv-cpk.pth.tar %*
