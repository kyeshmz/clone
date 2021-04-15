@echo off

call scripts/settings_windows.bat

set CONFIG=fomm/config/vox-adv-256.yaml

set PYTHONPATH=%PYTHONPATH%;%CD%;%CD%/fomm
call python afy/MI_fomm.py --OscMainIP "10.10.3.48" --instance_id 2 --config %CONFIG% --relative --adapt_scale --no-pad --checkpoint vox-adv-cpk.pth.tar %*
