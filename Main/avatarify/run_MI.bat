@echo off

call scripts/settings_windows.bat

set CONFIG=fomm/config/vox-adv-256.yaml

set PYTHONPATH=%PYTHONPATH%;%CD%;%CD%/fomm
call python afy/spout_osc_fomm.py --config %CONFIG% --relative --adapt_scale --no-pad --checkpoint vox-adv-cpk.pth.tar %*
