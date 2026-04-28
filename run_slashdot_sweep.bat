@echo off
set PYTHON=C:\conda\envs\gnn\python.exe
set ROOT=C:\Users\12143\Desktop\pythonProject\gnn
set LOGFILE=%ROOT%\run_slashdot_sweep.log

cd /d %ROOT%

echo [%date% %time%] Starting slashdot algo_sweep >> %LOGFILE%

for %%c in (
    slashdot_random
    slashdot_aa
    slashdot_cn
    slashdot_jaccard
    slashdot_pa
    slashdot_mlp
    slashdot_node_emb
    slashdot_gnn
    slashdot_gnn_sum
    slashdot_gnn_concat
    slashdot_ground_truth
) do (
    echo [%date% %time%] Running %%c >> %LOGFILE%
    %PYTHON% -m src.online.loop --config configs/online/algo_sweep_slashdot/%%c.yaml >> %LOGFILE% 2>&1
    echo [%date% %time%] Done %%c >> %LOGFILE%
)

echo [%date% %time%] All slashdot experiments done >> %LOGFILE%
