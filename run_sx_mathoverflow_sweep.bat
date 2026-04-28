@echo off
set PYTHON=C:\conda\envs\gnn\python.exe
set ROOT=C:\Users\12143\Desktop\pythonProject\gnn
set LOGFILE=%ROOT%\run_sx_mathoverflow_sweep.log

cd /d %ROOT%

echo [%date% %time%] Starting sx_mathoverflow algo_sweep >> %LOGFILE%

for %%c in (
    sx_mathoverflow_random
    sx_mathoverflow_aa
    sx_mathoverflow_cn
    sx_mathoverflow_jaccard
    sx_mathoverflow_pa
    sx_mathoverflow_mlp
    sx_mathoverflow_node_emb
    sx_mathoverflow_gnn
    sx_mathoverflow_gnn_sum
    sx_mathoverflow_gnn_concat
    sx_mathoverflow_ground_truth
) do (
    echo [%date% %time%] Running %%c >> %LOGFILE%
    %PYTHON% -m src.online.loop --config configs/online/algo_sweep_sx_mathoverflow/%%c.yaml >> %LOGFILE% 2>&1
    echo [%date% %time%] Done %%c >> %LOGFILE%
)

echo [%date% %time%] All sx_mathoverflow experiments done >> %LOGFILE%
