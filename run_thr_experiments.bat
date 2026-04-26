@echo off
set PYTHON=C:\conda\envs\gnn\python.exe
set ROOT=C:\Users\12143\Desktop\pythonProject\gnn

cd /d %ROOT%

echo [%date% %time%] Starting THR experiments >> run_thr_experiments.log

for %%c in (
    college_msg_thr_random
    college_msg_thr_gnn
    bitcoin_otc_thr_random
    bitcoin_otc_thr_gnn
    bitcoin_alpha_thr_random
    bitcoin_alpha_thr_gnn
    email_eu_thr_random
    email_eu_thr_gnn
    dnc_email_thr_random
    dnc_email_thr_gnn
    sx_mathoverflow_thr_random
    sx_mathoverflow_thr_gnn
    sx_askubuntu_thr_random
    sx_askubuntu_thr_gnn
    sx_superuser_thr_random
    sx_superuser_thr_gnn
    epinions_thr_random
    epinions_thr_gnn
) do (
    echo [%date% %time%] Running %%c >> run_thr_experiments.log
    %PYTHON% -m src.online.loop --config configs/online/%%c.yaml >> results/online/%%c.log 2>&1
    echo [%date% %time%] Done %%c >> run_thr_experiments.log
)

echo [%date% %time%] All experiments finished >> run_thr_experiments.log
