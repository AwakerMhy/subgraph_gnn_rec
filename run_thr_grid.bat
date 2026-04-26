@echo off
setlocal enabledelayedexpansion
set PYTHON=C:\conda\envs\gnn\python.exe
set ROOT=C:\Users\12143\Desktop\pythonProject\gnn

cd /d %ROOT%

echo [%date% %time%] Starting THR grid experiments >> run_thr_grid.log

for %%k in (10 20) do (
  for %%r in (50 100 200) do (
    for %%d in (college_msg bitcoin_otc bitcoin_alpha email_eu dnc_email sx_mathoverflow sx_askubuntu sx_superuser epinions) do (
      for %%m in (random gnn) do (
        set CFG=configs\online\%%d_thr_%%m_k%%k_r%%r.yaml
        echo [%date% %time%] Running %%d_thr_%%m_k%%k_r%%r >> run_thr_grid.log
        %PYTHON% -m src.online.loop --config !CFG! >> results\online\%%d_thr_%%m_k%%k_r%%r.log 2>&1
        echo [%date% %time%] Done %%d_thr_%%m_k%%k_r%%r >> run_thr_grid.log
      )
    )
  )
)

echo [%date% %time%] All grid experiments finished >> run_thr_grid.log
