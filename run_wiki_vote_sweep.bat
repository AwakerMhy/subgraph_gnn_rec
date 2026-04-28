@echo off
set PYTHON=C:\conda\envs\gnn\python.exe
set ROOT=C:\Users\12143\Desktop\pythonProject\gnn
set LOGFILE=%ROOT%\run_wiki_vote_sweep.log

cd /d %ROOT%

echo [%date% %time%] Starting wiki_vote algo_sweep >> %LOGFILE%

for %%c in (
    wiki_vote_random
    wiki_vote_cn
    wiki_vote_aa
    wiki_vote_jaccard
    wiki_vote_pa
    wiki_vote_node_emb
    wiki_vote_gnn_sum
    wiki_vote_ground_truth
) do (
    echo [%date% %time%] Running %%c >> %LOGFILE%
    %PYTHON% -m src.online.loop --config configs/online/algo_sweep_wiki_vote/%%c.yaml >> %LOGFILE% 2>&1
    echo [%date% %time%] Done %%c >> %LOGFILE%
)

echo [%date% %time%] All wiki_vote experiments done >> %LOGFILE%
