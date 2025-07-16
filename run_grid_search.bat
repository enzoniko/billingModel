@echo off
SETLOCAL EnableDelayedExpansion
echo Starting parallel grid search...
echo --- Running batch 1 --- 
START "RUN_0" python experiment5.py --results-dir grid_search_results\run_000_6c562d6c8f --dim-redux-target-variance 0.9 --hybrid-alpha 0.2 --post-process-dead-zone-k 2.5 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_1" python experiment5.py --results-dir grid_search_results\run_001_e964ab7a3d --dim-redux-target-variance 0.9 --hybrid-alpha 0.2 --post-process-dead-zone-k 2.5 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_2" python experiment5.py --results-dir grid_search_results\run_002_fc85b8e233 --dim-redux-target-variance 0.9 --hybrid-alpha 0.2 --post-process-dead-zone-k 2.5 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_3" python experiment5.py --results-dir grid_search_results\run_003_abbd189083 --dim-redux-target-variance 0.9 --hybrid-alpha 0.2 --post-process-dead-zone-k 2.5 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_4" python experiment5.py --results-dir grid_search_results\run_004_aac33df3a6 --dim-redux-target-variance 0.9 --hybrid-alpha 0.2 --post-process-dead-zone-k 2.5 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_5" python experiment5.py --results-dir grid_search_results\run_005_9d7550a32a --dim-redux-target-variance 0.9 --hybrid-alpha 0.2 --post-process-dead-zone-k 2.5 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_6" python experiment5.py --results-dir grid_search_results\run_006_74167e7e06 --dim-redux-target-variance 0.9 --hybrid-alpha 0.2 --post-process-dead-zone-k 2.5 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_7" python experiment5.py --results-dir grid_search_results\run_007_5abde49d5c --dim-redux-target-variance 0.9 --hybrid-alpha 0.2 --post-process-dead-zone-k 2.5 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_8" python experiment5.py --results-dir grid_search_results\run_008_dfdc18a238 --dim-redux-target-variance 0.9 --hybrid-alpha 0.2 --post-process-dead-zone-k 2.5 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_9" python experiment5.py --results-dir grid_search_results\run_009_1d779ab9c6 --dim-redux-target-variance 0.9 --hybrid-alpha 0.2 --post-process-dead-zone-k 3.0 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_10" python experiment5.py --results-dir grid_search_results\run_010_a23adb3989 --dim-redux-target-variance 0.9 --hybrid-alpha 0.2 --post-process-dead-zone-k 3.0 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_11" python experiment5.py --results-dir grid_search_results\run_011_8d5d1d1ede --dim-redux-target-variance 0.9 --hybrid-alpha 0.2 --post-process-dead-zone-k 3.0 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_12" python experiment5.py --results-dir grid_search_results\run_012_094c13abda --dim-redux-target-variance 0.9 --hybrid-alpha 0.2 --post-process-dead-zone-k 3.0 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_13" python experiment5.py --results-dir grid_search_results\run_013_4d784678fb --dim-redux-target-variance 0.9 --hybrid-alpha 0.2 --post-process-dead-zone-k 3.0 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_14" python experiment5.py --results-dir grid_search_results\run_014_049278b2cb --dim-redux-target-variance 0.9 --hybrid-alpha 0.2 --post-process-dead-zone-k 3.0 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_15" python experiment5.py --results-dir grid_search_results\run_015_453f434aa1 --dim-redux-target-variance 0.9 --hybrid-alpha 0.2 --post-process-dead-zone-k 3.0 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_16" python experiment5.py --results-dir grid_search_results\run_016_8645f23151 --dim-redux-target-variance 0.9 --hybrid-alpha 0.2 --post-process-dead-zone-k 3.0 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_17" python experiment5.py --results-dir grid_search_results\run_017_5466a0fa64 --dim-redux-target-variance 0.9 --hybrid-alpha 0.2 --post-process-dead-zone-k 3.0 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_18" python experiment5.py --results-dir grid_search_results\run_018_d7411244d6 --dim-redux-target-variance 0.9 --hybrid-alpha 0.2 --post-process-dead-zone-k 3.5 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_19" python experiment5.py --results-dir grid_search_results\run_019_0029435fe9 --dim-redux-target-variance 0.9 --hybrid-alpha 0.2 --post-process-dead-zone-k 3.5 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.8 --device cpu
echo Waiting for batch to complete...
:wait_loop_batch_1
set "completed_procs=0"
IF EXIST "grid_search_results\run_000_6c562d6c8f\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_001_e964ab7a3d\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_002_fc85b8e233\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_003_abbd189083\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_004_aac33df3a6\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_005_9d7550a32a\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_006_74167e7e06\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_007_5abde49d5c\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_008_dfdc18a238\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_009_1d779ab9c6\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_010_a23adb3989\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_011_8d5d1d1ede\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_012_094c13abda\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_013_4d784678fb\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_014_049278b2cb\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_015_453f434aa1\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_016_8645f23151\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_017_5466a0fa64\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_018_d7411244d6\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_019_0029435fe9\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
if !completed_procs! lss 20 (
    echo !completed_procs!/20 jobs in batch 1 complete. Waiting...
    TIMEOUT /T 5 /NOBREAK > nul
    goto wait_loop_batch_1
)
echo Batch complete.

echo --- Running batch 2 --- 
START "RUN_20" python experiment5.py --results-dir grid_search_results\run_020_0f7c7fad45 --dim-redux-target-variance 0.9 --hybrid-alpha 0.2 --post-process-dead-zone-k 3.5 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_21" python experiment5.py --results-dir grid_search_results\run_021_92f5787f06 --dim-redux-target-variance 0.9 --hybrid-alpha 0.2 --post-process-dead-zone-k 3.5 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_22" python experiment5.py --results-dir grid_search_results\run_022_44e477a120 --dim-redux-target-variance 0.9 --hybrid-alpha 0.2 --post-process-dead-zone-k 3.5 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_23" python experiment5.py --results-dir grid_search_results\run_023_481410c0c0 --dim-redux-target-variance 0.9 --hybrid-alpha 0.2 --post-process-dead-zone-k 3.5 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_24" python experiment5.py --results-dir grid_search_results\run_024_91e18002ff --dim-redux-target-variance 0.9 --hybrid-alpha 0.2 --post-process-dead-zone-k 3.5 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_25" python experiment5.py --results-dir grid_search_results\run_025_037db2eed4 --dim-redux-target-variance 0.9 --hybrid-alpha 0.2 --post-process-dead-zone-k 3.5 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_26" python experiment5.py --results-dir grid_search_results\run_026_bbd31aee0d --dim-redux-target-variance 0.9 --hybrid-alpha 0.2 --post-process-dead-zone-k 3.5 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_27" python experiment5.py --results-dir grid_search_results\run_027_5292a631d3 --dim-redux-target-variance 0.9 --hybrid-alpha 0.3 --post-process-dead-zone-k 2.5 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_28" python experiment5.py --results-dir grid_search_results\run_028_742893f5bc --dim-redux-target-variance 0.9 --hybrid-alpha 0.3 --post-process-dead-zone-k 2.5 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_29" python experiment5.py --results-dir grid_search_results\run_029_7821d3777b --dim-redux-target-variance 0.9 --hybrid-alpha 0.3 --post-process-dead-zone-k 2.5 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_30" python experiment5.py --results-dir grid_search_results\run_030_67b39554ad --dim-redux-target-variance 0.9 --hybrid-alpha 0.3 --post-process-dead-zone-k 2.5 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_31" python experiment5.py --results-dir grid_search_results\run_031_98d842561d --dim-redux-target-variance 0.9 --hybrid-alpha 0.3 --post-process-dead-zone-k 2.5 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_32" python experiment5.py --results-dir grid_search_results\run_032_beaee69634 --dim-redux-target-variance 0.9 --hybrid-alpha 0.3 --post-process-dead-zone-k 2.5 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_33" python experiment5.py --results-dir grid_search_results\run_033_b9ac06a1af --dim-redux-target-variance 0.9 --hybrid-alpha 0.3 --post-process-dead-zone-k 2.5 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_34" python experiment5.py --results-dir grid_search_results\run_034_0def1deeb0 --dim-redux-target-variance 0.9 --hybrid-alpha 0.3 --post-process-dead-zone-k 2.5 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_35" python experiment5.py --results-dir grid_search_results\run_035_034cf04fe2 --dim-redux-target-variance 0.9 --hybrid-alpha 0.3 --post-process-dead-zone-k 2.5 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_36" python experiment5.py --results-dir grid_search_results\run_036_765b8b89dc --dim-redux-target-variance 0.9 --hybrid-alpha 0.3 --post-process-dead-zone-k 3.0 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_37" python experiment5.py --results-dir grid_search_results\run_037_6c6994f891 --dim-redux-target-variance 0.9 --hybrid-alpha 0.3 --post-process-dead-zone-k 3.0 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_38" python experiment5.py --results-dir grid_search_results\run_038_6bfc5929de --dim-redux-target-variance 0.9 --hybrid-alpha 0.3 --post-process-dead-zone-k 3.0 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_39" python experiment5.py --results-dir grid_search_results\run_039_d3f5b1ad91 --dim-redux-target-variance 0.9 --hybrid-alpha 0.3 --post-process-dead-zone-k 3.0 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.7 --device cpu
echo Waiting for batch to complete...
:wait_loop_batch_2
set "completed_procs=0"
IF EXIST "grid_search_results\run_020_0f7c7fad45\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_021_92f5787f06\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_022_44e477a120\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_023_481410c0c0\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_024_91e18002ff\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_025_037db2eed4\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_026_bbd31aee0d\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_027_5292a631d3\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_028_742893f5bc\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_029_7821d3777b\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_030_67b39554ad\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_031_98d842561d\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_032_beaee69634\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_033_b9ac06a1af\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_034_0def1deeb0\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_035_034cf04fe2\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_036_765b8b89dc\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_037_6c6994f891\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_038_6bfc5929de\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_039_d3f5b1ad91\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
if !completed_procs! lss 20 (
    echo !completed_procs!/20 jobs in batch 2 complete. Waiting...
    TIMEOUT /T 5 /NOBREAK > nul
    goto wait_loop_batch_2
)
echo Batch complete.

echo --- Running batch 3 --- 
START "RUN_40" python experiment5.py --results-dir grid_search_results\run_040_3149a11527 --dim-redux-target-variance 0.9 --hybrid-alpha 0.3 --post-process-dead-zone-k 3.0 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_41" python experiment5.py --results-dir grid_search_results\run_041_fe06be9b08 --dim-redux-target-variance 0.9 --hybrid-alpha 0.3 --post-process-dead-zone-k 3.0 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_42" python experiment5.py --results-dir grid_search_results\run_042_3a052dd368 --dim-redux-target-variance 0.9 --hybrid-alpha 0.3 --post-process-dead-zone-k 3.0 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_43" python experiment5.py --results-dir grid_search_results\run_043_5f5ad47f66 --dim-redux-target-variance 0.9 --hybrid-alpha 0.3 --post-process-dead-zone-k 3.0 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_44" python experiment5.py --results-dir grid_search_results\run_044_fff89317fd --dim-redux-target-variance 0.9 --hybrid-alpha 0.3 --post-process-dead-zone-k 3.0 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_45" python experiment5.py --results-dir grid_search_results\run_045_b9bd7c222f --dim-redux-target-variance 0.9 --hybrid-alpha 0.3 --post-process-dead-zone-k 3.5 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_46" python experiment5.py --results-dir grid_search_results\run_046_e06ebc6df5 --dim-redux-target-variance 0.9 --hybrid-alpha 0.3 --post-process-dead-zone-k 3.5 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_47" python experiment5.py --results-dir grid_search_results\run_047_9b753ea54d --dim-redux-target-variance 0.9 --hybrid-alpha 0.3 --post-process-dead-zone-k 3.5 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_48" python experiment5.py --results-dir grid_search_results\run_048_8a42e814f4 --dim-redux-target-variance 0.9 --hybrid-alpha 0.3 --post-process-dead-zone-k 3.5 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_49" python experiment5.py --results-dir grid_search_results\run_049_17b2b2e989 --dim-redux-target-variance 0.9 --hybrid-alpha 0.3 --post-process-dead-zone-k 3.5 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_50" python experiment5.py --results-dir grid_search_results\run_050_38acc00ef6 --dim-redux-target-variance 0.9 --hybrid-alpha 0.3 --post-process-dead-zone-k 3.5 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_51" python experiment5.py --results-dir grid_search_results\run_051_a40b8bc68e --dim-redux-target-variance 0.9 --hybrid-alpha 0.3 --post-process-dead-zone-k 3.5 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_52" python experiment5.py --results-dir grid_search_results\run_052_86218ad13a --dim-redux-target-variance 0.9 --hybrid-alpha 0.3 --post-process-dead-zone-k 3.5 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_53" python experiment5.py --results-dir grid_search_results\run_053_7a752709c8 --dim-redux-target-variance 0.9 --hybrid-alpha 0.3 --post-process-dead-zone-k 3.5 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_54" python experiment5.py --results-dir grid_search_results\run_054_6c018eb751 --dim-redux-target-variance 0.9 --hybrid-alpha 0.4 --post-process-dead-zone-k 2.5 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_55" python experiment5.py --results-dir grid_search_results\run_055_afb78a0c3f --dim-redux-target-variance 0.9 --hybrid-alpha 0.4 --post-process-dead-zone-k 2.5 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_56" python experiment5.py --results-dir grid_search_results\run_056_f56404bd6d --dim-redux-target-variance 0.9 --hybrid-alpha 0.4 --post-process-dead-zone-k 2.5 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_57" python experiment5.py --results-dir grid_search_results\run_057_b0f717c125 --dim-redux-target-variance 0.9 --hybrid-alpha 0.4 --post-process-dead-zone-k 2.5 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_58" python experiment5.py --results-dir grid_search_results\run_058_24425f1f22 --dim-redux-target-variance 0.9 --hybrid-alpha 0.4 --post-process-dead-zone-k 2.5 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_59" python experiment5.py --results-dir grid_search_results\run_059_99cc654f5d --dim-redux-target-variance 0.9 --hybrid-alpha 0.4 --post-process-dead-zone-k 2.5 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.9 --device cpu
echo Waiting for batch to complete...
:wait_loop_batch_3
set "completed_procs=0"
IF EXIST "grid_search_results\run_040_3149a11527\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_041_fe06be9b08\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_042_3a052dd368\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_043_5f5ad47f66\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_044_fff89317fd\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_045_b9bd7c222f\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_046_e06ebc6df5\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_047_9b753ea54d\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_048_8a42e814f4\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_049_17b2b2e989\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_050_38acc00ef6\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_051_a40b8bc68e\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_052_86218ad13a\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_053_7a752709c8\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_054_6c018eb751\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_055_afb78a0c3f\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_056_f56404bd6d\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_057_b0f717c125\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_058_24425f1f22\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_059_99cc654f5d\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
if !completed_procs! lss 20 (
    echo !completed_procs!/20 jobs in batch 3 complete. Waiting...
    TIMEOUT /T 5 /NOBREAK > nul
    goto wait_loop_batch_3
)
echo Batch complete.

echo --- Running batch 4 --- 
START "RUN_60" python experiment5.py --results-dir grid_search_results\run_060_9e1cfc322c --dim-redux-target-variance 0.9 --hybrid-alpha 0.4 --post-process-dead-zone-k 2.5 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_61" python experiment5.py --results-dir grid_search_results\run_061_853c448c3f --dim-redux-target-variance 0.9 --hybrid-alpha 0.4 --post-process-dead-zone-k 2.5 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_62" python experiment5.py --results-dir grid_search_results\run_062_3b9edab1e9 --dim-redux-target-variance 0.9 --hybrid-alpha 0.4 --post-process-dead-zone-k 2.5 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_63" python experiment5.py --results-dir grid_search_results\run_063_4d71bf0215 --dim-redux-target-variance 0.9 --hybrid-alpha 0.4 --post-process-dead-zone-k 3.0 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_64" python experiment5.py --results-dir grid_search_results\run_064_d4bfe76f7f --dim-redux-target-variance 0.9 --hybrid-alpha 0.4 --post-process-dead-zone-k 3.0 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_65" python experiment5.py --results-dir grid_search_results\run_065_d41f28f989 --dim-redux-target-variance 0.9 --hybrid-alpha 0.4 --post-process-dead-zone-k 3.0 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_66" python experiment5.py --results-dir grid_search_results\run_066_f255c67e5d --dim-redux-target-variance 0.9 --hybrid-alpha 0.4 --post-process-dead-zone-k 3.0 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_67" python experiment5.py --results-dir grid_search_results\run_067_891b18b842 --dim-redux-target-variance 0.9 --hybrid-alpha 0.4 --post-process-dead-zone-k 3.0 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_68" python experiment5.py --results-dir grid_search_results\run_068_4ad0dd169c --dim-redux-target-variance 0.9 --hybrid-alpha 0.4 --post-process-dead-zone-k 3.0 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_69" python experiment5.py --results-dir grid_search_results\run_069_9843ff0d46 --dim-redux-target-variance 0.9 --hybrid-alpha 0.4 --post-process-dead-zone-k 3.0 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_70" python experiment5.py --results-dir grid_search_results\run_070_0f0c83f984 --dim-redux-target-variance 0.9 --hybrid-alpha 0.4 --post-process-dead-zone-k 3.0 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_71" python experiment5.py --results-dir grid_search_results\run_071_6151e7eb1a --dim-redux-target-variance 0.9 --hybrid-alpha 0.4 --post-process-dead-zone-k 3.0 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_72" python experiment5.py --results-dir grid_search_results\run_072_9a5e37afa0 --dim-redux-target-variance 0.9 --hybrid-alpha 0.4 --post-process-dead-zone-k 3.5 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_73" python experiment5.py --results-dir grid_search_results\run_073_beaf97b441 --dim-redux-target-variance 0.9 --hybrid-alpha 0.4 --post-process-dead-zone-k 3.5 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_74" python experiment5.py --results-dir grid_search_results\run_074_979fe540a1 --dim-redux-target-variance 0.9 --hybrid-alpha 0.4 --post-process-dead-zone-k 3.5 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_75" python experiment5.py --results-dir grid_search_results\run_075_8d8cf04059 --dim-redux-target-variance 0.9 --hybrid-alpha 0.4 --post-process-dead-zone-k 3.5 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_76" python experiment5.py --results-dir grid_search_results\run_076_47c20c9d12 --dim-redux-target-variance 0.9 --hybrid-alpha 0.4 --post-process-dead-zone-k 3.5 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_77" python experiment5.py --results-dir grid_search_results\run_077_f738c7d042 --dim-redux-target-variance 0.9 --hybrid-alpha 0.4 --post-process-dead-zone-k 3.5 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_78" python experiment5.py --results-dir grid_search_results\run_078_b830078cef --dim-redux-target-variance 0.9 --hybrid-alpha 0.4 --post-process-dead-zone-k 3.5 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_79" python experiment5.py --results-dir grid_search_results\run_079_e3498a915a --dim-redux-target-variance 0.9 --hybrid-alpha 0.4 --post-process-dead-zone-k 3.5 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.8 --device cpu
echo Waiting for batch to complete...
:wait_loop_batch_4
set "completed_procs=0"
IF EXIST "grid_search_results\run_060_9e1cfc322c\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_061_853c448c3f\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_062_3b9edab1e9\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_063_4d71bf0215\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_064_d4bfe76f7f\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_065_d41f28f989\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_066_f255c67e5d\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_067_891b18b842\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_068_4ad0dd169c\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_069_9843ff0d46\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_070_0f0c83f984\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_071_6151e7eb1a\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_072_9a5e37afa0\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_073_beaf97b441\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_074_979fe540a1\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_075_8d8cf04059\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_076_47c20c9d12\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_077_f738c7d042\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_078_b830078cef\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_079_e3498a915a\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
if !completed_procs! lss 20 (
    echo !completed_procs!/20 jobs in batch 4 complete. Waiting...
    TIMEOUT /T 5 /NOBREAK > nul
    goto wait_loop_batch_4
)
echo Batch complete.

echo --- Running batch 5 --- 
START "RUN_80" python experiment5.py --results-dir grid_search_results\run_080_c5a8b4ba79 --dim-redux-target-variance 0.9 --hybrid-alpha 0.4 --post-process-dead-zone-k 3.5 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_81" python experiment5.py --results-dir grid_search_results\run_081_6fdb4c7c01 --dim-redux-target-variance 0.95 --hybrid-alpha 0.2 --post-process-dead-zone-k 2.5 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_82" python experiment5.py --results-dir grid_search_results\run_082_5aef9fccf1 --dim-redux-target-variance 0.95 --hybrid-alpha 0.2 --post-process-dead-zone-k 2.5 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_83" python experiment5.py --results-dir grid_search_results\run_083_51e81d54ea --dim-redux-target-variance 0.95 --hybrid-alpha 0.2 --post-process-dead-zone-k 2.5 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_84" python experiment5.py --results-dir grid_search_results\run_084_810b33a1c8 --dim-redux-target-variance 0.95 --hybrid-alpha 0.2 --post-process-dead-zone-k 2.5 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_85" python experiment5.py --results-dir grid_search_results\run_085_fce0fc88a3 --dim-redux-target-variance 0.95 --hybrid-alpha 0.2 --post-process-dead-zone-k 2.5 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_86" python experiment5.py --results-dir grid_search_results\run_086_eca8c5888d --dim-redux-target-variance 0.95 --hybrid-alpha 0.2 --post-process-dead-zone-k 2.5 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_87" python experiment5.py --results-dir grid_search_results\run_087_fdd57caef0 --dim-redux-target-variance 0.95 --hybrid-alpha 0.2 --post-process-dead-zone-k 2.5 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_88" python experiment5.py --results-dir grid_search_results\run_088_7be19a8ce3 --dim-redux-target-variance 0.95 --hybrid-alpha 0.2 --post-process-dead-zone-k 2.5 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_89" python experiment5.py --results-dir grid_search_results\run_089_e95285e6fa --dim-redux-target-variance 0.95 --hybrid-alpha 0.2 --post-process-dead-zone-k 2.5 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_90" python experiment5.py --results-dir grid_search_results\run_090_75de962a45 --dim-redux-target-variance 0.95 --hybrid-alpha 0.2 --post-process-dead-zone-k 3.0 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_91" python experiment5.py --results-dir grid_search_results\run_091_ae1447c92e --dim-redux-target-variance 0.95 --hybrid-alpha 0.2 --post-process-dead-zone-k 3.0 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_92" python experiment5.py --results-dir grid_search_results\run_092_02d91dd577 --dim-redux-target-variance 0.95 --hybrid-alpha 0.2 --post-process-dead-zone-k 3.0 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_93" python experiment5.py --results-dir grid_search_results\run_093_0100fb5fc0 --dim-redux-target-variance 0.95 --hybrid-alpha 0.2 --post-process-dead-zone-k 3.0 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_94" python experiment5.py --results-dir grid_search_results\run_094_7178cedda4 --dim-redux-target-variance 0.95 --hybrid-alpha 0.2 --post-process-dead-zone-k 3.0 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_95" python experiment5.py --results-dir grid_search_results\run_095_97d75e1b03 --dim-redux-target-variance 0.95 --hybrid-alpha 0.2 --post-process-dead-zone-k 3.0 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_96" python experiment5.py --results-dir grid_search_results\run_096_19250d290a --dim-redux-target-variance 0.95 --hybrid-alpha 0.2 --post-process-dead-zone-k 3.0 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_97" python experiment5.py --results-dir grid_search_results\run_097_10cf0b451f --dim-redux-target-variance 0.95 --hybrid-alpha 0.2 --post-process-dead-zone-k 3.0 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_98" python experiment5.py --results-dir grid_search_results\run_098_42eca72fe7 --dim-redux-target-variance 0.95 --hybrid-alpha 0.2 --post-process-dead-zone-k 3.0 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_99" python experiment5.py --results-dir grid_search_results\run_099_eab97f86fe --dim-redux-target-variance 0.95 --hybrid-alpha 0.2 --post-process-dead-zone-k 3.5 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.7 --device cpu
echo Waiting for batch to complete...
:wait_loop_batch_5
set "completed_procs=0"
IF EXIST "grid_search_results\run_080_c5a8b4ba79\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_081_6fdb4c7c01\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_082_5aef9fccf1\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_083_51e81d54ea\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_084_810b33a1c8\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_085_fce0fc88a3\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_086_eca8c5888d\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_087_fdd57caef0\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_088_7be19a8ce3\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_089_e95285e6fa\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_090_75de962a45\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_091_ae1447c92e\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_092_02d91dd577\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_093_0100fb5fc0\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_094_7178cedda4\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_095_97d75e1b03\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_096_19250d290a\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_097_10cf0b451f\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_098_42eca72fe7\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_099_eab97f86fe\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
if !completed_procs! lss 20 (
    echo !completed_procs!/20 jobs in batch 5 complete. Waiting...
    TIMEOUT /T 5 /NOBREAK > nul
    goto wait_loop_batch_5
)
echo Batch complete.

echo --- Running batch 6 --- 
START "RUN_100" python experiment5.py --results-dir grid_search_results\run_100_0a9223835a --dim-redux-target-variance 0.95 --hybrid-alpha 0.2 --post-process-dead-zone-k 3.5 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_101" python experiment5.py --results-dir grid_search_results\run_101_e08a0dd2f9 --dim-redux-target-variance 0.95 --hybrid-alpha 0.2 --post-process-dead-zone-k 3.5 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_102" python experiment5.py --results-dir grid_search_results\run_102_141cb106f0 --dim-redux-target-variance 0.95 --hybrid-alpha 0.2 --post-process-dead-zone-k 3.5 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_103" python experiment5.py --results-dir grid_search_results\run_103_e50d84492e --dim-redux-target-variance 0.95 --hybrid-alpha 0.2 --post-process-dead-zone-k 3.5 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_104" python experiment5.py --results-dir grid_search_results\run_104_33e811a1c3 --dim-redux-target-variance 0.95 --hybrid-alpha 0.2 --post-process-dead-zone-k 3.5 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_105" python experiment5.py --results-dir grid_search_results\run_105_ad2de7f361 --dim-redux-target-variance 0.95 --hybrid-alpha 0.2 --post-process-dead-zone-k 3.5 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_106" python experiment5.py --results-dir grid_search_results\run_106_00dddf8e32 --dim-redux-target-variance 0.95 --hybrid-alpha 0.2 --post-process-dead-zone-k 3.5 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_107" python experiment5.py --results-dir grid_search_results\run_107_b483bb63e9 --dim-redux-target-variance 0.95 --hybrid-alpha 0.2 --post-process-dead-zone-k 3.5 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_108" python experiment5.py --results-dir grid_search_results\run_108_d9096f3e15 --dim-redux-target-variance 0.95 --hybrid-alpha 0.3 --post-process-dead-zone-k 2.5 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_109" python experiment5.py --results-dir grid_search_results\run_109_32db77d805 --dim-redux-target-variance 0.95 --hybrid-alpha 0.3 --post-process-dead-zone-k 2.5 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_110" python experiment5.py --results-dir grid_search_results\run_110_87d0b38ab2 --dim-redux-target-variance 0.95 --hybrid-alpha 0.3 --post-process-dead-zone-k 2.5 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_111" python experiment5.py --results-dir grid_search_results\run_111_d289afa03e --dim-redux-target-variance 0.95 --hybrid-alpha 0.3 --post-process-dead-zone-k 2.5 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_112" python experiment5.py --results-dir grid_search_results\run_112_323080d0ff --dim-redux-target-variance 0.95 --hybrid-alpha 0.3 --post-process-dead-zone-k 2.5 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_113" python experiment5.py --results-dir grid_search_results\run_113_141749ab66 --dim-redux-target-variance 0.95 --hybrid-alpha 0.3 --post-process-dead-zone-k 2.5 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_114" python experiment5.py --results-dir grid_search_results\run_114_ae6aa722e1 --dim-redux-target-variance 0.95 --hybrid-alpha 0.3 --post-process-dead-zone-k 2.5 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_115" python experiment5.py --results-dir grid_search_results\run_115_7e7f96855e --dim-redux-target-variance 0.95 --hybrid-alpha 0.3 --post-process-dead-zone-k 2.5 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_116" python experiment5.py --results-dir grid_search_results\run_116_11bbe4663a --dim-redux-target-variance 0.95 --hybrid-alpha 0.3 --post-process-dead-zone-k 2.5 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_117" python experiment5.py --results-dir grid_search_results\run_117_6227d7e99d --dim-redux-target-variance 0.95 --hybrid-alpha 0.3 --post-process-dead-zone-k 3.0 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_118" python experiment5.py --results-dir grid_search_results\run_118_0344d5ccdf --dim-redux-target-variance 0.95 --hybrid-alpha 0.3 --post-process-dead-zone-k 3.0 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_119" python experiment5.py --results-dir grid_search_results\run_119_558469fae7 --dim-redux-target-variance 0.95 --hybrid-alpha 0.3 --post-process-dead-zone-k 3.0 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.9 --device cpu
echo Waiting for batch to complete...
:wait_loop_batch_6
set "completed_procs=0"
IF EXIST "grid_search_results\run_100_0a9223835a\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_101_e08a0dd2f9\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_102_141cb106f0\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_103_e50d84492e\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_104_33e811a1c3\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_105_ad2de7f361\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_106_00dddf8e32\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_107_b483bb63e9\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_108_d9096f3e15\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_109_32db77d805\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_110_87d0b38ab2\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_111_d289afa03e\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_112_323080d0ff\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_113_141749ab66\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_114_ae6aa722e1\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_115_7e7f96855e\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_116_11bbe4663a\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_117_6227d7e99d\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_118_0344d5ccdf\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_119_558469fae7\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
if !completed_procs! lss 20 (
    echo !completed_procs!/20 jobs in batch 6 complete. Waiting...
    TIMEOUT /T 5 /NOBREAK > nul
    goto wait_loop_batch_6
)
echo Batch complete.

echo --- Running batch 7 --- 
START "RUN_120" python experiment5.py --results-dir grid_search_results\run_120_83dfc8305a --dim-redux-target-variance 0.95 --hybrid-alpha 0.3 --post-process-dead-zone-k 3.0 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_121" python experiment5.py --results-dir grid_search_results\run_121_56cd2f8721 --dim-redux-target-variance 0.95 --hybrid-alpha 0.3 --post-process-dead-zone-k 3.0 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_122" python experiment5.py --results-dir grid_search_results\run_122_eec1c9938b --dim-redux-target-variance 0.95 --hybrid-alpha 0.3 --post-process-dead-zone-k 3.0 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_123" python experiment5.py --results-dir grid_search_results\run_123_90b2c5fa2b --dim-redux-target-variance 0.95 --hybrid-alpha 0.3 --post-process-dead-zone-k 3.0 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_124" python experiment5.py --results-dir grid_search_results\run_124_3d237ce2ca --dim-redux-target-variance 0.95 --hybrid-alpha 0.3 --post-process-dead-zone-k 3.0 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_125" python experiment5.py --results-dir grid_search_results\run_125_276ad3677f --dim-redux-target-variance 0.95 --hybrid-alpha 0.3 --post-process-dead-zone-k 3.0 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_126" python experiment5.py --results-dir grid_search_results\run_126_72d79e42f9 --dim-redux-target-variance 0.95 --hybrid-alpha 0.3 --post-process-dead-zone-k 3.5 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_127" python experiment5.py --results-dir grid_search_results\run_127_de1342a70c --dim-redux-target-variance 0.95 --hybrid-alpha 0.3 --post-process-dead-zone-k 3.5 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_128" python experiment5.py --results-dir grid_search_results\run_128_56896ce928 --dim-redux-target-variance 0.95 --hybrid-alpha 0.3 --post-process-dead-zone-k 3.5 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_129" python experiment5.py --results-dir grid_search_results\run_129_1fe7f207d9 --dim-redux-target-variance 0.95 --hybrid-alpha 0.3 --post-process-dead-zone-k 3.5 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_130" python experiment5.py --results-dir grid_search_results\run_130_2f27f08746 --dim-redux-target-variance 0.95 --hybrid-alpha 0.3 --post-process-dead-zone-k 3.5 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_131" python experiment5.py --results-dir grid_search_results\run_131_a0d7458c30 --dim-redux-target-variance 0.95 --hybrid-alpha 0.3 --post-process-dead-zone-k 3.5 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_132" python experiment5.py --results-dir grid_search_results\run_132_8ad772a956 --dim-redux-target-variance 0.95 --hybrid-alpha 0.3 --post-process-dead-zone-k 3.5 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_133" python experiment5.py --results-dir grid_search_results\run_133_de57e8338e --dim-redux-target-variance 0.95 --hybrid-alpha 0.3 --post-process-dead-zone-k 3.5 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_134" python experiment5.py --results-dir grid_search_results\run_134_dfba1b6df4 --dim-redux-target-variance 0.95 --hybrid-alpha 0.3 --post-process-dead-zone-k 3.5 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_135" python experiment5.py --results-dir grid_search_results\run_135_d22344faa1 --dim-redux-target-variance 0.95 --hybrid-alpha 0.4 --post-process-dead-zone-k 2.5 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_136" python experiment5.py --results-dir grid_search_results\run_136_67216e5081 --dim-redux-target-variance 0.95 --hybrid-alpha 0.4 --post-process-dead-zone-k 2.5 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_137" python experiment5.py --results-dir grid_search_results\run_137_5be31f796a --dim-redux-target-variance 0.95 --hybrid-alpha 0.4 --post-process-dead-zone-k 2.5 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_138" python experiment5.py --results-dir grid_search_results\run_138_3352d23b67 --dim-redux-target-variance 0.95 --hybrid-alpha 0.4 --post-process-dead-zone-k 2.5 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_139" python experiment5.py --results-dir grid_search_results\run_139_176667aaee --dim-redux-target-variance 0.95 --hybrid-alpha 0.4 --post-process-dead-zone-k 2.5 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.8 --device cpu
echo Waiting for batch to complete...
:wait_loop_batch_7
set "completed_procs=0"
IF EXIST "grid_search_results\run_120_83dfc8305a\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_121_56cd2f8721\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_122_eec1c9938b\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_123_90b2c5fa2b\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_124_3d237ce2ca\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_125_276ad3677f\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_126_72d79e42f9\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_127_de1342a70c\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_128_56896ce928\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_129_1fe7f207d9\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_130_2f27f08746\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_131_a0d7458c30\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_132_8ad772a956\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_133_de57e8338e\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_134_dfba1b6df4\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_135_d22344faa1\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_136_67216e5081\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_137_5be31f796a\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_138_3352d23b67\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_139_176667aaee\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
if !completed_procs! lss 20 (
    echo !completed_procs!/20 jobs in batch 7 complete. Waiting...
    TIMEOUT /T 5 /NOBREAK > nul
    goto wait_loop_batch_7
)
echo Batch complete.

echo --- Running batch 8 --- 
START "RUN_140" python experiment5.py --results-dir grid_search_results\run_140_1d56137b60 --dim-redux-target-variance 0.95 --hybrid-alpha 0.4 --post-process-dead-zone-k 2.5 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_141" python experiment5.py --results-dir grid_search_results\run_141_4c7dec4365 --dim-redux-target-variance 0.95 --hybrid-alpha 0.4 --post-process-dead-zone-k 2.5 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_142" python experiment5.py --results-dir grid_search_results\run_142_3cf5756c2f --dim-redux-target-variance 0.95 --hybrid-alpha 0.4 --post-process-dead-zone-k 2.5 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_143" python experiment5.py --results-dir grid_search_results\run_143_08d51fe84f --dim-redux-target-variance 0.95 --hybrid-alpha 0.4 --post-process-dead-zone-k 2.5 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_144" python experiment5.py --results-dir grid_search_results\run_144_942bff34d5 --dim-redux-target-variance 0.95 --hybrid-alpha 0.4 --post-process-dead-zone-k 3.0 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_145" python experiment5.py --results-dir grid_search_results\run_145_0da63b3e99 --dim-redux-target-variance 0.95 --hybrid-alpha 0.4 --post-process-dead-zone-k 3.0 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_146" python experiment5.py --results-dir grid_search_results\run_146_4ea9d96552 --dim-redux-target-variance 0.95 --hybrid-alpha 0.4 --post-process-dead-zone-k 3.0 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_147" python experiment5.py --results-dir grid_search_results\run_147_b0d1741edd --dim-redux-target-variance 0.95 --hybrid-alpha 0.4 --post-process-dead-zone-k 3.0 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_148" python experiment5.py --results-dir grid_search_results\run_148_f269387731 --dim-redux-target-variance 0.95 --hybrid-alpha 0.4 --post-process-dead-zone-k 3.0 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_149" python experiment5.py --results-dir grid_search_results\run_149_b6e38ac3e9 --dim-redux-target-variance 0.95 --hybrid-alpha 0.4 --post-process-dead-zone-k 3.0 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_150" python experiment5.py --results-dir grid_search_results\run_150_948d59201f --dim-redux-target-variance 0.95 --hybrid-alpha 0.4 --post-process-dead-zone-k 3.0 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_151" python experiment5.py --results-dir grid_search_results\run_151_6a11e2a05d --dim-redux-target-variance 0.95 --hybrid-alpha 0.4 --post-process-dead-zone-k 3.0 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_152" python experiment5.py --results-dir grid_search_results\run_152_0c44e20521 --dim-redux-target-variance 0.95 --hybrid-alpha 0.4 --post-process-dead-zone-k 3.0 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_153" python experiment5.py --results-dir grid_search_results\run_153_4d6c50c50a --dim-redux-target-variance 0.95 --hybrid-alpha 0.4 --post-process-dead-zone-k 3.5 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_154" python experiment5.py --results-dir grid_search_results\run_154_46d44c15af --dim-redux-target-variance 0.95 --hybrid-alpha 0.4 --post-process-dead-zone-k 3.5 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_155" python experiment5.py --results-dir grid_search_results\run_155_7d056dd8d7 --dim-redux-target-variance 0.95 --hybrid-alpha 0.4 --post-process-dead-zone-k 3.5 --post-process-gamma 0.5 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_156" python experiment5.py --results-dir grid_search_results\run_156_9af2c3f18b --dim-redux-target-variance 0.95 --hybrid-alpha 0.4 --post-process-dead-zone-k 3.5 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.7 --device cpu
START "RUN_157" python experiment5.py --results-dir grid_search_results\run_157_91bfc70690 --dim-redux-target-variance 0.95 --hybrid-alpha 0.4 --post-process-dead-zone-k 3.5 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_158" python experiment5.py --results-dir grid_search_results\run_158_5ea53278fa --dim-redux-target-variance 0.95 --hybrid-alpha 0.4 --post-process-dead-zone-k 3.5 --post-process-gamma 0.6 --pricing-strategy-threshold-distance 0.9 --device cpu
START "RUN_159" python experiment5.py --results-dir grid_search_results\run_159_220d686017 --dim-redux-target-variance 0.95 --hybrid-alpha 0.4 --post-process-dead-zone-k 3.5 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.7 --device cpu
echo Waiting for batch to complete...
:wait_loop_batch_8
set "completed_procs=0"
IF EXIST "grid_search_results\run_140_1d56137b60\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_141_4c7dec4365\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_142_3cf5756c2f\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_143_08d51fe84f\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_144_942bff34d5\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_145_0da63b3e99\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_146_4ea9d96552\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_147_b0d1741edd\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_148_f269387731\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_149_b6e38ac3e9\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_150_948d59201f\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_151_6a11e2a05d\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_152_0c44e20521\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_153_4d6c50c50a\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_154_46d44c15af\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_155_7d056dd8d7\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_156_9af2c3f18b\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_157_91bfc70690\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_158_5ea53278fa\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_159_220d686017\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
if !completed_procs! lss 20 (
    echo !completed_procs!/20 jobs in batch 8 complete. Waiting...
    TIMEOUT /T 5 /NOBREAK > nul
    goto wait_loop_batch_8
)
echo Batch complete.

echo --- Running batch 9 --- 
START "RUN_160" python experiment5.py --results-dir grid_search_results\run_160_f527331c61 --dim-redux-target-variance 0.95 --hybrid-alpha 0.4 --post-process-dead-zone-k 3.5 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.8 --device cpu
START "RUN_161" python experiment5.py --results-dir grid_search_results\run_161_6e00992c88 --dim-redux-target-variance 0.95 --hybrid-alpha 0.4 --post-process-dead-zone-k 3.5 --post-process-gamma 0.7 --pricing-strategy-threshold-distance 0.9 --device cpu
echo Waiting for batch to complete...
:wait_loop_batch_9
set "completed_procs=0"
IF EXIST "grid_search_results\run_160_f527331c61\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
IF EXIST "grid_search_results\run_161_6e00992c88\summary_evaluation_by_group_experiment5.csv" (set /a completed_procs+=1)
if !completed_procs! lss 2 (
    echo !completed_procs!/2 jobs in batch 9 complete. Waiting...
    TIMEOUT /T 5 /NOBREAK > nul
    goto wait_loop_batch_9
)
echo Batch complete.

echo Grid search finished.
pause
