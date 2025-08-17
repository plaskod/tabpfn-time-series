import json
from pathlib import Path

"""
LOOP_SEATTLE 
INFO:__main__:Evaluating 1/3 dataset LOOP_SEATTLE/5T
INFO:__main__:Dataset size: 6460
INFO:__main__:Dataset freq: 5T
INFO:__main__:Dataset term: short
INFO:__main__:Dataset prediction length: 48
INFO:__main__:Dataset target dim: 1
DEBUG:gift_eval.tabpfn_ts_wrapper:len(test_data_input): 6460
DEBUG:gift_eval.tabpfn_ts_wrapper:Processing batch of size: 1024
INFO:gift_eval.tabpfn_ts_wrapper:Slicing train_tsdf to 4096 timesteps for each time series

INFO:__main__:Evaluating 2/3 dataset LOOP_SEATTLE/5T
INFO:__main__:Dataset size: 6460
INFO:__main__:Dataset freq: 5T
INFO:__main__:Dataset term: medium
INFO:__main__:Dataset prediction length: 480
INFO:__main__:Dataset target dim: 1
DEBUG:gift_eval.tabpfn_ts_wrapper:len(test_data_input): 6460
DEBUG:gift_eval.tabpfn_ts_wrapper:Processing batch of size: 1024
INFO:gift_eval.tabpfn_ts_wrapper:Slicing train_tsdf to 4096 timesteps for each time series


M_DENSE 


SZ_TAXI 


bitbrains_fast_storage 


bitbrains_rnd 


bizitobs_application 


bizitobs_l2c 


bizitobs_service
wandb:                                                                                
wandb: 
wandb: Run history:
wandb:                         MAE_0.5 ▁▅█
wandb:                        MAPE_0.5 ▁▅█
wandb:                        MASE_0.5 ▁▆█
wandb:                         MSE_0.5 ▁▆█
wandb:                        MSE_mean ▁▆█
wandb:                            MSIS ▁▅█
wandb:                          ND_0.5 ▁▆█
wandb:                      NRMSE_mean ▁▆█
wandb:                       RMSE_mean ▁▆█
wandb: mean_weighted_sum_quantile_loss ▁▆█
wandb:                    num_variates ▁▁▁
wandb:                       sMAPE_0.5 ▁▆█
wandb: 
wandb: Run summary:
wandb:                         MAE_0.5 75.55771
wandb:                        MAPE_0.5 0.07702
wandb:                        MASE_0.5 1.3448
wandb:                         MSE_0.5 130448.28395
wandb:                        MSE_mean 130448.28395
wandb:                            MSIS 22.38653
wandb:                          ND_0.5 0.05598
wandb:                      NRMSE_mean 0.26757
wandb:                       RMSE_mean 361.17625
wandb:                          domain Web/CloudOps
wandb: mean_weighted_sum_quantile_loss 0.0516
wandb:                    num_variates 2
wandb:                       sMAPE_0.5 0.07521
wandb:                            term long
wandb: 
wandb: 🚀 View run tabpfn-ts-paper/bizitobs_service at: https://wandb.ai/plaskod/tabpfn-ts-experiments/runs/qip2alek
wandb: ⭐️ View project at: https://wandb.ai/plaskod/tabpfn-ts-experiments
done

hierarchical_sales 

restaurant 
"""
SHORT_DATASETS = "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H".split()
MED_LONG_DATASETS = "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H".split()

# Get union of short and med_long datasets
ALL_DATASETS = list(set(SHORT_DATASETS + MED_LONG_DATASETS))
DATASET_PROPERTIES_MAP = json.load(
    open(Path(__file__).parent / "dataset_properties.json")
)

