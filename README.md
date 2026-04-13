# AdcSR Non-uniform Stage Pruning (paper-release)

This folder is a clean export for reproducing the paper experiments.

## Key knobs (stage ratios)
Set environment variables before running training/eval:
- ADCSR_R0, ADCSR_R1, ADCSR_R2, ADCSR_R3, ADCSR_RMID

Paper best setting:
- R0=R1=R2=0.75
- R3=RMID=0.25

## Weights
- `weight/pretrained/` contains required pretrained files.
- `runs_ckpt/net_params_5.pkl` is an example trained checkpoint (best model).
