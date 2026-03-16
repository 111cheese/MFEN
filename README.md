# MFEN
## Requirements

- Python 3.8.16
- PyTorch 1.12.1
- cudatoolkit 11.3.1
- torchvision 0.13.1

## Data

Put datasets under `datasets/`:
- Indian Pines: `Indian_pines_corrected.mat`, `Indian_pines_gt.mat`
- PaviaU: `PaviaU.mat`, `PaviaU_gt.mat`
- Salinas: `Salinas_corrected.mat`, `Salinas_gt.mat`
- WHU_LongKou: `WHU_Hi_LongKou.mat`, `WHU_Hi_LongKou_gt.mat`

## Run Our Method

1. Generate AGF files in `AGF`.
2. Edit 'Main.py`:
	- dataset paths
	- `(FLAG, curr_train_ratio, Scale)`
3. Run:

```bash
python Main.py
```
