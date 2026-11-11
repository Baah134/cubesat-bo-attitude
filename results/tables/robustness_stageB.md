| method                              |   designs | spec compliance % (mean ± std over designs)   |   min compliance % |   t_settle median [s] |   t_settle 95th pct [s] |   worst hold RMS [deg] |   runs with momentum violation % |
|:------------------------------------|----------:|:----------------------------------------------|-------------------:|----------------------:|------------------------:|-----------------------:|---------------------------------:|
| Robust BO (worst of 8 plants) [PID] |         9 | 95.2 ± 7.6                                    |               77   |                  8.23 |                   19.56 |                 0.0344 |                                0 |
| Constrained BO (GP) [PID]           |        10 | 86.2 ± 16.4                                   |               45.5 |                  8.11 |                   10.51 |                 0.0085 |                                0 |
| TPE (IECON) [PID]                   |         7 | 84.2 ± 15.9                                   |               56   |                  8.5  |                   24.17 |                 0.4038 |                                0 |
| CMA-ES [PID]                        |         9 | 82.1 ± 22.1                                   |               37.5 |                 14.27 |                  120    |                 9.1679 |                                0 |
| GA [PID]                            |         1 | 31.5 ± 0.0                                    |               31.5 |                 10.66 |                   11.95 |                 0.2818 |                                0 |
| Random (Sobol) [PID]                |         6 | 69.1 ± 36.2                                   |               14   |                 37.35 |                  120    |                 0.2918 |                                0 |
| MATLAB pidtune [PID]                |         1 | 100.0 ± 0.0                                   |              100   |                 48.01 |                   48.14 |                 0.0009 |                                0 |

200 sampled plants (seed 777), identical for every design.
