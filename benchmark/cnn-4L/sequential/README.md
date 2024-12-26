# 4-Layer CNN logs/measurements

- CSIM:
```
Cycle count: 126785
Kernel time (ns): 218364559
Kernel time (us): 218365
Kernel time (ms): 218.365
```

- Vitis HW EMU
```
Cycle count: 1497275
Kernel time (ns): 746067902335
Kernel time (us): 7.46068e+08
Kernel time (ms): 746068
```

- Vitis HW
```
Cycle count: 2369862
Kernel time (ns): 8110721
Kernel time (us): 8110.72
Kernel time (ms): 8.11072
```

TAPA Estimation (from `work.out/run-1/run/autobridge-XXX-XX-XXXX-XX:XX.log`):
```
The total area of the design:
  BRAM: 241 / 3504 = 6.9%
  DSP: 576 / 8496 = 6.8%
  FF: 90483 / 2331840 = 3.9%
  LUT: 83376.125 / 1165920 = 7.2%
  URAM: 0 / 960 = 0.0%

total wire length: 0
SLR boundary 0 - 1 has 0 crossings
SLR boundary 1 - 2 has 0 crossings
SLR boundary 2 - 3 has 0 crossings
Floorplan finishes

+--------------------+----------+---------+--------+---------+----------+
|     Slot Name      | BRAM (%) | DSP (%) | FF (%) | LUT (%) | URAM (%) |
+--------------------+----------+---------+--------+---------+----------+
| CR_X0Y0_To_CR_X3Y3 |   31.4   |   40.0  |  20.5  |   37.8  |   0.0    |
+--------------------+----------+---------+--------+---------+----------+
```

Utilization (from `work.out/run-1/vitis_run_hw/CNN4L_xilinx_u280_xdma_201920_3.temp/reports/link/imp/impl_1_init_report_utilization_0.rpt`):
```
+----------------------------+--------+--------+------------+-----------+-------+
|          Site Type         |  Used  |  Fixed | Prohibited | Available | Util% |
+----------------------------+--------+--------+------------+-----------+-------+
| CLB LUTs*                  | 173043 | 105683 |        960 |   1302720 | 13.28 |
|   LUT as Logic             | 137547 |  97436 |        960 |   1302720 | 10.56 |
|   LUT as Memory            |  35496 |   8247 |        480 |    600480 |  5.91 |
|     LUT as Distributed RAM |  16127 |   5563 |            |           |       |
|     LUT as Shift Register  |  19369 |   2684 |            |           |       |
| CLB Registers              | 205816 | 128746 |          0 |   2607360 |  7.89 |
|   Register as Flip Flop    | 205814 | 128744 |          0 |   2607360 |  7.89 |
|   Register as Latch        |      0 |      0 |          0 |   2607360 |  0.00 |
|   Register as AND/OR       |      2 |      2 |          0 |   2607360 | <0.01 |
| CARRY8                     |   2532 |   1069 |        120 |    162840 |  1.55 |
| F7 Muxes                   |   9565 |   1495 |        480 |    651360 |  1.47 |
| F8 Muxes                   |    211 |    204 |        240 |    325680 |  0.06 |
| F9 Muxes                   |      0 |      0 |        120 |    162840 |  0.00 |
+----------------------------+--------+--------+------------+-----------+-------+
```

```
+-------------------+-------+-------+------------+-----------+-------+
|     Site Type     |  Used | Fixed | Prohibited | Available | Util% |
+-------------------+-------+-------+------------+-----------+-------+
| Block RAM Tile    | 322.5 |     0 |          0 |      2016 | 16.00 |
|   RAMB36/FIFO*    |   198 |   196 |          0 |      2016 |  9.82 |
|     RAMB36E2 only |   198 |       |            |           |       |
|   RAMB18          |   249 |     8 |          0 |      4032 |  6.18 |
|     RAMB18E2 only |   249 |       |            |           |       |
| URAM              |     0 |     0 |          0 |       960 |  0.00 |
+-------------------+-------+-------+------------+-----------+-------+
```

```
+----------------+------+-------+------------+-----------+-------+
|    Site Type   | Used | Fixed | Prohibited | Available | Util% |
+----------------+------+-------+------------+-----------+-------+
| DSPs           |  580 |     4 |          0 |      9024 |  6.43 |
|   DSP48E2 only |  580 |       |            |           |       |
+----------------+------+-------+------------+-----------+-------+
```
