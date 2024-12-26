# 4-Layer CNN logs/measurements

- CSIM:
```
Cycle count: 11730
Kernel time (ns): 100634424
Kernel time (us): 100634
Kernel time (ms): 100.634
```

- Vitis HW EMU
```
Cycle count: 713071
Kernel time (ns): 360029685278
Kernel time (us): 3.6003e+08
Kernel time (ms): 360030
```

- Vitis HW
```
Cycle count: 713144
Kernel time (ns): 2536209
Kernel time (us): 2536.21
Kernel time (ms): 2.53621
```

TAPA Estimation (from `work.out/run-1/run/autobridge-XXX-XX-XXXX-XX:XX.log`):
```
The total area of the design:
  BRAM: 241 / 3504 = 6.9%
  DSP: 579 / 8496 = 6.8%
  FF: 71560 / 2331840 = 3.1%
  LUT: 66549.0625 / 1165920 = 5.7%
  URAM: 0 / 960 = 0.0%

total wire length: 0
SLR boundary 0 - 1 has 0 crossings
SLR boundary 1 - 2 has 0 crossings
SLR boundary 2 - 3 has 0 crossings
Floorplan finishes

+--------------------+----------+---------+--------+---------+----------+
|     Slot Name      | BRAM (%) | DSP (%) | FF (%) | LUT (%) | URAM (%) |
+--------------------+----------+---------+--------+---------+----------+
| CR_X0Y0_To_CR_X3Y3 |   31.4   |   40.2  |  16.2  |   30.1  |   0.0    |
+--------------------+----------+---------+--------+---------+----------+
```

Utilization (from `work.out/run-1/vitis_run_hw/CNN4L_xilinx_u280_xdma_201920_3.temp/reports/link/imp/impl_1_init_report_utilization_0.rpt`):
```
+----------------------------+--------+--------+------------+-----------+-------+
|          Site Type         |  Used  |  Fixed | Prohibited | Available | Util% |
+----------------------------+--------+--------+------------+-----------+-------+
| CLB LUTs*                  | 164154 | 105683 |        960 |   1302720 | 12.60 |
|   LUT as Logic             | 132288 |  97436 |        960 |   1302720 | 10.15 |
|   LUT as Memory            |  31866 |   8247 |        480 |    600480 |  5.31 |
|     LUT as Distributed RAM |  14711 |   5563 |            |           |       |
|     LUT as Shift Register  |  17155 |   2684 |            |           |       |
| CLB Registers              | 200691 | 128746 |          0 |   2607360 |  7.70 |
|   Register as Flip Flop    | 200689 | 128744 |          0 |   2607360 |  7.70 |
|   Register as Latch        |      0 |      0 |          0 |   2607360 |  0.00 |
|   Register as AND/OR       |      2 |      2 |          0 |   2607360 | <0.01 |
| CARRY8                     |   2337 |   1069 |        120 |    162840 |  1.44 |
| F7 Muxes                   |   7109 |   1495 |        480 |    651360 |  1.09 |
| F8 Muxes                   |    204 |    204 |        240 |    325680 |  0.06 |
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
| DSPs           |  583 |     4 |          0 |      9024 |  6.46 |
|   DSP48E2 only |  583 |       |            |           |       |
+----------------+------+-------+------------+-----------+-------+
```
