# 4-Layer CNN logs/measurements

- CSIM:
```
Cycle count: 126786
Kernel time (ns): 242647422
Kernel time (us): 242647
Kernel time (ms): 242.647
```

- Vitis HW EMU
```
Cycle count: 1667034
Kernel time (ns): 805060884350
Kernel time (us): 8.05061e+08
Kernel time (ms): 805061
```

- Vitis HW
```
Cycle count: 2369894
Kernel time (ns): 8134492
Kernel time (us): 8134.49
Kernel time (ms): 8.13449
```

TAPA Estimation (from `work.out/run-1/run/autobridge-XXX-XX-XXXX-XX:XX.log`):
```
The total area of the design:
  BRAM: 1 / 3504 = 0.0%
  DSP: 576 / 8496 = 6.8%
  FF: 84499 / 2331840 = 3.6%
  LUT: 83583.125 / 1165920 = 7.2%
  URAM: 0 / 960 = 0.0%

total wire length: 0
SLR boundary 0 - 1 has 0 crossings
SLR boundary 1 - 2 has 0 crossings
SLR boundary 2 - 3 has 0 crossings
Floorplan finishes

+--------------------+----------+---------+--------+---------+----------+
|     Slot Name      | BRAM (%) | DSP (%) | FF (%) | LUT (%) | URAM (%) |
+--------------------+----------+---------+--------+---------+----------+
| CR_X0Y0_To_CR_X3Y3 |   0.1    |   40.0  |  19.1  |   37.9  |   0.0    |
+--------------------+----------+---------+--------+---------+----------+
```

Utilization (from `work.out/run-1/vitis_run_hw/CNN4L_xilinx_u280_xdma_201920_3.temp/reports/link/imp/impl_1_init_report_utilization_0.rpt`):
```
+----------------------------+--------+--------+------------+-----------+-------+
|          Site Type         |  Used  |  Fixed | Prohibited | Available | Util% |
+----------------------------+--------+--------+------------+-----------+-------+
| CLB LUTs*                  | 174269 | 105683 |        960 |   1302720 | 13.38 |
|   LUT as Logic             | 134860 |  97436 |        960 |   1302720 | 10.35 |
|   LUT as Memory            |  39409 |   8247 |        480 |    600480 |  6.56 |
|     LUT as Distributed RAM |  20031 |   5563 |            |           |       |
|     LUT as Shift Register  |  19378 |   2684 |            |           |       |
| CLB Registers              | 201436 | 128746 |          0 |   2607360 |  7.73 |
|   Register as Flip Flop    | 201434 | 128744 |          0 |   2607360 |  7.73 |
|   Register as Latch        |      0 |      0 |          0 |   2607360 |  0.00 |
|   Register as AND/OR       |      2 |      2 |          0 |   2607360 | <0.01 |
| CARRY8                     |   2835 |   1069 |        120 |    162840 |  1.74 |
| F7 Muxes                   |  10205 |   1495 |        480 |    651360 |  1.57 |
| F8 Muxes                   |    211 |    204 |        240 |    325680 |  0.06 |
| F9 Muxes                   |      0 |      0 |        120 |    162840 |  0.00 |
+----------------------------+--------+--------+------------+-----------+-------+
```

```
+-------------------+-------+-------+------------+-----------+-------+
|     Site Type     |  Used | Fixed | Prohibited | Available | Util% |
+-------------------+-------+-------+------------+-----------+-------+
| Block RAM Tile    | 202.5 |     0 |          0 |      2016 | 10.04 |
|   RAMB36/FIFO*    |   198 |   196 |          0 |      2016 |  9.82 |
|     RAMB36E2 only |   198 |       |            |           |       |
|   RAMB18          |     9 |     8 |          0 |      4032 |  0.22 |
|     RAMB18E2 only |     9 |       |            |           |       |
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
