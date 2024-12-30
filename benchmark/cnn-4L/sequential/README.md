# 4-Layer CNN - Sequential Design - logs/measurements
- Targeting ~400 BRAM and ~100 URAM
- Actual: 382 BRAM and 96 URAM

- CSIM:
```
Cycle count: 126339
Kernel time (ns): 183928832
Kernel time (us): 183929
Kernel time (ms): 183.929
```

- Vitis HW EMU
```
Cycle count: 1500860
Kernel time (ns): 774072121200
Kernel time (us): 7.74072e+08
Kernel time (ms): 774072
```

- Vitis HW
```
Cycle count: 2215282
Kernel time (ns): 7629614
Kernel time (us): 7629.61
Kernel time (ms): 7.62961
```

TAPA Estimation (from `work.out/run-1/run/autobridge-XXX-XX-XXXX-XX:XX.log`):
```
The total area of the design:
  BRAM: 431 / 3504 = 12.3%
  DSP: 576 / 8496 = 6.8%
  FF: 89432 / 2331840 = 3.8%
  LUT: 85704.125 / 1165920 = 7.4%
  URAM: 96 / 960 = 10.0%

Slot CR_X0Y0_To_CR_X3Y3:
  [BRAM]: 56.1% (431 / 768)
  [DSP ]: 40.0% (576 / 1440)
  [FF  ]: 20.3% (89432 / 441600)
  [LUT ]: 38.8% (85704.125 / 220800)
  [URAM]: 75.0% (96 / 128)

total wire length: 0
SLR boundary 0 - 1 has 0 crossings
SLR boundary 1 - 2 has 0 crossings
SLR boundary 2 - 3 has 0 crossings

+--------------------+----------+---------+--------+---------+----------+
|     Slot Name      | BRAM (%) | DSP (%) | FF (%) | LUT (%) | URAM (%) |
+--------------------+----------+---------+--------+---------+----------+
| CR_X0Y0_To_CR_X3Y3 |   56.1   |   40.0  |  20.3  |   38.8  |   75.0   |
+--------------------+----------+---------+--------+---------+----------+

The device could be partitioned into 1 slots.
```

Utilization (from `work.out/run-1/vitis_run_hw/CNN4L_xilinx_u280_xdma_201920_3.temp/reports/link/imp/impl_1_init_report_utilization_0.rpt`):
```
+----------------------------+--------+--------+------------+-----------+-------+
|          Site Type         |  Used  |  Fixed | Prohibited | Available | Util% |
+----------------------------+--------+--------+------------+-----------+-------+
| CLB LUTs*                  | 174798 | 105683 |        960 |   1302720 | 13.42 |
|   LUT as Logic             | 140930 |  97436 |        960 |   1302720 | 10.82 |
|   LUT as Memory            |  33868 |   8247 |        480 |    600480 |  5.64 |
|     LUT as Distributed RAM |  14495 |   5563 |            |           |       |
|     LUT as Shift Register  |  19373 |   2684 |            |           |       |
| CLB Registers              | 207196 | 128746 |          0 |   2607360 |  7.95 |
|   Register as Flip Flop    | 207194 | 128744 |          0 |   2607360 |  7.95 |
|   Register as Latch        |      0 |      0 |          0 |   2607360 |  0.00 |
|   Register as AND/OR       |      2 |      2 |          0 |   2607360 | <0.01 |
| CARRY8                     |   2877 |   1069 |        120 |    162840 |  1.77 |
| F7 Muxes                   |   9565 |   1495 |        480 |    651360 |  1.47 |
| F8 Muxes                   |    211 |    204 |        240 |    325680 |  0.06 |
| F9 Muxes                   |      0 |      0 |        120 |    162840 |  0.00 |
+----------------------------+--------+--------+------------+-----------+-------+
```

```
+-------------------+-------+-------+------------+-----------+-------+
|     Site Type     |  Used | Fixed | Prohibited | Available | Util% |
+-------------------+-------+-------+------------+-----------+-------+
| Block RAM Tile    | 382.5 |     0 |          0 |      2016 | 18.97 |
|   RAMB36/FIFO*    |   198 |   196 |          0 |      2016 |  9.82 |
|     RAMB36E2 only |   198 |       |            |           |       |
|   RAMB18          |   369 |     8 |          0 |      4032 |  9.15 |
|     RAMB18E2 only |   369 |       |            |           |       |
| URAM              |    96 |     0 |          0 |       960 | 10.00 |
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
