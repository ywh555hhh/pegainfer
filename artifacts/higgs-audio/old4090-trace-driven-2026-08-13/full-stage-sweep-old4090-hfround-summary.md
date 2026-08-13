# Old4090 HF-round Full-Stage Sweep Summary

| layer | first mean alert | worst stage | worst mean | input hidden | gate proj | down proj | output hidden | out cos |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 0 | none | layer0.output_hidden.bf16 | 0.001837 | 0.000000 | 0.000340 | 0.001592 | 0.001837 | 0.999996662 |
| 1 | layer1.k_norm.bf16 | layer1.gate_proj.bf16 | 0.005716 | 0.001837 | 0.005716 | 0.001149 | 0.003201 | 0.999998093 |
| 2 | layer2.input_hidden.bf16 | layer2.gate_proj.bf16 | 0.008443 | 0.003201 | 0.008443 | 0.002712 | 0.006660 | 0.999996603 |
| 3 | layer3.input_hidden.bf16 | layer3.gate_proj.bf16 | 0.016479 | 0.006660 | 0.016479 | 0.002650 | 0.008661 | 0.999995410 |
| 4 | layer4.input_hidden.bf16 | layer4.gate_proj.bf16 | 0.027967 | 0.008661 | 0.027967 | 0.005975 | 0.011613 | 0.999996066 |
| 5 | layer5.input_hidden.bf16 | layer5.k_norm_rope.bf16 | 0.016827 | 0.011613 | 0.005515 | 0.008461 | 0.012024 | 0.999997020 |
| 6 | layer6.input_hidden.bf16 | layer6.output_hidden.bf16 | 0.012229 | 0.012024 | 0.003370 | 0.006447 | 0.012229 | 0.999996483 |
| 7 | layer7.input_hidden.bf16 | layer7.output_hidden.bf16 | 0.016650 | 0.012229 | 0.003984 | 0.009503 | 0.016650 | 0.999996126 |
| 8 | layer8.input_hidden.bf16 | layer8.output_hidden.bf16 | 0.018916 | 0.016650 | 0.004476 | 0.010767 | 0.018916 | 0.999994993 |
| 9 | layer9.input_hidden.bf16 | layer9.output_hidden.bf16 | 0.020586 | 0.018916 | 0.005218 | 0.014162 | 0.020586 | 0.999995291 |
| 10 | layer10.input_hidden.bf16 | layer10.output_hidden.bf16 | 0.021973 | 0.020586 | 0.005193 | 0.014633 | 0.021973 | 0.999995410 |
| 11 | layer11.input_hidden.bf16 | layer11.output_hidden.bf16 | 0.022219 | 0.021973 | 0.005381 | 0.012817 | 0.022219 | 0.999996364 |
| 12 | layer12.input_hidden.bf16 | layer12.input_hidden.bf16 | 0.022219 | 0.022219 | 0.004144 | 0.009748 | 0.021040 | 0.999996185 |
| 13 | layer13.input_hidden.bf16 | layer13.output_hidden.bf16 | 0.022183 | 0.021040 | 0.003596 | 0.008882 | 0.022183 | 0.999996841 |
| 14 | layer14.input_hidden.bf16 | layer14.output_hidden.bf16 | 0.023501 | 0.022183 | 0.003523 | 0.009781 | 0.023501 | 0.999994755 |
| 15 | layer15.input_hidden.bf16 | layer15.input_hidden.bf16 | 0.023501 | 0.023501 | 0.003576 | 0.008932 | 0.023470 | 0.999995291 |
| 16 | layer16.input_hidden.bf16 | layer16.input_hidden.bf16 | 0.023470 | 0.023470 | 0.003397 | 0.007864 | 0.023237 | 0.999994457 |
| 17 | layer17.input_hidden.bf16 | layer17.output_hidden.bf16 | 0.024428 | 0.023237 | 0.003356 | 0.008043 | 0.024428 | 0.999995589 |
| 18 | layer18.input_hidden.bf16 | layer18.output_hidden.bf16 | 0.024889 | 0.024428 | 0.003305 | 0.007976 | 0.024889 | 0.999995768 |
| 19 | layer19.input_hidden.bf16 | layer19.output_hidden.bf16 | 0.030796 | 0.024889 | 0.004059 | 0.014455 | 0.030796 | 0.999994516 |
| 20 | layer20.input_hidden.bf16 | layer20.output_hidden.bf16 | 0.033005 | 0.030796 | 0.004318 | 0.016168 | 0.033005 | 0.999995589 |
| 21 | layer21.input_hidden.bf16 | layer21.output_hidden.bf16 | 0.037063 | 0.033005 | 0.004550 | 0.016634 | 0.037063 | 0.999993682 |
| 22 | layer22.input_hidden.bf16 | layer22.output_hidden.bf16 | 0.040443 | 0.037063 | 0.004123 | 0.020671 | 0.040443 | 0.999997377 |
| 23 | layer23.input_hidden.bf16 | layer23.output_hidden.bf16 | 0.051274 | 0.040443 | 0.004896 | 0.032381 | 0.051274 | 0.999997795 |
| 24 | layer24.input_hidden.bf16 | layer24.output_hidden.bf16 | 0.057731 | 0.051274 | 0.004699 | 0.025590 | 0.057731 | 0.999997497 |
| 25 | layer25.input_hidden.bf16 | layer25.output_hidden.bf16 | 0.065322 | 0.057731 | 0.004769 | 0.025690 | 0.065322 | 0.999997497 |
| 26 | layer26.input_hidden.bf16 | layer26.output_hidden.bf16 | 0.071905 | 0.065322 | 0.004976 | 0.027988 | 0.071905 | 0.999997735 |
| 27 | layer27.input_hidden.bf16 | layer27.output_hidden.bf16 | 0.079988 | 0.071905 | 0.005185 | 0.027533 | 0.079988 | 0.999997675 |
| 28 | layer28.input_hidden.bf16 | layer28.output_hidden.bf16 | 0.090144 | 0.079988 | 0.005417 | 0.030530 | 0.090144 | 0.999997616 |
| 29 | layer29.input_hidden.bf16 | layer29.output_hidden.bf16 | 0.104378 | 0.090144 | 0.005883 | 0.034254 | 0.104378 | 0.999997675 |
| 30 | layer30.input_hidden.bf16 | layer30.output_hidden.bf16 | 0.121396 | 0.104378 | 0.006007 | 0.044518 | 0.121396 | 0.999998152 |
| 31 | layer31.input_hidden.bf16 | layer31.output_hidden.bf16 | 0.142101 | 0.121396 | 0.006467 | 0.060405 | 0.142101 | 0.999997437 |
| 32 | layer32.input_hidden.bf16 | layer32.output_hidden.bf16 | 0.197028 | 0.142101 | 0.008314 | 0.097637 | 0.197028 | 0.999994576 |
| 33 | layer33.input_hidden.bf16 | layer33.output_hidden.bf16 | 0.249898 | 0.197028 | 0.008926 | 0.119294 | 0.249898 | 0.999994576 |
| 34 | layer34.input_hidden.bf16 | layer34.output_hidden.bf16 | 0.546417 | 0.249898 | 0.021403 | 0.340734 | 0.546417 | 0.999990582 |
| 35 | layer35.input_hidden.bf16 | layer35.output_hidden.bf16 | 1.036249 | 0.546417 | 0.033704 | 0.668163 | 1.036249 | 0.999991179 |
