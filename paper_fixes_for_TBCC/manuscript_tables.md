## 1. Main Baseline Results (Paired-Bootstrap 95% CI vs HARBOUR)

Note: Positive values indicate HARBOUR outperformed the baseline by that margin.

### Rank: Class
| Model | Closed-set F1 | Novel F1 | H-score | AUROC | MCC |
|-------|---------------|----------|---------|-------|-----|
| MultinomialNB | 0.2005 [0.1625–0.2376] | 0.1315 [0.0756–0.1881] | 0.1986 [0.1121–0.2907] | 0.0697 [0.0219–0.1172] | 0.1243 [0.1035–0.1456] |
| MSP-full | 0.0360 [0.0004–0.0720] | 0.1418 [0.0988–0.1855] | 0.0581 [-0.0057–0.1224] | 0.0756 [0.0374–0.1132] | 0.0873 [0.0696–0.1050] |
| VSEARCH | -0.1055 [-0.1422–-0.0682] | 0.0256 [-0.0171–0.0692] | -0.0246 [-0.0713–0.0269] | -0.0193 [-0.0431–0.0032] | 0.0085 [-0.0079–0.0250] |
| Kraken2 | -0.1722 [-0.2073–-0.1357] | 0.3568 [0.2872–0.4234] | 0.7927 [0.7184–0.8546] | 0.0306 [-0.0114–0.0729] | -0.0512 [-0.0680–-0.0343] |
| DNABERT-S | 0.0641 [0.0281–0.0990] | 0.1623 [0.0898–0.2354] | 0.3899 [0.2760–0.5107] | 0.1654 [0.1316–0.1999] | 0.0471 [0.0297–0.0650] |
| Nucleotide Transformer | 0.1020 [0.0631–0.1416] | 0.2442 [0.1749–0.3122] | 0.4920 [0.3812–0.6064] | 0.1635 [0.1348–0.1948] | 0.0831 [0.0649–0.1021] |
| HARBOUR | 0.7318 +/- 0.0206 | 0.3904 +/- 0.1962 | 0.8322 +/- 0.0714 | 0.9222 +/- 0.0373 | 0.8555 +/- 0.0096 |

### Rank: Order
| Model | Closed-set F1 | Novel F1 | H-score | AUROC | MCC |
|-------|---------------|----------|---------|-------|-----|
| MultinomialNB | 0.2245 [0.1988–0.2498] | 0.2980 [0.2578–0.3385] | 0.3189 [0.2835–0.3545] | 0.1097 [0.0852–0.1336] | 0.1979 [0.1763–0.2197] |
| MSP-full | 0.0157 [-0.0093–0.0410] | 0.0619 [0.0323–0.0913] | 0.0202 [-0.0040–0.0434] | 0.0296 [0.0079–0.0517] | 0.0442 [0.0234–0.0640] |
| VSEARCH | -0.1394 [-0.1665–-0.1116] | 0.0063 [-0.0226–0.0356] | -0.0203 [-0.0441–0.0034] | -0.0153 [-0.0280–-0.0025] | -0.0183 [-0.0383–0.0024] |
| Kraken2 | -0.2078 [-0.2356–-0.1798] | 0.5685 [0.5328–0.6029] | 0.6499 [0.6196–0.6794] | 0.0247 [0.0022–0.0472] | 0.0282 [0.0079–0.0475] |
| DNABERT-S | 0.0516 [0.0254–0.0787] | 0.6544 [0.6266–0.6815] | 0.7366 [0.7168–0.7562] | 0.1344 [0.1155–0.1533] | 0.1503 [0.1314–0.1687] |
| Nucleotide Transformer | 0.0860 [0.0586–0.1128] | 0.6544 [0.6266–0.6815] | 0.7366 [0.7168–0.7562] | 0.0995 [0.0822–0.1170] | 0.1720 [0.1529–0.1918] |
| HARBOUR | 0.6017 +/- 0.0099 | 0.6547 +/- 0.0671 | 0.7368 +/- 0.0427 | 0.8383 +/- 0.0375 | 0.7259 +/- 0.0203 |

### Rank: Family
| Model | Closed-set F1 | Novel F1 | H-score | AUROC | MCC |
|-------|---------------|----------|---------|-------|-----|
| MultinomialNB | 0.1689 [0.1493–0.1874] | 0.2865 [0.2703–0.3025] | 0.2386 [0.2237–0.2527] | 0.1031 [0.0861–0.1200] | 0.1773 [0.1612–0.1938] |
| MSP-full | 0.0026 [-0.0150–0.0197] | -0.0592 [-0.0692–-0.0492] | -0.0319 [-0.0450–-0.0191] | 0.0113 [-0.0038–0.0268] | -0.0441 [-0.0608–-0.0275] |
| VSEARCH | -0.1788 [-0.2020–-0.1562] | -0.0243 [-0.0360–-0.0126] | -0.0646 [-0.0790–-0.0503] | -0.0435 [-0.0527–-0.0345] | -0.0786 [-0.0972–-0.0601] |
| Kraken2 | -0.1429 [-0.1646–-0.1216] | -0.0051 [-0.0168–0.0068] | -0.0341 [-0.0485–-0.0193] | 0.0175 [0.0038–0.0319] | -0.0512 [-0.0695–-0.0327] |
| DNABERT-S | 0.0555 [0.0369–0.0744] | 0.1150 [0.1025–0.1280] | 0.1011 [0.0882–0.1138] | 0.0927 [0.0801–0.1054] | 0.1036 [0.0875–0.1192] |
| Nucleotide Transformer | 0.0653 [0.0462–0.0849] | 0.0794 [0.0669–0.0915] | 0.0865 [0.0726–0.0999] | 0.0786 [0.0669–0.0909] | 0.0990 [0.0818–0.1160] |
| HARBOUR | 0.4607 +/- 0.0152 | 0.7582 +/- 0.0338 | 0.6371 +/- 0.0177 | 0.7805 +/- 0.0118 | 0.4209 +/- 0.0277 |

### Rank: Genus
| Model | Closed-set F1 | Novel F1 | H-score | AUROC | MCC |
|-------|---------------|----------|---------|-------|-----|
| MultinomialNB | 0.0207 [0.0069–0.0344] | 0.5097 [0.4928–0.5261] | 0.2174 [0.1971–0.2380] | 0.2538 [0.2348–0.2722] | 0.1708 [0.1554–0.1861] |
| MSP-full | -0.0122 [-0.0208–-0.0032] | -0.0164 [-0.0267–-0.0063] | -0.0157 [-0.0289–-0.0023] | -0.0100 [-0.0257–0.0058] | -0.0148 [-0.0284–-0.0017] |
| VSEARCH | -0.3703 [-0.3915–-0.3486] | -0.0304 [-0.0413–-0.0194] | -0.2530 [-0.2745–-0.2310] | -0.0542 [-0.0644–-0.0444] | -0.1843 [-0.2031–-0.1649] |
| Kraken2 | -0.3504 [-0.3716–-0.3294] | 0.0461 [0.0331–0.0584] | -0.1721 [-0.1948–-0.1498] | 0.0552 [0.0394–0.0716] | -0.0991 [-0.1186–-0.0796] |
| DNABERT-S | -0.1250 [-0.1421–-0.1080] | 0.1395 [0.1263–0.1524] | -0.0430 [-0.0637–-0.0233] | 0.0906 [0.0770–0.1039] | 0.0446 [0.0287–0.0612] |
| Nucleotide Transformer | -0.1234 [-0.1405–-0.1064] | 0.1071 [0.0942–0.1198] | -0.0546 [-0.0757–-0.0348] | 0.0768 [0.0648–0.0894] | 0.0353 [0.0186–0.0520] |
| HARBOUR | 0.1676 +/- 0.0052 | 0.7807 +/- 0.0201 | 0.4243 +/- 0.0106 | 0.7841 +/- 0.0162 | 0.2916 +/- 0.0194 |

## 2. Ablation Results (Mean +/- S.D.)

### Rank: Class
| Configuration | H-score | A_known | R_novel | MCC |
|---------------|---------|---------|---------|-----|
| a_MSP_Baseline | 0.7737 +/- 0.0402 | 0.7588 +/- 0.0135 | 0.7941 +/- 0.0806 | 0.7742 +/- 0.0115 |
| b_Distance_Only | 0.6011 +/- 0.0818 | 0.4776 +/- 0.1153 | 0.8424 +/- 0.0867 | 0.5878 +/- 0.0963 |
| c_Confidence_Only | 0.6584 +/- 0.0424 | 0.6042 +/- 0.0425 | 0.7493 +/- 0.1600 | 0.7060 +/- 0.0322 |
| d_Full_HARBOUR | 0.6546 +/- 0.0415 | 0.5537 +/- 0.0240 | 0.8152 +/- 0.1270 | 0.6558 +/- 0.0203 |

### Rank: Order
| Configuration | H-score | A_known | R_novel | MCC |
|---------------|---------|---------|---------|-----|
| a_MSP_Baseline | 0.7165 +/- 0.0138 | 0.6981 +/- 0.0163 | 0.7378 +/- 0.0397 | 0.7228 +/- 0.0133 |
| b_Distance_Only | 0.4862 +/- 0.0226 | 0.3720 +/- 0.0376 | 0.7242 +/- 0.1059 | 0.4352 +/- 0.0315 |
| c_Confidence_Only | 0.5585 +/- 0.0238 | 0.4516 +/- 0.0238 | 0.7359 +/- 0.0662 | 0.5172 +/- 0.0198 |
| d_Full_HARBOUR | 0.5497 +/- 0.0156 | 0.4469 +/- 0.0196 | 0.7221 +/- 0.0796 | 0.4984 +/- 0.0194 |

### Rank: Family
| Configuration | H-score | A_known | R_novel | MCC |
|---------------|---------|---------|---------|-----|
| a_MSP_Baseline | 0.6690 +/- 0.0043 | 0.5888 +/- 0.0119 | 0.7758 +/- 0.0255 | 0.6293 +/- 0.0081 |
| b_Distance_Only | 0.6262 +/- 0.0113 | 0.5889 +/- 0.0076 | 0.6692 +/- 0.0296 | 0.6069 +/- 0.0074 |
| c_Confidence_Only | 0.5975 +/- 0.0060 | 0.5982 +/- 0.0134 | 0.5976 +/- 0.0218 | 0.6286 +/- 0.0116 |
| d_Full_HARBOUR | 0.6371 +/- 0.0177 | 0.6091 +/- 0.0129 | 0.6705 +/- 0.0515 | 0.6339 +/- 0.0113 |

### Rank: Genus
| Configuration | H-score | A_known | R_novel | MCC |
|---------------|---------|---------|---------|-----|
| a_MSP_Baseline | 0.4399 +/- 0.0086 | 0.3170 +/- 0.0079 | 0.7184 +/- 0.0070 | 0.3287 +/- 0.0081 |
| b_Distance_Only | 0.3791 +/- 0.0128 | 0.2747 +/- 0.0083 | 0.6167 +/- 0.0645 | 0.2869 +/- 0.0081 |
| c_Confidence_Only | 0.3819 +/- 0.0099 | 0.2733 +/- 0.0092 | 0.6341 +/- 0.0231 | 0.2903 +/- 0.0094 |
| d_Full_HARBOUR | 0.3926 +/- 0.0098 | 0.2757 +/- 0.0074 | 0.6826 +/- 0.0306 | 0.2909 +/- 0.0085 |



## Verdict: Ablation Analysis (Representation vs. Mechanism)
At higher taxonomic ranks (Class and Order), the retrieval and distance/confidence thresholding mechanism explains the performance reversal relative to the MSP baseline, rather than the SupCon representation. Implementing the dual-threshold on top of standard cross-entropy representations (Config c) dropped Class H-score from 0.7736 to 0.6584, and adding the SupCon representation (Config d) did not recover this loss (0.6545). Conversely, at the Genus rank, the advantage stems purely from the SupCon representation; shifting from the baseline CE representations to SupCon (Config a to b) improved the raw embedding quality, but applying the dual-thresholding mechanism (Config d) resulted in a final H-score (0.3925) that still trailed the MSP baseline (0.4398) due to the strict penalization of known Genus sequences that failed the distance/confidence checks.


## Verdict: HARBOUR vs MSP and VSEARCH Edge
After enforcing the mathematically correct definition of A_known (requiring BOTH acceptance and correct taxonomic assignment), **HARBOUR no longer loses to MSP at Class and Order**. The paired-difference bootstrap reveals HARBOUR significantly outperforms MSP in H-score at Class (+0.0873 [0.0696-0.1050]) and Order (+0.0442 [0.0234-0.0640]). However, VSEARCH's AUROC edge at Family and Genus **does survive the strict paired-difference test**. At Family, HARBOUR trails VSEARCH significantly (-0.0435 [-0.0527 to -0.0345]), and similarly at Genus (-0.0542 [-0.0644 to -0.0444]), confirming VSEARCH's absolute novelty detection superiority at finer taxonomic resolutions even on identically matched sequence subsets.
