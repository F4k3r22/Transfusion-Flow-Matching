# Transfusion with Flow-Matching

Hey, we're implementing the Transfusion model outlined in this META paper: [Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model](https://arxiv.org/pdf/2408.11039)
We're experiencing these errors. Any suggestions on how to resolve them are welcome. :b. Thank you all in advance.

Errors we are having when trying to train the model:
```
Advertencia: Número de parches no coincide - transformer: 16, esperado: 4
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [2, 8:24] shape: torch.Size([16, 512])
Contenido no cero: 8192
dx_t shape: torch.Size([2, 8, 4, 4]), flow_idx: 1
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - transformer: 16, esperado: 4
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Pérdida de flow inválida, usando valor por defecto
Evaluando: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:01<00:00,  6.82it/s]
2025-04-14 04:56:07,759 - INFO - Pérdida de validación: nan, Pérdida LM: nan, Pérdida FM: 0.1000
Época 1/10:   1%|█                                                                                 | 3/238 [00:16<21:14,  5.42s/it, loss=15, lm_loss=15, fm_loss=0, lr=0.00e+00]Procesando batch con 4 imágenes
BOI índices: [7, 12, 6, 10]
Imagen 3606846822_28c40b933a.jpg.pt: calculados 16 parches necesarios
Espacio para parches: 16, requerido: 16
Creado image_latents_dict con 4 entradas
Claves en image_latents_dict: [7, 12, 6, 10]
Forward recibiendo 4 latentes de imagen
Imagen 2735979477_eef7c680f9.jpg.pt: calculados 16 parches necesarios
Espacio para parches: 16, requerido: 16
Imagen 3089742441_d42531c14f.jpg.pt: calculados 16 parches necesarios
Espacio para parches: 16, requerido: 16
Token BOI: 128256, EOI: 128257
Tokens únicos: tensor([     0,      8,     11,     13,     14,     18,     21,     22,     25,
            26,     32,     33,     37,     40,     42,     53,    220,    235,
           259,    264,    271,    279,    287,    288,    291,    292,    295,
           300,    304,    306,    311,    315,    320,    323,    348,    358,
           362,    369,    374,    382,    386,    387,    388,    389,    398,
           400,    420,    422,    423,    430,    433,    437,    439,    449,
           459,    462,    477,    478,    479,    481,    482,    499,    502,
           505,    520,    527,    535,    538,    539,    549,    555,    568,
           574,    578,    584,    596,    610,    611,    614,    617,    636,
           644,    647,    649,    659,    662,    667,    679,    682,    690,
           701,    704,    706,    709,    719,    733,    735,    751,    753,
           757,    763,    772,    777,    779,    791,    810,    813,    814,
           815,    818,    832,    834,    856,    863,    868,    872,    879,
           889,    892,    893,    902,    912,    922,    927,    955,    961,
           966,    972,    988,    990,   1008,   1022,   1027,   1041,   1047,
          1051,   1053,   1054,   1057,   1063,   1070,   1071,   1093,   1101,
          1102,   1109,   1111,   1115,   1120,   1121,   1124,   1131,   1135,
          1139,   1148,   1162,   1176,   1180,   1193,   1202,   1205,   1226,
          1234,   1253,   1260,   1268,   1272,   1274,   1288,   1291,   1304,
          1306,   1317,   1358,   1363,   1364,   1389,   1390,   1399,   1403,
          1404,   1405,   1418,   1431,   1436,   1440,   1442,   1456,   1461,
          1464,   1473,   1475,   1510,   1511,   1518,   1521,   1524,   1541,
          1543,   1550,   1555,   1561,   1566,   1606,   1614,   1618,   1629,
          1633,   1648,   1664,   1667,   1684,   1695,   1701,   1748,   1772,
          1773,   1778,   1781,   1790,   1820,   1828,   1847,   1862,   1866,
          1884,   1887,   1888,   1893,   1899,   1903,   1912,   1916,   1935,
          1939,   1945,   1948,   1957,   1959,   1964,   1966,   1972,   1975,
          1984,   1988,   1992,   2001,   2007,   2025,   2029,   2046,   2082,
          2085,   2115,   2128,   2133,   2162,   2170,   2175,   2204,   2216,
          2217,   2231,   2294,   2326,   2343,   2349,   2380,   2385,   2403,
          2410,   2435,   2460,   2468,   2476,   2478,   2500,   2543,   2555,
          2563,   2567,   2574,   2586,   2643,   2646,   2656,   2678,   2688,
          2697,   2728,   2731,   2751,   2753,   2763,   2766,   2768,   2771,
          2795,   2834,   2835,   2867,   2883,   2950,   2997,   3010,   3025,
          3041,   3115,   3117,   3135,   3136,   3196,   3197,   3207,   3224,
          3230,   3235,   3245,   3268,   3296,   3297,   3300,   3309,   3318,
          3325,   3335,   3339,   3361,   3392,   3403,   3420,   3432,   3449,
          3463,   3477,   3485,   3488,   3492,   3508,   3512,   3549,   3568,
          3575,   3582,   3585,   3604,   3610,   3634,   3662,   3697,   3719,
          3754,   3775,   3776,   3782,   3814,   3827,   3828,   3830,   3831,
          3839,   3842,   3871,   3892,   3907,   3966,   3995,   4018,   4028,
          4029,   4032,   4034,   4054,   4062,   4070,   4072,   4131,   4219,
          4227,   4236,   4255,   4315,   4344,   4375,   4395,   4409,   4465,
          4491,   4495,   4500,   4510,   4516,   4519,   4528,   4546,   4577,
          4604,   4619,   4676,   4702,   4717,   4718,   4726,   4737,   4779,
          4780,   4805,   4839,   4846,   4857,   4900,   4901,   4920,   4933,
          4947,   4949,   4987,   5015,   5016,   5030,   5076,   5097,   5115,
          5133,   5151,   5153,   5157,   5219,   5220,   5321,   5326,   5348,
          5352,   5353,   5357,   5370,   5403,   5513,   5575,   5597,   5603,
          5605,   5613,   5616,   5647,   5651,   5678,   5679,   5688,   5694,
          5695,   5730,   5734,   5737,   5762,   5817,   5856,   5922,   5955,
          5975,   6007,   6029,   6062,   6068,   6108,   6136,   6137,   6164,
          6197,   6205,   6211,   6276,   6278,   6316,   6355,   6380,   6381,
          6411,   6420,   6439,   6541,   6574,   6612,   6652,   6740,   6776,
          6781,   6821,   6859,   6864,   6928,   6968,   6982,   6992,   6996,
          7009,   7041,   7043,   7055,   7070,   7072,   7077,   7123,   7124,
          7182,   7184,   7188,   7224,   7231,   7239,   7294,   7351,   7362,
          7378,   7422,   7429,   7430,   7447,   7463,   7493,   7532,   7556,
          7572,   7616,   7671,   7742,   7855,   7872,   7926,   7953,   7974,
          7997,   8010,   8060,   8079,   8096,   8140,   8184,   8272,   8312,
          8334,   8389,   8421,   8427,   8431,   8450,   8468,   8518,   8612,
          8620,   8671,   8722,   8776,   8821,   8826,   8830,   8850,   8857,
          8881,   8903,   8922,   8950,   9002,   9041,   9048,   9052,   9057,
          9070,   9087,   9145,   9160,   9204,   9251,   9257,   9322,   9333,
          9337,   9341,   9504,   9508,   9527,   9540,   9640,   9860,   9903,
          9932,   9973,  10014,  10082,  10118,  10212,  10240,  10255,  10269,
         10307,  10323,  10374,  10388,  10399,  10510,  10552,  10578,  10666,
         10712,  10936,  10960,  11012,  11040,  11050,  11062,  11196,  11262,
         11291,  11335,  11411,  11439,  11450,  11493,  11650,  11668,  11704,
         11709,  11710,  11714,  11846,  11890,  11897,  11922,  11927,  11934,
         11969,  11989,  12014,  12135,  12157,  12158,  12205,  12265,  12330,
         12331,  12332,  12334,  12361,  12374,  12410,  12437,  12490,  12634,
         12643,  12649,  12789,  12860,  12875,  12885,  12886,  12889,  12912,
         13030,  13063,  13203,  13238,  13256,  13266,  13299,  13358,  13426,
         13452,  13461,  13520,  13551,  13606,  13658,  13780,  13788,  13806,
         13929,  13933,  13942,  13967,  13987,  14050,  14058,  14305,  14327,
         14393,  14435,  14454,  14610,  14718,  14867,  14880,  14947,  14956,
         15061,  15085,  15109,  15177,  15187,  15236,  15338,  15393,  15455,
         15664,  15692,  15718,  15845,  15859,  15879,  16058,  16064,  16168,
         16207,  16459,  16716,  16763,  16926,  17033,  17045,  17067,  17102,
         17104,  17271,  17395,  17715,  17821,  18200,  18317,  18396,  18598,
         18651,  18671,  18711,  18796,  18885,  18988,  19031,  19035,  19092,
         19130,  19300,  19336,  19558,  19569,  19582,  19698,  19730,  19738,
         19768,  19983,  20136,  20147,  20323,  20413,  20643,  20991,  21047,
         21077,  21319,  21357,  21455,  21463,  21760,  21973,  21988,  22178,
         22362,  22465,  22538,  22670,  22797,  22848,  22936,  22959,  23062,
         23194,  23212,  23304,  23442,  23746,  23786,  23872,  24038,  24060,
         24151,  24269,  24364,  24426,  24565,  24846,  24991,  25223,  25428,
         25497,  25559,  25700,  25949,  26160,  26283,  26368,  26386,  26454,
         26475,  26526,  27052,  27053,  27253,  27267,  27292,  27565,  27741,
         27992,  28040,  28055,  28338,  28366,  28591,  28640,  28818,  28994,
         28999,  29084,  29091,  29237,  29447,  29695,  29742,  29755,  30090,
         30293,  30350,  30447,  30511,  30530,  30627,  30693,  31005,  31202,
         31352,  31378,  31613,  31696,  31997,  32157,  32799,  32838,  33230,
         33352,  33718,  33980,  34106,  34241,  34375,  34464,  34558,  34606,
         34734,  34821,  35255,  35271,  35331,  35336,  35353,  35709,  35789,
         35938,  36635,  36698,  36892,  36978,  37514,  37631,  37987,  38132,
         38964,  39473,  40020,  40495,  40565,  40916,  41675,  41691,  42258,
         42769,  43080,  43241,  43495,  43535,  43749,  44929,  44963,  46704,
         47044,  47249,  47459,  47652,  47678,  47769,  48852,  49227,  49664,
         50104,  50863,  50977,  52042,  52249,  53008,  53221,  53703,  53825,
         54774,  55060,  55273,  55806,  55907,  56168,  57133,  57995,  58302,
         58348,  58737,  58840,  59412,  60080,  60867,  61118,  61279,  62182,
         62464,  62469,  63788,  64452,  65235,  65308,  87820, 128256, 128257],
       device='cuda:0')
Batch 0 tiene 1 tokens BOI en posiciones tensor([7], device='cuda:0')
Batch 0 tiene 1 tokens EOI en posiciones tensor([24], device='cuda:0')
Batch 1 tiene 1 tokens BOI en posiciones tensor([12], device='cuda:0')
Batch 1 tiene 1 tokens EOI en posiciones tensor([29], device='cuda:0')
Batch 4 tiene 1 tokens BOI en posiciones tensor([6], device='cuda:0')
Batch 4 tiene 1 tokens EOI en posiciones tensor([23], device='cuda:0')
Batch 7 tiene 1 tokens BOI en posiciones tensor([10], device='cuda:0')
Batch 7 tiene 1 tokens EOI en posiciones tensor([27], device='cuda:0')
Posiciones de imágenes encontradas: [(0, 7, 24), (1, 12, 29), (4, 6, 23), (7, 10, 27)]
x_t shape: torch.Size([4, 8, 4, 4]), dx_t shape: torch.Size([4, 8, 4, 4])
batch_idx_map: {0: 0, 1: 1, 4: 2, 7: 3}
Tensor en posición [0, 8:24] shape: torch.Size([16, 512])
Contenido no cero: 8192
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 0
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - transformer: 16, esperado: 4
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [1, 13:29] shape: torch.Size([16, 512])
Contenido no cero: 8192
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 1
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - transformer: 16, esperado: 4
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [4, 7:23] shape: torch.Size([16, 512])
Contenido no cero: 8192
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 2
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - transformer: 16, esperado: 4
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [7, 11:27] shape: torch.Size([16, 512])
Contenido no cero: 8192
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 3
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - transformer: 16, esperado: 4
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Pérdida de flow inválida, usando valor por defecto
¡Advertencia! Pérdida NaN detectada en paso 0
Época 1/10:   1%|█                                                                                 | 3/238 [00:16<21:19,  5.45s/it, loss=15, lm_loss=15, fm_loss=0, lr=0.00e+00]
Traceback (most recent call last):
  File "/teamspace/studios/this_studio/tr-fl-local/train.py", line 647, in <module>
    train_transfusion_model()
  File "/teamspace/studios/this_studio/tr-fl-local/train.py", line 642, in train_transfusion_model
    trained_model = trainer.train(train_dataloader, val_dataloader)
  File "/teamspace/studios/this_studio/tr-fl-local/train.py", line 501, in train
    stop_training = self.train_epoch(train_dataloader, epoch)
  File "/teamspace/studios/this_studio/tr-fl-local/train.py", line 236, in train_epoch
    self.scaler.step(self.optimizer)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/amp/grad_scaler.py", line 454, in step
    len(optimizer_state["found_inf_per_device"]) > 0
AssertionError: No inf checks were recorded for this optimizer.
```