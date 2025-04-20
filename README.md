# Transfusion with Flow-Matching

Hey, we're implementing the Transfusion model outlined in this META paper: [Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model](https://arxiv.org/pdf/2408.11039) and [Flow Matching Guide and Code](https://arxiv.org/pdf/2412.06264)
We're experiencing these errors. Any suggestions on how to resolve them are welcome. :b. Thank you all in advance.

Errors we are having when trying to train the model:
```
Pérdida de flow inválida, usando valor por defecto
Época 1/10:   0%|                                                                                   | 0/238 [00:01<?, ?it/s, loss=nan, lm_loss=nan, fm_loss=0.0167, lr=0.00e+00]2025-04-20 21:57:02,961 - INFO - Generando muestras en paso 0...
Forward recibiendo 1 latentes de imagen
Advertencia: Número de parches no coincide - recibidos: 1, esperados: 16
Forward recibiendo 1 latentes de imagen
Advertencia: Número de parches no coincide - recibidos: 1, esperados: 16
Forward recibiendo 1 latentes de imagen
Advertencia: Número de parches no coincide - recibidos: 1, esperados: 16
Forward recibiendo 1 latentes de imagen
Advertencia: Número de parches no coincide - recibidos: 1, esperados: 16
Forward recibiendo 1 latentes de imagen
Advertencia: Número de parches no coincide - recibidos: 1, esperados: 16
Forward recibiendo 1 latentes de imagen
Advertencia: Número de parches no coincide - recibidos: 1, esperados: 16
Forward recibiendo 1 latentes de imagen
Advertencia: Número de parches no coincide - recibidos: 1, esperados: 16
Forward recibiendo 1 latentes de imagen
Advertencia: Número de parches no coincide - recibidos: 1, esperados: 16
Forward recibiendo 1 latentes de imagen
Advertencia: Número de parches no coincide - recibidos: 1, esperados: 16
Forward recibiendo 1 latentes de imagen
Advertencia: Número de parches no coincide - recibidos: 1, esperados: 16
Forward recibiendo 1 latentes de imagen
Advertencia: Número de parches no coincide - recibidos: 1, esperados: 16
Forward recibiendo 1 latentes de imagen
Advertencia: Número de parches no coincide - recibidos: 1, esperados: 16
Forward recibiendo 1 latentes de imagen
Advertencia: Número de parches no coincide - recibidos: 1, esperados: 16
Forward recibiendo 1 latentes de imagen
Advertencia: Número de parches no coincide - recibidos: 1, esperados: 16
Forward recibiendo 1 latentes de imagen
Advertencia: Número de parches no coincide - recibidos: 1, esperados: 16
Forward recibiendo 1 latentes de imagen
Advertencia: Número de parches no coincide - recibidos: 1, esperados: 16
Forward recibiendo 1 latentes de imagen
Advertencia: Número de parches no coincide - recibidos: 1, esperados: 16
Forward recibiendo 1 latentes de imagen
Advertencia: Número de parches no coincide - recibidos: 1, esperados: 16
Forward recibiendo 1 latentes de imagen
Advertencia: Número de parches no coincide - recibidos: 1, esperados: 16
Forward recibiendo 1 latentes de imagen
Advertencia: Número de parches no coincide - recibidos: 1, esperados: 16
Forward recibiendo 1 latentes de imagen
Advertencia: Número de parches no coincide - recibidos: 1, esperados: 16
Forward recibiendo 1 latentes de imagen
Advertencia: Número de parches no coincide - recibidos: 1, esperados: 16
Forward recibiendo 1 latentes de imagen
Advertencia: Número de parches no coincide - recibidos: 1, esperados: 16
Forward recibiendo 1 latentes de imagen
Advertencia: Número de parches no coincide - recibidos: 1, esperados: 16
Forward recibiendo 1 latentes de imagen
Advertencia: Número de parches no coincide - recibidos: 1, esperados: 16
/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torchvision/transforms/functional.py:282: RuntimeWarning: invalid value encountered in cast
  npimg = (npimg * 255).astype(np.uint8)
2025-04-20 21:57:05,842 - INFO - Checkpoint guardado en paso 0
2025-04-20 21:57:05,843 - INFO - Ejecutando evaluación...
                                                                                                                                                                                Imagen 3274691778_94bb57bba3.jpg.pt: calculados 4 parches necesarios                                                                                     | 0/13 [00:00<?, ?it/s]
Espacio para parches: 4, requerido: 4
Imagen 86542183_5e312ae4d4.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 2567812221_30fb64f5e9.jpg.pt: calculados 4 parches necesarios
Imagen 3605100550_01214a1224.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Espacio para parches: 4, requerido: 4
Imagen 3243588540_b418ac7eda.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 2090327868_9f99e2740d.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 2423138514_950f79e432.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 2709683703_5385ea9ef4.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 241346402_5c070a0c6d.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 1433142189_cda8652603.jpg.pt: calculados 4 parches necesarios
Imagen 470887781_faae5dae83.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Espacio para parches: 4, requerido: 4
Imagen 2623930900_b9df917b82.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 3134341610_3c55e373a7.jpg.pt: calculados 4 parches necesarios
Imagen 3262647146_a53770a21d.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Espacio para parches: 4, requerido: 4
Imagen 2505465055_f1e6cf9b76.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 1163282319_b729b24c46.jpg.pt: calculados 4 parches necesarios
Imagen 3336065481_2c21e622c8.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Espacio para parches: 4, requerido: 4
Imagen 3357708906_fb3a54dd78.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 1056873310_49c665eb22.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 2513260012_03d33305cf.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 1434607942_da5432c28c.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 3274691778_94bb57bba3.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 3243588540_b418ac7eda.jpg.pt: calculados 4 parches necesarios
Imagen 1355833561_9c43073eda.jpg.pt: calculados 4 parches necesarios
Imagen 3247052319_da8aba1983.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Espacio para parches: 4, requerido: 4
Imagen 241346402_5c070a0c6d.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 2567812221_30fb64f5e9.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 2623930900_b9df917b82.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 2879241506_b421536330.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 142802798_962a4ec5ce.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 3638992163_a085cc0c24.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 2709683703_5385ea9ef4.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 3262647146_a53770a21d.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Espacio para parches: 4, requerido: 4
Imagen 2739331794_4ae78f69a0.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 3533394378_1513ec90db.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 3576741633_671340544c.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 2647229826_e0e0c65ef1.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 470887781_faae5dae83.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 2455286250_fb6a66175a.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 2505465055_f1e6cf9b76.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Token BOI: 128256, EOI: 128257
Batch 0 tiene 1 tokens BOI en posiciones tensor([6], device='cuda:0')
Batch 0 tiene 1 tokens EOI en posiciones tensor([11], device='cuda:0')
Batch 1 tiene 1 tokens BOI en posiciones tensor([8], device='cuda:0')
Batch 1 tiene 1 tokens EOI en posiciones tensor([13], device='cuda:0')
Batch 2 tiene 1 tokens BOI en posiciones tensor([17], device='cuda:0')
Batch 2 tiene 1 tokens EOI en posiciones tensor([22], device='cuda:0')
Batch 3 tiene 1 tokens BOI en posiciones tensor([15], device='cuda:0')
Batch 3 tiene 1 tokens EOI en posiciones tensor([20], device='cuda:0')
Batch 4 tiene 1 tokens BOI en posiciones tensor([17], device='cuda:0')
Batch 4 tiene 1 tokens EOI en posiciones tensor([22], device='cuda:0')
Posiciones de imágenes encontradas: [(0, 6, 11), (1, 8, 13), (2, 17, 22), (3, 15, 20), (4, 17, 22)]
x_t shape: torch.Size([5, 8, 4, 4]), dx_t shape: torch.Size([5, 8, 4, 4])
batch_idx_map: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
Tensor en posición [0, 7:11] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([5, 8, 4, 4]), flow_idx: 0
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [1, 9:13] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([5, 8, 4, 4]), flow_idx: 1
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [2, 18:22] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([5, 8, 4, 4]), flow_idx: 2
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [3, 16:20] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([5, 8, 4, 4]), flow_idx: 3
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [4, 18:22] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([5, 8, 4, 4]), flow_idx: 4
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Pérdida de flow inválida, usando valor por defecto
                                                                                                                                                                                Imagen 3336065481_2c21e622c8.jpg.pt: calculados 4 parches necesarios                                                                             | 1/13 [00:00<00:03,  3.26it/s]
Espacio para parches: 4, requerido: 4
Imagen 2090327868_9f99e2740d.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 2603792708_18a97bac97.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Token BOI: 128256, EOI: 128257
Batch 0 tiene 1 tokens BOI en posiciones tensor([12], device='cuda:0')
Batch 0 tiene 1 tokens EOI en posiciones tensor([17], device='cuda:0')
Batch 3 tiene 1 tokens BOI en posiciones tensor([14], device='cuda:0')
Batch 3 tiene 1 tokens EOI en posiciones tensor([19], device='cuda:0')
Batch 4 tiene 1 tokens BOI en posiciones tensor([16], device='cuda:0')
Batch 4 tiene 1 tokens EOI en posiciones tensor([21], device='cuda:0')
Batch 5 tiene 1 tokens BOI en posiciones tensor([7], device='cuda:0')
Batch 5 tiene 1 tokens EOI en posiciones tensor([12], device='cuda:0')
Posiciones de imágenes encontradas: [(0, 12, 17), (3, 14, 19), (4, 16, 21), (5, 7, 12)]
x_t shape: torch.Size([4, 8, 4, 4]), dx_t shape: torch.Size([4, 8, 4, 4])
batch_idx_map: {0: 0, 3: 1, 4: 2, 5: 3}
Tensor en posición [0, 13:17] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 0
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [3, 15:19] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 1
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [4, 17:21] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 2
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [5, 8:12] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 3
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Pérdida de flow inválida, usando valor por defecto
                                                                                                                                                                                Imagen 2318721455_80c6644441.jpg.pt: calculados 4 parches necesarios                                                                             | 2/13 [00:00<00:02,  4.73it/s]
Espacio para parches: 4, requerido: 4
Imagen 3094064787_aed1666fc9.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Token BOI: 128256, EOI: 128257
Batch 0 tiene 1 tokens BOI en posiciones tensor([13], device='cuda:0')
Batch 0 tiene 1 tokens EOI en posiciones tensor([18], device='cuda:0')
Batch 2 tiene 1 tokens BOI en posiciones tensor([15], device='cuda:0')
Batch 2 tiene 1 tokens EOI en posiciones tensor([20], device='cuda:0')
Batch 3 tiene 1 tokens BOI en posiciones tensor([13], device='cuda:0')
Batch 3 tiene 1 tokens EOI en posiciones tensor([18], device='cuda:0')
Batch 5 tiene 1 tokens BOI en posiciones tensor([8], device='cuda:0')
Batch 5 tiene 1 tokens EOI en posiciones tensor([13], device='cuda:0')
Batch 6 tiene 1 tokens BOI en posiciones tensor([8], device='cuda:0')
Batch 6 tiene 1 tokens EOI en posiciones tensor([13], device='cuda:0')
Posiciones de imágenes encontradas: [(0, 13, 18), (2, 15, 20), (3, 13, 18), (5, 8, 13), (6, 8, 13)]
x_t shape: torch.Size([5, 8, 4, 4]), dx_t shape: torch.Size([5, 8, 4, 4])
batch_idx_map: {0: 0, 2: 1, 3: 2, 5: 3, 6: 4}
Tensor en posición [0, 14:18] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([5, 8, 4, 4]), flow_idx: 0
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [2, 16:20] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([5, 8, 4, 4]), flow_idx: 1
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [3, 14:18] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([5, 8, 4, 4]), flow_idx: 2
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [5, 9:13] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([5, 8, 4, 4]), flow_idx: 3
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [6, 9:13] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([5, 8, 4, 4]), flow_idx: 4
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Pérdida de flow inválida, usando valor por defecto
                                                                                                                                                                                Imagen 3533394378_1513ec90db.jpg.pt: calculados 4 parches necesarios                                                                             | 3/13 [00:00<00:01,  5.61it/s]
Espacio para parches: 4, requerido: 4
Imagen 3576741633_671340544c.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 3228960484_9aab98b91a.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 2238759450_6475641bdb.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 2833431496_09d999db4d.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Token BOI: 128256, EOI: 128257
Batch 0 tiene 1 tokens BOI en posiciones tensor([21], device='cuda:0')
Batch 0 tiene 1 tokens EOI en posiciones tensor([26], device='cuda:0')
Batch 4 tiene 1 tokens BOI en posiciones tensor([17], device='cuda:0')
Batch 4 tiene 1 tokens EOI en posiciones tensor([22], device='cuda:0')
Batch 5 tiene 1 tokens BOI en posiciones tensor([10], device='cuda:0')
Batch 5 tiene 1 tokens EOI en posiciones tensor([15], device='cuda:0')
Batch 7 tiene 1 tokens BOI en posiciones tensor([11], device='cuda:0')
Batch 7 tiene 1 tokens EOI en posiciones tensor([16], device='cuda:0')
Posiciones de imágenes encontradas: [(0, 21, 26), (4, 17, 22), (5, 10, 15), (7, 11, 16)]
x_t shape: torch.Size([4, 8, 4, 4]), dx_t shape: torch.Size([4, 8, 4, 4])
batch_idx_map: {0: 0, 4: 1, 5: 2, 7: 3}
Tensor en posición [0, 22:26] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 0
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [4, 18:22] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 1
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [5, 11:15] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 2
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [7, 12:16] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 3
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Pérdida de flow inválida, usando valor por defecto
                                                                                                                                                                                Token BOI: 128256, EOI: 128257█████████████████████████▋                                                                                         | 4/13 [00:00<00:01,  6.18it/s]
Batch 0 tiene 1 tokens BOI en posiciones tensor([15], device='cuda:0')
Batch 0 tiene 1 tokens EOI en posiciones tensor([20], device='cuda:0')
Batch 5 tiene 1 tokens BOI en posiciones tensor([14], device='cuda:0')
Batch 5 tiene 1 tokens EOI en posiciones tensor([19], device='cuda:0')
Batch 6 tiene 1 tokens BOI en posiciones tensor([13], device='cuda:0')
Batch 6 tiene 1 tokens EOI en posiciones tensor([18], device='cuda:0')
Batch 7 tiene 1 tokens BOI en posiciones tensor([9], device='cuda:0')
Batch 7 tiene 1 tokens EOI en posiciones tensor([14], device='cuda:0')
Posiciones de imágenes encontradas: [(0, 15, 20), (5, 14, 19), (6, 13, 18), (7, 9, 14)]
x_t shape: torch.Size([4, 8, 4, 4]), dx_t shape: torch.Size([4, 8, 4, 4])
batch_idx_map: {0: 0, 5: 1, 6: 2, 7: 3}
Tensor en posición [0, 16:20] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 0
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [5, 15:19] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 1
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [6, 14:18] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 2
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [7, 10:14] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 3
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Pérdida de flow inválida, usando valor por defecto
                                                                                                                                                                                Token BOI: 128256, EOI: 128257███████████████████████████████████▌                                                                               | 5/13 [00:00<00:01,  6.54it/s]
Batch 3 tiene 1 tokens BOI en posiciones tensor([14], device='cuda:0')
Batch 3 tiene 1 tokens EOI en posiciones tensor([19], device='cuda:0')
Batch 6 tiene 1 tokens BOI en posiciones tensor([15], device='cuda:0')
Batch 6 tiene 1 tokens EOI en posiciones tensor([20], device='cuda:0')
Posiciones de imágenes encontradas: [(3, 14, 19), (6, 15, 20)]
x_t shape: torch.Size([2, 8, 4, 4]), dx_t shape: torch.Size([2, 8, 4, 4])
batch_idx_map: {3: 0, 6: 1}
Tensor en posición [3, 15:19] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([2, 8, 4, 4]), flow_idx: 0
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [6, 16:20] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([2, 8, 4, 4]), flow_idx: 1
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Pérdida de flow inválida, usando valor por defecto
                                                                                                                                                                                Token BOI: 128256, EOI: 128257█████████████████████████████████████████████▌                                                                     | 6/13 [00:00<00:01,  6.81it/s]
Batch 0 tiene 1 tokens BOI en posiciones tensor([7], device='cuda:0')
Batch 0 tiene 1 tokens EOI en posiciones tensor([12], device='cuda:0')
Batch 2 tiene 1 tokens BOI en posiciones tensor([6], device='cuda:0')
Batch 2 tiene 1 tokens EOI en posiciones tensor([11], device='cuda:0')
Batch 4 tiene 1 tokens BOI en posiciones tensor([17], device='cuda:0')
Batch 4 tiene 1 tokens EOI en posiciones tensor([22], device='cuda:0')
Batch 5 tiene 1 tokens BOI en posiciones tensor([15], device='cuda:0')
Batch 5 tiene 1 tokens EOI en posiciones tensor([20], device='cuda:0')
Batch 6 tiene 1 tokens BOI en posiciones tensor([17], device='cuda:0')
Batch 6 tiene 1 tokens EOI en posiciones tensor([22], device='cuda:0')
Batch 7 tiene 1 tokens BOI en posiciones tensor([9], device='cuda:0')
Batch 7 tiene 1 tokens EOI en posiciones tensor([14], device='cuda:0')
Posiciones de imágenes encontradas: [(0, 7, 12), (2, 6, 11), (4, 17, 22), (5, 15, 20), (6, 17, 22), (7, 9, 14)]
x_t shape: torch.Size([6, 8, 4, 4]), dx_t shape: torch.Size([6, 8, 4, 4])
batch_idx_map: {0: 0, 2: 1, 4: 2, 5: 3, 6: 4, 7: 5}
Tensor en posición [0, 8:12] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([6, 8, 4, 4]), flow_idx: 0
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [2, 7:11] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([6, 8, 4, 4]), flow_idx: 1
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [4, 18:22] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([6, 8, 4, 4]), flow_idx: 2
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [5, 16:20] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([6, 8, 4, 4]), flow_idx: 3
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [6, 18:22] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([6, 8, 4, 4]), flow_idx: 4
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [7, 10:14] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([6, 8, 4, 4]), flow_idx: 5
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Pérdida de flow inválida, usando valor por defecto
                                                                                                                                                                                Token BOI: 128256, EOI: 128257███████████████████████████████████████████████████████▍                                                           | 7/13 [00:01<00:00,  6.98it/s]
Batch 0 tiene 1 tokens BOI en posiciones tensor([12], device='cuda:0')
Batch 0 tiene 1 tokens EOI en posiciones tensor([17], device='cuda:0')
Batch 2 tiene 1 tokens BOI en posiciones tensor([12], device='cuda:0')
Batch 2 tiene 1 tokens EOI en posiciones tensor([17], device='cuda:0')
Batch 3 tiene 1 tokens BOI en posiciones tensor([16], device='cuda:0')
Batch 3 tiene 1 tokens EOI en posiciones tensor([21], device='cuda:0')
Batch 4 tiene 1 tokens BOI en posiciones tensor([12], device='cuda:0')
Batch 4 tiene 1 tokens EOI en posiciones tensor([17], device='cuda:0')
Batch 5 tiene 1 tokens BOI en posiciones tensor([14], device='cuda:0')
Batch 5 tiene 1 tokens EOI en posiciones tensor([19], device='cuda:0')
Batch 7 tiene 1 tokens BOI en posiciones tensor([7], device='cuda:0')
Batch 7 tiene 1 tokens EOI en posiciones tensor([12], device='cuda:0')
Posiciones de imágenes encontradas: [(0, 12, 17), (2, 12, 17), (3, 16, 21), (4, 12, 17), (5, 14, 19), (7, 7, 12)]
x_t shape: torch.Size([6, 8, 4, 4]), dx_t shape: torch.Size([6, 8, 4, 4])
batch_idx_map: {0: 0, 2: 1, 3: 2, 4: 3, 5: 4, 7: 5}
Tensor en posición [0, 13:17] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([6, 8, 4, 4]), flow_idx: 0
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [2, 13:17] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([6, 8, 4, 4]), flow_idx: 1
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [3, 17:21] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([6, 8, 4, 4]), flow_idx: 2
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [4, 13:17] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([6, 8, 4, 4]), flow_idx: 3
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [5, 15:19] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([6, 8, 4, 4]), flow_idx: 4
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [7, 8:12] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([6, 8, 4, 4]), flow_idx: 5
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Pérdida de flow inválida, usando valor por defecto
                                                                                                                                                                                Token BOI: 128256, EOI: 128257█████████████████████████████████████████████████████████████████▍                                                 | 8/13 [00:01<00:00,  7.15it/s]
Batch 3 tiene 1 tokens BOI en posiciones tensor([18], device='cuda:0')
Batch 3 tiene 1 tokens EOI en posiciones tensor([23], device='cuda:0')
Batch 5 tiene 1 tokens BOI en posiciones tensor([13], device='cuda:0')
Batch 5 tiene 1 tokens EOI en posiciones tensor([18], device='cuda:0')
Batch 6 tiene 1 tokens BOI en posiciones tensor([17], device='cuda:0')
Batch 6 tiene 1 tokens EOI en posiciones tensor([22], device='cuda:0')
Batch 7 tiene 1 tokens BOI en posiciones tensor([8], device='cuda:0')
Batch 7 tiene 1 tokens EOI en posiciones tensor([13], device='cuda:0')
Posiciones de imágenes encontradas: [(3, 18, 23), (5, 13, 18), (6, 17, 22), (7, 8, 13)]
x_t shape: torch.Size([4, 8, 4, 4]), dx_t shape: torch.Size([4, 8, 4, 4])
batch_idx_map: {3: 0, 5: 1, 6: 2, 7: 3}
Tensor en posición [3, 19:23] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 0
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [5, 14:18] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 1
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [6, 18:22] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 2
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [7, 9:13] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 3
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Pérdida de flow inválida, usando valor por defecto
                                                                                                                                                                                Token BOI: 128256, EOI: 128257███████████████████████████████████████████████████████████████████████████▎                                       | 9/13 [00:01<00:00,  7.33it/s]
Batch 0 tiene 1 tokens BOI en posiciones tensor([8], device='cuda:0')
Batch 0 tiene 1 tokens EOI en posiciones tensor([13], device='cuda:0')
Batch 2 tiene 1 tokens BOI en posiciones tensor([21], device='cuda:0')
Batch 2 tiene 1 tokens EOI en posiciones tensor([26], device='cuda:0')
Batch 5 tiene 1 tokens BOI en posiciones tensor([10], device='cuda:0')
Batch 5 tiene 1 tokens EOI en posiciones tensor([15], device='cuda:0')
Posiciones de imágenes encontradas: [(0, 8, 13), (2, 21, 26), (5, 10, 15)]
x_t shape: torch.Size([3, 8, 4, 4]), dx_t shape: torch.Size([3, 8, 4, 4])
batch_idx_map: {0: 0, 2: 1, 5: 2}
Tensor en posición [0, 9:13] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([3, 8, 4, 4]), flow_idx: 0
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [2, 22:26] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([3, 8, 4, 4]), flow_idx: 1
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [5, 11:15] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([3, 8, 4, 4]), flow_idx: 2
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Pérdida de flow inválida, usando valor por defecto
                                                                                                                                                                                Token BOI: 128256, EOI: 128257████████████████████████████████████████████████████████████████████████████████████▍                             | 10/13 [00:01<00:00,  7.31it/s]
Batch 0 tiene 1 tokens BOI en posiciones tensor([7], device='cuda:0')
Batch 0 tiene 1 tokens EOI en posiciones tensor([12], device='cuda:0')
Batch 4 tiene 1 tokens BOI en posiciones tensor([7], device='cuda:0')
Batch 4 tiene 1 tokens EOI en posiciones tensor([12], device='cuda:0')
Posiciones de imágenes encontradas: [(0, 7, 12), (4, 7, 12)]
x_t shape: torch.Size([2, 8, 4, 4]), dx_t shape: torch.Size([2, 8, 4, 4])
batch_idx_map: {0: 0, 4: 1}
Tensor en posición [0, 8:12] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([2, 8, 4, 4]), flow_idx: 0
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [4, 8:12] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([2, 8, 4, 4]), flow_idx: 1
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Pérdida de flow inválida, usando valor por defecto
                                                                                                                                                                                Token BOI: 128256, EOI: 128257██████████████████████████████████████████████████████████████████████████████████████████████▎                   | 11/13 [00:01<00:00,  7.38it/s]
Batch 0 tiene 1 tokens BOI en posiciones tensor([13], device='cuda:0')
Batch 0 tiene 1 tokens EOI en posiciones tensor([18], device='cuda:0')
Batch 1 tiene 1 tokens BOI en posiciones tensor([9], device='cuda:0')
Batch 1 tiene 1 tokens EOI en posiciones tensor([14], device='cuda:0')
Batch 2 tiene 1 tokens BOI en posiciones tensor([14], device='cuda:0')
Batch 2 tiene 1 tokens EOI en posiciones tensor([19], device='cuda:0')
Batch 4 tiene 1 tokens BOI en posiciones tensor([9], device='cuda:0')
Batch 4 tiene 1 tokens EOI en posiciones tensor([14], device='cuda:0')
Batch 6 tiene 1 tokens BOI en posiciones tensor([16], device='cuda:0')
Batch 6 tiene 1 tokens EOI en posiciones tensor([21], device='cuda:0')
Posiciones de imágenes encontradas: [(0, 13, 18), (1, 9, 14), (2, 14, 19), (4, 9, 14), (6, 16, 21)]
x_t shape: torch.Size([5, 8, 4, 4]), dx_t shape: torch.Size([5, 8, 4, 4])
batch_idx_map: {0: 0, 1: 1, 2: 2, 4: 3, 6: 4}
Tensor en posición [0, 14:18] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([5, 8, 4, 4]), flow_idx: 0
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [1, 10:14] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([5, 8, 4, 4]), flow_idx: 1
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [2, 15:19] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([5, 8, 4, 4]), flow_idx: 2
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [4, 10:14] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([5, 8, 4, 4]), flow_idx: 3
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [6, 17:21] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([5, 8, 4, 4]), flow_idx: 4
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Pérdida de flow inválida, usando valor por defecto
Evaluando: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:01<00:00,  6.80it/s]
2025-04-20 21:57:07,757 - INFO - Pérdida de validación: nan, Pérdida LM: nan, Pérdida FM: 0.0923██████████████████████████████████████▏         | 12/13 [00:01<00:00,  7.36it/s]
Época 1/10:   0%|▎                                                                          | 1/238 [00:06<24:11,  6.12s/it, loss=nan, lm_loss=nan, fm_loss=0.0167, lr=0.00e+00]Procesando batch con 4 imágenes
BOI índices: [13, 9, 19, 11]
Creado image_latents_dict con 4 entradas
Claves en image_latents_dict: [13, 9, 19, 11]
Imagen 3458577912_67db47209d.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 2650620212_0586016e0d.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 300148649_72f7f0399c.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Token BOI: 128256, EOI: 128257
Batch 0 tiene 1 tokens BOI en posiciones tensor([13], device='cuda:0')
Batch 0 tiene 1 tokens EOI en posiciones tensor([18], device='cuda:0')
Batch 1 tiene 1 tokens BOI en posiciones tensor([9], device='cuda:0')
Batch 1 tiene 1 tokens EOI en posiciones tensor([14], device='cuda:0')
Batch 5 tiene 1 tokens BOI en posiciones tensor([19], device='cuda:0')
Batch 5 tiene 1 tokens EOI en posiciones tensor([24], device='cuda:0')
Batch 6 tiene 1 tokens BOI en posiciones tensor([11], device='cuda:0')
Batch 6 tiene 1 tokens EOI en posiciones tensor([16], device='cuda:0')
Posiciones de imágenes encontradas: [(0, 13, 18), (1, 9, 14), (5, 19, 24), (6, 11, 16)]
x_t shape: torch.Size([4, 8, 4, 4]), dx_t shape: torch.Size([4, 8, 4, 4])
batch_idx_map: {0: 0, 1: 1, 5: 2, 6: 3}
Tensor en posición [0, 14:18] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 0
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [1, 10:14] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 1
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [5, 20:24] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 2
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [6, 12:16] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 3
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Pérdida de flow inválida, usando valor por defecto
Época 1/10:   1%|▋                                                                            | 2/238 [00:06<11:03,  2.81s/it, loss=nan, lm_loss=nan, fm_loss=0.05, lr=3.00e-07]Procesando batch con 4 imágenes
BOI índices: [10, 14, 17, 12]
Creado image_latents_dict con 4 entradas
Claves en image_latents_dict: [10, 14, 17, 12]
Imagen 461019788_bc0993dabd.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 3253060519_55d98c208f.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 3187924573_203223e6c0.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 496606439_9333831e73.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 359082432_c1fd5aa2d6.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Token BOI: 128256, EOI: 128257
Batch 0 tiene 1 tokens BOI en posiciones tensor([10], device='cuda:0')
Batch 0 tiene 1 tokens EOI en posiciones tensor([15], device='cuda:0')
Batch 1 tiene 1 tokens BOI en posiciones tensor([14], device='cuda:0')
Batch 1 tiene 1 tokens EOI en posiciones tensor([19], device='cuda:0')
Batch 4 tiene 1 tokens BOI en posiciones tensor([17], device='cuda:0')
Batch 4 tiene 1 tokens EOI en posiciones tensor([22], device='cuda:0')
Batch 6 tiene 1 tokens BOI en posiciones tensor([12], device='cuda:0')
Batch 6 tiene 1 tokens EOI en posiciones tensor([17], device='cuda:0')
Posiciones de imágenes encontradas: [(0, 10, 15), (1, 14, 19), (4, 17, 22), (6, 12, 17)]
x_t shape: torch.Size([4, 8, 4, 4]), dx_t shape: torch.Size([4, 8, 4, 4])
batch_idx_map: {0: 0, 1: 1, 4: 2, 6: 3}
Tensor en posición [0, 11:15] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 0
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [1, 15:19] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 1
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [4, 18:22] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 2
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [6, 13:17] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 3
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Pérdida de flow inválida, usando valor por defecto
Época 1/10:   1%|▉                                                                           | 3/238 [00:07<06:44,  1.72s/it, loss=nan, lm_loss=nan, fm_loss=0.075, lr=6.00e-07]Procesando batch con 3 imágenes
BOI índices: [12, 15, 4]
Creado image_latents_dict con 3 entradas
Claves en image_latents_dict: [12, 15, 4]
Imagen 61209225_8512e1dad5.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 3307667255_26bede91eb.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 2692635048_16c279ff9e.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 3110614694_fecc23ca65.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 533979933_a95b03323b.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 3225478803_f7a9a41a1d.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Token BOI: 128256, EOI: 128257
Batch 3 tiene 1 tokens BOI en posiciones tensor([12], device='cuda:0')
Batch 3 tiene 1 tokens EOI en posiciones tensor([17], device='cuda:0')
Batch 4 tiene 1 tokens BOI en posiciones tensor([15], device='cuda:0')
Batch 4 tiene 1 tokens EOI en posiciones tensor([20], device='cuda:0')
Batch 7 tiene 1 tokens BOI en posiciones tensor([4], device='cuda:0')
Batch 7 tiene 1 tokens EOI en posiciones tensor([9], device='cuda:0')
Posiciones de imágenes encontradas: [(3, 12, 17), (4, 15, 20), (7, 4, 9)]
x_t shape: torch.Size([3, 8, 4, 4]), dx_t shape: torch.Size([3, 8, 4, 4])
batch_idx_map: {3: 0, 4: 1, 7: 2}
Tensor en posición [3, 13:17] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([3, 8, 4, 4]), flow_idx: 0
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [4, 16:20] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([3, 8, 4, 4]), flow_idx: 1
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [7, 5:9] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([3, 8, 4, 4]), flow_idx: 2
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Pérdida de flow inválida, usando valor por defecto
Época 1/10:   2%|█▎                                                                          | 4/238 [00:07<04:36,  1.18s/it, loss=nan, lm_loss=nan, fm_loss=0.133, lr=6.00e-07]Procesando batch con 4 imágenes
BOI índices: [17, 13, 9, 9]
Creado image_latents_dict con 3 entradas
Claves en image_latents_dict: [17, 13, 9]
Imagen 2921670682_6a77a6c3e9.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 512306469_1392697d32.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 3658016590_f761e72dc3.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 3534183988_3763593dfb.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 1357689954_72588dfdc4.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Token BOI: 128256, EOI: 128257
Batch 0 tiene 1 tokens BOI en posiciones tensor([17], device='cuda:0')
Batch 0 tiene 1 tokens EOI en posiciones tensor([22], device='cuda:0')
Batch 1 tiene 1 tokens BOI en posiciones tensor([13], device='cuda:0')
Batch 1 tiene 1 tokens EOI en posiciones tensor([18], device='cuda:0')
Batch 2 tiene 1 tokens BOI en posiciones tensor([9], device='cuda:0')
Batch 2 tiene 1 tokens EOI en posiciones tensor([14], device='cuda:0')
Batch 6 tiene 1 tokens BOI en posiciones tensor([9], device='cuda:0')
Batch 6 tiene 1 tokens EOI en posiciones tensor([14], device='cuda:0')
Posiciones de imágenes encontradas: [(0, 17, 22), (1, 13, 18), (2, 9, 14), (6, 9, 14)]
x_t shape: torch.Size([4, 8, 4, 4]), dx_t shape: torch.Size([4, 8, 4, 4])
batch_idx_map: {0: 0, 1: 1, 2: 2, 6: 3}
Tensor en posición [0, 18:22] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 0
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [1, 14:18] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 1
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [2, 10:14] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 2
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [6, 10:14] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 3
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Pérdida de flow inválida, usando valor por defecto
Época 1/10:   2%|█▌                                                                          | 5/238 [00:07<03:32,  1.10it/s, loss=nan, lm_loss=nan, fm_loss=0.125, lr=9.00e-07]Procesando batch con 1 imágenes
BOI índices: [9]
Imagen 2439384468_58934deab6.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 487894806_352d9b5e66.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 3099694681_19a72c8bdc.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 3593556797_46b49a02a8.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 269898095_d00ac7d7a4.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Creado image_latents_dict con 1 entradas
Claves en image_latents_dict: [9]
Token BOI: 128256, EOI: 128257
Batch 1 tiene 1 tokens BOI en posiciones tensor([9], device='cuda:0')
Batch 1 tiene 1 tokens EOI en posiciones tensor([14], device='cuda:0')
Posiciones de imágenes encontradas: [(1, 9, 14)]
x_t shape: torch.Size([1, 8, 4, 4]), dx_t shape: torch.Size([1, 8, 4, 4])
batch_idx_map: {1: 0}
Tensor en posición [1, 10:14] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([1, 8, 4, 4]), flow_idx: 0
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Pérdida de flow inválida, usando valor por defecto
Época 1/10:   3%|██                                                                              | 6/238 [00:08<02:48,  1.38it/s, loss=nan, lm_loss=nan, fm_loss=0, lr=9.00e-07]Procesando batch con 2 imágenes
BOI índices: [8, 7]
Creado image_latents_dict con 2 entradas
Claves en image_latents_dict: [8, 7]
Imagen 267015208_d80b3eb94d.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 3014080715_f4f0dbb56e.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 2176147758_9a8deba576.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 1433088025_bce2cb69f8.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 2419221084_01a14176b4.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Token BOI: 128256, EOI: 128257
Batch 5 tiene 1 tokens BOI en posiciones tensor([8], device='cuda:0')
Batch 5 tiene 1 tokens EOI en posiciones tensor([13], device='cuda:0')
Batch 6 tiene 1 tokens BOI en posiciones tensor([7], device='cuda:0')
Batch 6 tiene 1 tokens EOI en posiciones tensor([12], device='cuda:0')
Posiciones de imágenes encontradas: [(5, 8, 13), (6, 7, 12)]
x_t shape: torch.Size([2, 8, 4, 4]), dx_t shape: torch.Size([2, 8, 4, 4])
batch_idx_map: {5: 0, 6: 1}
Tensor en posición [5, 9:13] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([2, 8, 4, 4]), flow_idx: 0
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [6, 8:12] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([2, 8, 4, 4]), flow_idx: 1
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Pérdida de flow inválida, usando valor por defecto
Época 1/10:   3%|██▎                                                                          | 7/238 [00:08<02:20,  1.64it/s, loss=nan, lm_loss=nan, fm_loss=0.35, lr=9.00e-07]Procesando batch con 2 imágenes
BOI índices: [17, 17]
Creado image_latents_dict con 1 entradas
Claves en image_latents_dict: [17]
Imagen 241346471_c756a8f139.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 3477369101_8e0c61d8f4.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 824923476_d85edce294.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 269898095_d00ac7d7a4.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Token BOI: 128256, EOI: 128257
Batch 1 tiene 1 tokens BOI en posiciones tensor([17], device='cuda:0')
Batch 1 tiene 1 tokens EOI en posiciones tensor([22], device='cuda:0')
Batch 4 tiene 1 tokens BOI en posiciones tensor([17], device='cuda:0')
Batch 4 tiene 1 tokens EOI en posiciones tensor([22], device='cuda:0')
Posiciones de imágenes encontradas: [(1, 17, 22), (4, 17, 22)]
x_t shape: torch.Size([2, 8, 4, 4]), dx_t shape: torch.Size([2, 8, 4, 4])
batch_idx_map: {1: 0, 4: 1}
Tensor en posición [1, 18:22] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([2, 8, 4, 4]), flow_idx: 0
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [4, 18:22] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([2, 8, 4, 4]), flow_idx: 1
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Pérdida de flow inválida, usando valor por defecto
Época 1/10:   3%|██▌                                                                           | 8/238 [00:08<02:02,  1.88it/s, loss=nan, lm_loss=nan, fm_loss=0.4, lr=9.00e-07]Procesando batch con 2 imágenes
BOI índices: [7, 7]
Creado image_latents_dict con 1 entradas
Claves en image_latents_dict: [7]
Imagen 434938585_fbf913dfb4.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 3293596075_973b0bfd08.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 3582742297_1daa29968e.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Token BOI: 128256, EOI: 128257
Batch 1 tiene 1 tokens BOI en posiciones tensor([7], device='cuda:0')
Batch 1 tiene 1 tokens EOI en posiciones tensor([12], device='cuda:0')
Batch 5 tiene 1 tokens BOI en posiciones tensor([7], device='cuda:0')
Batch 5 tiene 1 tokens EOI en posiciones tensor([12], device='cuda:0')
Posiciones de imágenes encontradas: [(1, 7, 12), (5, 7, 12)]
x_t shape: torch.Size([2, 8, 4, 4]), dx_t shape: torch.Size([2, 8, 4, 4])
batch_idx_map: {1: 0, 5: 1}
Tensor en posición [1, 8:12] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([2, 8, 4, 4]), flow_idx: 0
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [5, 8:12] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([2, 8, 4, 4]), flow_idx: 1
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Pérdida de flow inválida, usando valor por defecto
Época 1/10:   4%|██▉                                                                          | 9/238 [00:09<01:49,  2.08it/s, loss=nan, lm_loss=nan, fm_loss=0.45, lr=9.00e-07]Procesando batch con 3 imágenes
BOI índices: [12, 8, 12]
Creado image_latents_dict con 2 entradas
Claves en image_latents_dict: [12, 8]
Imagen 3088399255_1bd9a6aa04.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 3084001782_41a848df4e.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 1510669311_75330b4781.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 2718049631_e7aa74cb9b.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Token BOI: 128256, EOI: 128257
Batch 0 tiene 1 tokens BOI en posiciones tensor([12], device='cuda:0')
Batch 0 tiene 1 tokens EOI en posiciones tensor([17], device='cuda:0')
Batch 2 tiene 1 tokens BOI en posiciones tensor([8], device='cuda:0')
Batch 2 tiene 1 tokens EOI en posiciones tensor([13], device='cuda:0')
Batch 7 tiene 1 tokens BOI en posiciones tensor([12], device='cuda:0')
Batch 7 tiene 1 tokens EOI en posiciones tensor([17], device='cuda:0')
Posiciones de imágenes encontradas: [(0, 12, 17), (2, 8, 13), (7, 12, 17)]
x_t shape: torch.Size([3, 8, 4, 4]), dx_t shape: torch.Size([3, 8, 4, 4])
batch_idx_map: {0: 0, 2: 1, 7: 2}
Tensor en posición [0, 13:17] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([3, 8, 4, 4]), flow_idx: 0
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [2, 9:13] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([3, 8, 4, 4]), flow_idx: 1
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [7, 13:17] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([3, 8, 4, 4]), flow_idx: 2
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Pérdida de flow inválida, usando valor por defecto
Época 1/10:   4%|███▏                                                                       | 10/238 [00:09<01:41,  2.24it/s, loss=nan, lm_loss=nan, fm_loss=0.333, lr=9.00e-07]Procesando batch con 5 imágenes
BOI índices: [10, 6, 16, 9, 17]
Creado image_latents_dict con 5 entradas
Claves en image_latents_dict: [10, 6, 16, 9, 17]
Imagen 2522297487_57edf117f7.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 3583321426_f373c52161.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 299572828_4b38b80d16.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 2709648336_15455e60b2.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Token BOI: 128256, EOI: 128257
Batch 0 tiene 1 tokens BOI en posiciones tensor([10], device='cuda:0')
Batch 0 tiene 1 tokens EOI en posiciones tensor([15], device='cuda:0')
Batch 1 tiene 1 tokens BOI en posiciones tensor([6], device='cuda:0')
Batch 1 tiene 1 tokens EOI en posiciones tensor([11], device='cuda:0')
Batch 3 tiene 1 tokens BOI en posiciones tensor([16], device='cuda:0')
Batch 3 tiene 1 tokens EOI en posiciones tensor([21], device='cuda:0')
Batch 5 tiene 1 tokens BOI en posiciones tensor([9], device='cuda:0')
Batch 5 tiene 1 tokens EOI en posiciones tensor([14], device='cuda:0')
Batch 7 tiene 1 tokens BOI en posiciones tensor([17], device='cuda:0')
Batch 7 tiene 1 tokens EOI en posiciones tensor([22], device='cuda:0')
Posiciones de imágenes encontradas: [(0, 10, 15), (1, 6, 11), (3, 16, 21), (5, 9, 14), (7, 17, 22)]
x_t shape: torch.Size([5, 8, 4, 4]), dx_t shape: torch.Size([5, 8, 4, 4])
batch_idx_map: {0: 0, 1: 1, 3: 2, 5: 3, 7: 4}
Tensor en posición [0, 11:15] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([5, 8, 4, 4]), flow_idx: 0
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [1, 7:11] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([5, 8, 4, 4]), flow_idx: 1
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [3, 17:21] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([5, 8, 4, 4]), flow_idx: 2
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [5, 10:14] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([5, 8, 4, 4]), flow_idx: 3
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [7, 18:22] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([5, 8, 4, 4]), flow_idx: 4
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Pérdida de flow inválida, usando valor por defecto
Época 1/10:   4%|███▏                                                                       | 10/238 [00:10<03:48,  1.00s/it, loss=nan, lm_loss=nan, fm_loss=0.333, lr=9.00e-07]

```