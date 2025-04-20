# Transfusion with Flow-Matching

Hey, we're implementing the Transfusion model outlined in this META paper: [Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model](https://arxiv.org/pdf/2408.11039) and [Flow Matching Guide and Code](https://arxiv.org/pdf/2412.06264)
We're experiencing these errors. Any suggestions on how to resolve them are welcome. :b. Thank you all in advance.

Errors we are having when trying to train the model:
```
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 2
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [7, 11:15] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 3
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Pérdida de flow inválida, usando valor por defecto
Época 1/10:  51%|██████████████████████████████████████▍                                    | 122/238 [00:48<00:46,  2.47it/s, loss=nan, lm_loss=nan, fm_loss=3.05, lr=1.14e-05]Procesando batch con 6 imágenes
BOI índices: [8, 12, 6, 9, 12, 13]
Creado image_latents_dict con 5 entradas
Claves en image_latents_dict: [8, 12, 6, 9, 13]
Imagen 1579206585_5ca6a24db0.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 2833820456_143ea6ce47.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 3527261343_efa07ea596.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Token BOI: 128256, EOI: 128257
Batch 0 tiene 1 tokens BOI en posiciones tensor([8], device='cuda:0')
Batch 0 tiene 1 tokens EOI en posiciones tensor([13], device='cuda:0')
Batch 1 tiene 1 tokens BOI en posiciones tensor([12], device='cuda:0')
Batch 1 tiene 1 tokens EOI en posiciones tensor([17], device='cuda:0')
Batch 2 tiene 1 tokens BOI en posiciones tensor([6], device='cuda:0')
Batch 2 tiene 1 tokens EOI en posiciones tensor([11], device='cuda:0')
Batch 5 tiene 1 tokens BOI en posiciones tensor([9], device='cuda:0')
Batch 5 tiene 1 tokens EOI en posiciones tensor([14], device='cuda:0')
Batch 6 tiene 1 tokens BOI en posiciones tensor([12], device='cuda:0')
Batch 6 tiene 1 tokens EOI en posiciones tensor([17], device='cuda:0')
Batch 7 tiene 1 tokens BOI en posiciones tensor([13], device='cuda:0')
Batch 7 tiene 1 tokens EOI en posiciones tensor([18], device='cuda:0')
Posiciones de imágenes encontradas: [(0, 8, 13), (1, 12, 17), (2, 6, 11), (5, 9, 14), (6, 12, 17), (7, 13, 18)]
x_t shape: torch.Size([6, 8, 4, 4]), dx_t shape: torch.Size([6, 8, 4, 4])
batch_idx_map: {0: 0, 1: 1, 2: 2, 5: 3, 6: 4, 7: 5}
Tensor en posición [0, 9:13] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([6, 8, 4, 4]), flow_idx: 0
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [1, 13:17] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([6, 8, 4, 4]), flow_idx: 1
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [2, 7:11] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([6, 8, 4, 4]), flow_idx: 2
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [5, 10:14] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([6, 8, 4, 4]), flow_idx: 3
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [6, 13:17] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([6, 8, 4, 4]), flow_idx: 4
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [7, 14:18] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([6, 8, 4, 4]), flow_idx: 5
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Pérdida de flow inválida, usando valor por defecto
Época 1/10:  52%|██████████████████████████████████████▊                                    | 123/238 [00:49<00:45,  2.52it/s, loss=nan, lm_loss=nan, fm_loss=2.05, lr=1.14e-05]Procesando batch con 4 imágenes
BOI índices: [14, 18, 8, 9]
Creado image_latents_dict con 4 entradas
Claves en image_latents_dict: [14, 18, 8, 9]
Imagen 2308978137_bfe776d541.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 3572144280_ea42bbd927.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 3423802527_94bd2b23b0.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 2982881046_45765ced2c.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 3402081035_a54cfab1d9.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Token BOI: 128256, EOI: 128257
Batch 0 tiene 1 tokens BOI en posiciones tensor([14], device='cuda:0')
Batch 0 tiene 1 tokens EOI en posiciones tensor([19], device='cuda:0')
Batch 3 tiene 1 tokens BOI en posiciones tensor([18], device='cuda:0')
Batch 3 tiene 1 tokens EOI en posiciones tensor([23], device='cuda:0')
Batch 4 tiene 1 tokens BOI en posiciones tensor([8], device='cuda:0')
Batch 4 tiene 1 tokens EOI en posiciones tensor([13], device='cuda:0')
Batch 7 tiene 1 tokens BOI en posiciones tensor([9], device='cuda:0')
Batch 7 tiene 1 tokens EOI en posiciones tensor([14], device='cuda:0')
Posiciones de imágenes encontradas: [(0, 14, 19), (3, 18, 23), (4, 8, 13), (7, 9, 14)]
x_t shape: torch.Size([4, 8, 4, 4]), dx_t shape: torch.Size([4, 8, 4, 4])
batch_idx_map: {0: 0, 3: 1, 4: 2, 7: 3}
Tensor en posición [0, 15:19] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 0
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [3, 19:23] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 1
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [4, 9:13] shape: torch.Size([4, 512])
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
Época 1/10:  52%|███████████████████████████████████████▌                                    | 124/238 [00:49<00:46,  2.46it/s, loss=nan, lm_loss=nan, fm_loss=3.1, lr=1.17e-05]Procesando batch con 3 imágenes
BOI índices: [14, 8, 21]
Creado image_latents_dict con 3 entradas
Claves en image_latents_dict: [14, 8, 21]
Imagen 262439544_e71cd26b24.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 3215847501_c723905ba4.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 534200447_b0f3ff02be.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 2351479551_e8820a1ff3.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 3207343907_995f7ac1d2.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Token BOI: 128256, EOI: 128257
Batch 2 tiene 1 tokens BOI en posiciones tensor([14], device='cuda:0')
Batch 2 tiene 1 tokens EOI en posiciones tensor([19], device='cuda:0')
Batch 3 tiene 1 tokens BOI en posiciones tensor([8], device='cuda:0')
Batch 3 tiene 1 tokens EOI en posiciones tensor([13], device='cuda:0')
Batch 7 tiene 1 tokens BOI en posiciones tensor([21], device='cuda:0')
Batch 7 tiene 1 tokens EOI en posiciones tensor([26], device='cuda:0')
Posiciones de imágenes encontradas: [(2, 14, 19), (3, 8, 13), (7, 21, 26)]
x_t shape: torch.Size([3, 8, 4, 4]), dx_t shape: torch.Size([3, 8, 4, 4])
batch_idx_map: {2: 0, 3: 1, 7: 2}
Tensor en posición [2, 15:19] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([3, 8, 4, 4]), flow_idx: 0
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [3, 9:13] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([3, 8, 4, 4]), flow_idx: 1
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [7, 22:26] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([3, 8, 4, 4]), flow_idx: 2
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Pérdida de flow inválida, usando valor por defecto
Época 1/10:  53%|███████████████████████████████████████▍                                   | 125/238 [00:49<00:44,  2.53it/s, loss=nan, lm_loss=nan, fm_loss=4.17, lr=1.17e-05]Procesando batch con 6 imágenes
BOI índices: [7, 14, 16, 10, 12, 25]
Creado image_latents_dict con 6 entradas
Claves en image_latents_dict: [7, 14, 16, 10, 12, 25]
Imagen 3331102049_bc65cf6198.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 3201666946_04fe837aff.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 2045109977_b00ec93491.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Token BOI: 128256, EOI: 128257
Batch 1 tiene 1 tokens BOI en posiciones tensor([7], device='cuda:0')
Batch 1 tiene 1 tokens EOI en posiciones tensor([12], device='cuda:0')
Batch 2 tiene 1 tokens BOI en posiciones tensor([14], device='cuda:0')
Batch 2 tiene 1 tokens EOI en posiciones tensor([19], device='cuda:0')
Batch 3 tiene 1 tokens BOI en posiciones tensor([16], device='cuda:0')
Batch 3 tiene 1 tokens EOI en posiciones tensor([21], device='cuda:0')
Batch 4 tiene 1 tokens BOI en posiciones tensor([10], device='cuda:0')
Batch 4 tiene 1 tokens EOI en posiciones tensor([15], device='cuda:0')
Batch 5 tiene 1 tokens BOI en posiciones tensor([12], device='cuda:0')
Batch 5 tiene 1 tokens EOI en posiciones tensor([17], device='cuda:0')
Batch 7 tiene 1 tokens BOI en posiciones tensor([25], device='cuda:0')
Batch 7 tiene 1 tokens EOI en posiciones tensor([30], device='cuda:0')
Posiciones de imágenes encontradas: [(1, 7, 12), (2, 14, 19), (3, 16, 21), (4, 10, 15), (5, 12, 17), (7, 25, 30)]
x_t shape: torch.Size([6, 8, 4, 4]), dx_t shape: torch.Size([6, 8, 4, 4])
batch_idx_map: {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 7: 5}
Tensor en posición [1, 8:12] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([6, 8, 4, 4]), flow_idx: 0
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [2, 15:19] shape: torch.Size([4, 512])
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
Tensor en posición [4, 11:15] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([6, 8, 4, 4]), flow_idx: 3
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [5, 13:17] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([6, 8, 4, 4]), flow_idx: 4
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [7, 26:30] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([6, 8, 4, 4]), flow_idx: 5
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Pérdida de flow inválida, usando valor por defecto
Época 1/10:  53%|████████████████████████████████████████▏                                   | 126/238 [00:50<00:43,  2.57it/s, loss=nan, lm_loss=nan, fm_loss=2.1, lr=1.17e-05]Procesando batch con 2 imágenes
BOI índices: [15, 13]
Creado image_latents_dict con 2 entradas
Claves en image_latents_dict: [15, 13]
Imagen 95783195_e1ba3f57ca.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 2907073768_08fd7bdf60.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 2978409165_acc4f29a40.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Token BOI: 128256, EOI: 128257
Batch 1 tiene 1 tokens BOI en posiciones tensor([15], device='cuda:0')
Batch 1 tiene 1 tokens EOI en posiciones tensor([20], device='cuda:0')
Batch 6 tiene 1 tokens BOI en posiciones tensor([13], device='cuda:0')
Batch 6 tiene 1 tokens EOI en posiciones tensor([18], device='cuda:0')
Posiciones de imágenes encontradas: [(1, 15, 20), (6, 13, 18)]
x_t shape: torch.Size([2, 8, 4, 4]), dx_t shape: torch.Size([2, 8, 4, 4])
batch_idx_map: {1: 0, 6: 1}
Tensor en posición [1, 16:20] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([2, 8, 4, 4]), flow_idx: 0
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [6, 14:18] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([2, 8, 4, 4]), flow_idx: 1
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Pérdida de flow inválida, usando valor por defecto
Época 1/10:  53%|████████████████████████████████████████                                   | 127/238 [00:50<00:42,  2.60it/s, loss=nan, lm_loss=nan, fm_loss=6.35, lr=1.17e-05]Procesando batch con 5 imágenes
BOI índices: [11, 6, 10, 11, 9]
Creado image_latents_dict con 4 entradas
Claves en image_latents_dict: [11, 6, 10, 9]
Imagen 339658315_fbb178c252.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 1461653394_8ab96aae63.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 2271890493_da441718ba.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Token BOI: 128256, EOI: 128257
Batch 1 tiene 1 tokens BOI en posiciones tensor([11], device='cuda:0')
Batch 1 tiene 1 tokens EOI en posiciones tensor([16], device='cuda:0')
Batch 3 tiene 1 tokens BOI en posiciones tensor([6], device='cuda:0')
Batch 3 tiene 1 tokens EOI en posiciones tensor([11], device='cuda:0')
Batch 4 tiene 1 tokens BOI en posiciones tensor([10], device='cuda:0')
Batch 4 tiene 1 tokens EOI en posiciones tensor([15], device='cuda:0')
Batch 6 tiene 1 tokens BOI en posiciones tensor([11], device='cuda:0')
Batch 6 tiene 1 tokens EOI en posiciones tensor([16], device='cuda:0')
Batch 7 tiene 1 tokens BOI en posiciones tensor([9], device='cuda:0')
Batch 7 tiene 1 tokens EOI en posiciones tensor([14], device='cuda:0')
Posiciones de imágenes encontradas: [(1, 11, 16), (3, 6, 11), (4, 10, 15), (6, 11, 16), (7, 9, 14)]
x_t shape: torch.Size([5, 8, 4, 4]), dx_t shape: torch.Size([5, 8, 4, 4])
batch_idx_map: {1: 0, 3: 1, 4: 2, 6: 3, 7: 4}
Tensor en posición [1, 12:16] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([5, 8, 4, 4]), flow_idx: 0
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [3, 7:11] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([5, 8, 4, 4]), flow_idx: 1
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [4, 11:15] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([5, 8, 4, 4]), flow_idx: 2
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [6, 12:16] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([5, 8, 4, 4]), flow_idx: 3
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Tensor en posición [7, 10:14] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([5, 8, 4, 4]), flow_idx: 4
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Advertencia: Número de parches no coincide - recibidos: 4, esperados: 16
Formas no coinciden: imagen=torch.Size([1, 8, 8, 8]), target=torch.Size([1, 8, 4, 4])
Pérdida de flow inválida, usando valor por defecto
Época 1/10:  54%|████████████████████████████████████████▎                                  | 128/238 [00:51<00:42,  2.62it/s, loss=nan, lm_loss=nan, fm_loss=2.56, lr=1.17e-05]Imagen 380537190_11d6c0a412.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 69189650_6687da7280.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 3534183988_3763593dfb.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 506343925_b30a235de6.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Época 1/10:  54%|███████████████████████████████████████▌                                 | 129/238 [00:51<00:41,  2.65it/s, loss=nan, lm_loss=nan, fm_loss=0.0992, lr=1.17e-05]Procesando batch con 3 imágenes
BOI índices: [8, 11, 7]
Creado image_latents_dict con 3 entradas
Claves en image_latents_dict: [8, 11, 7]
Imagen 3463268965_f22884fc69.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 3412249548_00820fc4ca.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 880220939_0ef1c37f1f.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 3003612178_8230d65833.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 2565703445_dd6899bc0e.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Token BOI: 128256, EOI: 128257
Época 1/10:  54%|███████████████████████████████████████▌                                 | 129/238 [00:51<00:43,  2.50it/s, loss=nan, lm_loss=nan, fm_loss=0.0992, lr=1.17e-05]
Traceback (most recent call last):
  File "/teamspace/studios/this_studio/tr-fl-local/train.py", line 650, in <module>
    train_transfusion_model()
  File "/teamspace/studios/this_studio/tr-fl-local/train.py", line 645, in train_transfusion_model
    trained_model = trainer.train(train_dataloader, val_dataloader)
  File "/teamspace/studios/this_studio/tr-fl-local/train.py", line 504, in train
    stop_training = self.train_epoch(train_dataloader, epoch)
  File "/teamspace/studios/this_studio/tr-fl-local/train.py", line 257, in train_epoch
    _, total_loss, lm_loss, fm_loss = self.model(
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 735, in forward
    fm_loss = self.forward_flow_matching(h, tokens, flow_x0, flow_x1, flow_t)
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 565, in forward_flow_matching
    image_positions = self.extract_image_positions(tokens)
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 515, in extract_image_positions
    boi_pos = (tokens == self.boi_token).nonzero(as_tuple=True)
KeyboardInterrupt

```