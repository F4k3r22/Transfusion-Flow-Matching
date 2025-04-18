# Transfusion with Flow-Matching

Hey, we're implementing the Transfusion model outlined in this META paper: [Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model](https://arxiv.org/pdf/2408.11039)
We're experiencing these errors. Any suggestions on how to resolve them are welcome. :b. Thank you all in advance.

Errors we are having when trying to train the model:
```
Batch 0 tiene 1 tokens BOI en posiciones tensor([15], device='cuda:0')
Batch 0 tiene 1 tokens EOI en posiciones tensor([20], device='cuda:0')
Batch 1 tiene 1 tokens BOI en posiciones tensor([11], device='cuda:0')
Batch 1 tiene 1 tokens EOI en posiciones tensor([16], device='cuda:0')
Batch 3 tiene 1 tokens BOI en posiciones tensor([15], device='cuda:0')
Batch 3 tiene 1 tokens EOI en posiciones tensor([20], device='cuda:0')
Posiciones de imágenes encontradas: [(0, 15, 20), (1, 11, 16), (3, 15, 20)]
x_t shape: torch.Size([3, 8, 4, 4]), dx_t shape: torch.Size([3, 8, 4, 4])
batch_idx_map: {0: 0, 1: 1, 3: 2}
Tensor en posición [0, 16:20] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([3, 8, 4, 4]), flow_idx: 0
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Error al decodificar imagen: cannot unpack non-iterable builtin_function_or_method object
Tensor en posición [1, 12:16] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([3, 8, 4, 4]), flow_idx: 1
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Error al decodificar imagen: cannot unpack non-iterable builtin_function_or_method object
Tensor en posición [3, 16:20] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([3, 8, 4, 4]), flow_idx: 2
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Error al decodificar imagen: cannot unpack non-iterable builtin_function_or_method object
tensor(nan, device='cuda:0')
Pérdida de flow inválida, usando valor por defecto
Evaluando: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:02<00:00,  6.43it/s]
2025-04-18 15:53:47,770 - INFO - Pérdida de validación: nan, Pérdida LM: nan, Pérdida FM: 0.1000
Época 1/10:   3%|██                                                                                | 6/238 [00:30<19:17,  4.99s/it, loss=30, lm_loss=30, fm_loss=0, lr=0.00e+00]Procesando batch con 4 imágenes
BOI índices: [8, 19, 10, 10]
Creado image_latents_dict con 3 entradas
Claves en image_latents_dict: [8, 19, 10]
Forward recibiendo 3 latentes de imagen
Imagen 1197800988_7fb0ca4888.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 2600386812_8790879d9a.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 3431121650_056db85987.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 769260947_02bc973d76.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Token BOI: 128256, EOI: 128257
Tokens únicos: tensor([     0,      1,      8,  ...,  93091, 128256, 128257], device='cuda:0')
Batch 3 tiene 1 tokens BOI en posiciones tensor([8], device='cuda:0')
Batch 3 tiene 1 tokens EOI en posiciones tensor([13], device='cuda:0')
Batch 5 tiene 1 tokens BOI en posiciones tensor([19], device='cuda:0')
Batch 5 tiene 1 tokens EOI en posiciones tensor([24], device='cuda:0')
Batch 6 tiene 1 tokens BOI en posiciones tensor([10], device='cuda:0')
Batch 6 tiene 1 tokens EOI en posiciones tensor([15], device='cuda:0')
Batch 7 tiene 1 tokens BOI en posiciones tensor([10], device='cuda:0')
Batch 7 tiene 1 tokens EOI en posiciones tensor([15], device='cuda:0')
Posiciones de imágenes encontradas: [(3, 8, 13), (5, 19, 24), (6, 10, 15), (7, 10, 15)]
x_t shape: torch.Size([4, 8, 4, 4]), dx_t shape: torch.Size([4, 8, 4, 4])
batch_idx_map: {3: 0, 5: 1, 6: 2, 7: 3}
Tensor en posición [3, 9:13] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 0
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Error al decodificar imagen: cannot unpack non-iterable builtin_function_or_method object
Tensor en posición [5, 20:24] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 1
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Error al decodificar imagen: cannot unpack non-iterable builtin_function_or_method object
Tensor en posición [6, 11:15] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 2
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Error al decodificar imagen: cannot unpack non-iterable builtin_function_or_method object
Tensor en posición [7, 11:15] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 3
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Error al decodificar imagen: cannot unpack non-iterable builtin_function_or_method object
tensor(nan, device='cuda:0', grad_fn=<MeanBackward0>)
Pérdida de flow inválida, usando valor por defecto
¡Advertencia! Pérdida NaN detectada en paso 0
Época 1/10:   3%|██                                                                                | 6/238 [00:30<19:29,  5.04s/it, loss=30, lm_loss=30, fm_loss=0, lr=0.00e+00]
Traceback (most recent call last):
  File "/teamspace/studios/this_studio/tr-fl-local/train.py", line 648, in <module>
    train_transfusion_model()
  File "/teamspace/studios/this_studio/tr-fl-local/train.py", line 643, in train_transfusion_model
    trained_model = trainer.train(train_dataloader, val_dataloader)
  File "/teamspace/studios/this_studio/tr-fl-local/train.py", line 502, in train
    stop_training = self.train_epoch(train_dataloader, epoch)
  File "/teamspace/studios/this_studio/tr-fl-local/train.py", line 237, in train_epoch
    self.scaler.step(self.optimizer)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/amp/grad_scaler.py", line 454, in step
    len(optimizer_state["found_inf_per_device"]) > 0
AssertionError: No inf checks were recorded for this optimizer.
```