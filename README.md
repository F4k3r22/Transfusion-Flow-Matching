# Transfusion with Flow-Matching

Hey, we're implementing the Transfusion model outlined in this META paper: [Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model](https://arxiv.org/pdf/2408.11039)
We're experiencing these errors. Any suggestions on how to resolve them are welcome. :b. Thank you all in advance.

Errors we are having when trying to train the model:
```
Error al decodificar imagen: cannot unpack non-iterable builtin_function_or_method object
Tensor en posición [5, 15:19] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 2
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Traceback (most recent call last):
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 634, in forward_flow_matching
    image_latent = self.patch_decoder(image_repr, latent_h * self.params.patch_size, latent_w * self.params.patch_size)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 103, in forward
    B, L, _ = x.reshape
TypeError: cannot unpack non-iterable builtin_function_or_method object
Error al decodificar imagen: cannot unpack non-iterable builtin_function_or_method object
Tensor en posición [7, 8:12] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 3
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Traceback (most recent call last):
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 634, in forward_flow_matching
    image_latent = self.patch_decoder(image_repr, latent_h * self.params.patch_size, latent_w * self.params.patch_size)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 103, in forward
    B, L, _ = x.reshape
TypeError: cannot unpack non-iterable builtin_function_or_method object
Error al decodificar imagen: cannot unpack non-iterable builtin_function_or_method object
                                                                                                                                                                                Token BOI: 128256, EOI: 128257█████████████████████████████████████████████████████████████████▍                                                 | 8/13 [00:01<00:00,  7.18it/s]
Batch 2 tiene 1 tokens BOI en posiciones tensor([13], device='cuda:0')
Batch 2 tiene 1 tokens EOI en posiciones tensor([18], device='cuda:0')
Batch 3 tiene 1 tokens BOI en posiciones tensor([18], device='cuda:0')
Batch 3 tiene 1 tokens EOI en posiciones tensor([23], device='cuda:0')
Batch 6 tiene 1 tokens BOI en posiciones tensor([17], device='cuda:0')
Batch 6 tiene 1 tokens EOI en posiciones tensor([22], device='cuda:0')
Batch 7 tiene 1 tokens BOI en posiciones tensor([8], device='cuda:0')
Batch 7 tiene 1 tokens EOI en posiciones tensor([13], device='cuda:0')
Posiciones de imágenes encontradas: [(2, 13, 18), (3, 18, 23), (6, 17, 22), (7, 8, 13)]
x_t shape: torch.Size([4, 8, 4, 4]), dx_t shape: torch.Size([4, 8, 4, 4])
batch_idx_map: {2: 0, 3: 1, 6: 2, 7: 3}
Tensor en posición [2, 14:18] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 0
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Traceback (most recent call last):
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 634, in forward_flow_matching
    image_latent = self.patch_decoder(image_repr, latent_h * self.params.patch_size, latent_w * self.params.patch_size)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 103, in forward
    B, L, _ = x.reshape
TypeError: cannot unpack non-iterable builtin_function_or_method object
Error al decodificar imagen: cannot unpack non-iterable builtin_function_or_method object
Tensor en posición [3, 19:23] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 1
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Traceback (most recent call last):
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 634, in forward_flow_matching
    image_latent = self.patch_decoder(image_repr, latent_h * self.params.patch_size, latent_w * self.params.patch_size)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 103, in forward
    B, L, _ = x.reshape
TypeError: cannot unpack non-iterable builtin_function_or_method object
Error al decodificar imagen: cannot unpack non-iterable builtin_function_or_method object
Tensor en posición [6, 18:22] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 2
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Traceback (most recent call last):
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 634, in forward_flow_matching
    image_latent = self.patch_decoder(image_repr, latent_h * self.params.patch_size, latent_w * self.params.patch_size)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 103, in forward
    B, L, _ = x.reshape
TypeError: cannot unpack non-iterable builtin_function_or_method object
Error al decodificar imagen: cannot unpack non-iterable builtin_function_or_method object
Tensor en posición [7, 9:13] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 3
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Traceback (most recent call last):
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 634, in forward_flow_matching
    image_latent = self.patch_decoder(image_repr, latent_h * self.params.patch_size, latent_w * self.params.patch_size)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 103, in forward
    B, L, _ = x.reshape
TypeError: cannot unpack non-iterable builtin_function_or_method object
Error al decodificar imagen: cannot unpack non-iterable builtin_function_or_method object
                                                                                                                                                                                Token BOI: 128256, EOI: 128257███████████████████████████████████████████████████████████████████████████▎                                       | 9/13 [00:01<00:00,  7.26it/s]
Batch 0 tiene 1 tokens BOI en posiciones tensor([8], device='cuda:0')
Batch 0 tiene 1 tokens EOI en posiciones tensor([13], device='cuda:0')
Batch 3 tiene 1 tokens BOI en posiciones tensor([14], device='cuda:0')
Batch 3 tiene 1 tokens EOI en posiciones tensor([19], device='cuda:0')
Batch 6 tiene 1 tokens BOI en posiciones tensor([17], device='cuda:0')
Batch 6 tiene 1 tokens EOI en posiciones tensor([22], device='cuda:0')
Batch 7 tiene 1 tokens BOI en posiciones tensor([10], device='cuda:0')
Batch 7 tiene 1 tokens EOI en posiciones tensor([15], device='cuda:0')
Posiciones de imágenes encontradas: [(0, 8, 13), (3, 14, 19), (6, 17, 22), (7, 10, 15)]
x_t shape: torch.Size([4, 8, 4, 4]), dx_t shape: torch.Size([4, 8, 4, 4])
batch_idx_map: {0: 0, 3: 1, 6: 2, 7: 3}
Tensor en posición [0, 9:13] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 0
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Traceback (most recent call last):
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 634, in forward_flow_matching
    image_latent = self.patch_decoder(image_repr, latent_h * self.params.patch_size, latent_w * self.params.patch_size)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 103, in forward
    B, L, _ = x.reshape
TypeError: cannot unpack non-iterable builtin_function_or_method object
Error al decodificar imagen: cannot unpack non-iterable builtin_function_or_method object
Tensor en posición [3, 15:19] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 1
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Traceback (most recent call last):
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 634, in forward_flow_matching
    image_latent = self.patch_decoder(image_repr, latent_h * self.params.patch_size, latent_w * self.params.patch_size)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 103, in forward
    B, L, _ = x.reshape
TypeError: cannot unpack non-iterable builtin_function_or_method object
Error al decodificar imagen: cannot unpack non-iterable builtin_function_or_method object
Tensor en posición [6, 18:22] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 2
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Traceback (most recent call last):
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 634, in forward_flow_matching
    image_latent = self.patch_decoder(image_repr, latent_h * self.params.patch_size, latent_w * self.params.patch_size)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 103, in forward
    B, L, _ = x.reshape
TypeError: cannot unpack non-iterable builtin_function_or_method object
Error al decodificar imagen: cannot unpack non-iterable builtin_function_or_method object
Tensor en posición [7, 11:15] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([4, 8, 4, 4]), flow_idx: 3
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Traceback (most recent call last):
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 634, in forward_flow_matching
    image_latent = self.patch_decoder(image_repr, latent_h * self.params.patch_size, latent_w * self.params.patch_size)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 103, in forward
    B, L, _ = x.reshape
TypeError: cannot unpack non-iterable builtin_function_or_method object
Error al decodificar imagen: cannot unpack non-iterable builtin_function_or_method object
                                                                                                                                                                                Token BOI: 128256, EOI: 128257████████████████████████████████████████████████████████████████████████████████████▍                             | 10/13 [00:01<00:00,  7.26it/s]
Batch 2 tiene 1 tokens BOI en posiciones tensor([15], device='cuda:0')
Batch 2 tiene 1 tokens EOI en posiciones tensor([20], device='cuda:0')
Batch 5 tiene 1 tokens BOI en posiciones tensor([10], device='cuda:0')
Batch 5 tiene 1 tokens EOI en posiciones tensor([15], device='cuda:0')
Batch 7 tiene 1 tokens BOI en posiciones tensor([14], device='cuda:0')
Batch 7 tiene 1 tokens EOI en posiciones tensor([19], device='cuda:0')
Posiciones de imágenes encontradas: [(2, 15, 20), (5, 10, 15), (7, 14, 19)]
x_t shape: torch.Size([3, 8, 4, 4]), dx_t shape: torch.Size([3, 8, 4, 4])
batch_idx_map: {2: 0, 5: 1, 7: 2}
Tensor en posición [2, 16:20] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([3, 8, 4, 4]), flow_idx: 0
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Traceback (most recent call last):
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 634, in forward_flow_matching
    image_latent = self.patch_decoder(image_repr, latent_h * self.params.patch_size, latent_w * self.params.patch_size)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 103, in forward
    B, L, _ = x.reshape
TypeError: cannot unpack non-iterable builtin_function_or_method object
Error al decodificar imagen: cannot unpack non-iterable builtin_function_or_method object
Tensor en posición [5, 11:15] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([3, 8, 4, 4]), flow_idx: 1
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Traceback (most recent call last):
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 634, in forward_flow_matching
    image_latent = self.patch_decoder(image_repr, latent_h * self.params.patch_size, latent_w * self.params.patch_size)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 103, in forward
    B, L, _ = x.reshape
TypeError: cannot unpack non-iterable builtin_function_or_method object
Error al decodificar imagen: cannot unpack non-iterable builtin_function_or_method object
Tensor en posición [7, 15:19] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([3, 8, 4, 4]), flow_idx: 2
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Traceback (most recent call last):
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 634, in forward_flow_matching
    image_latent = self.patch_decoder(image_repr, latent_h * self.params.patch_size, latent_w * self.params.patch_size)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 103, in forward
    B, L, _ = x.reshape
TypeError: cannot unpack non-iterable builtin_function_or_method object
Error al decodificar imagen: cannot unpack non-iterable builtin_function_or_method object
                                                                                                                                                                                Token BOI: 128256, EOI: 128257██████████████████████████████████████████████████████████████████████████████████████████████▎                   | 11/13 [00:01<00:00,  7.34it/s]
Batch 1 tiene 1 tokens BOI en posiciones tensor([9], device='cuda:0')
Batch 1 tiene 1 tokens EOI en posiciones tensor([14], device='cuda:0')
Batch 3 tiene 1 tokens BOI en posiciones tensor([16], device='cuda:0')
Batch 3 tiene 1 tokens EOI en posiciones tensor([21], device='cuda:0')
Batch 4 tiene 1 tokens BOI en posiciones tensor([9], device='cuda:0')
Batch 4 tiene 1 tokens EOI en posiciones tensor([14], device='cuda:0')
Posiciones de imágenes encontradas: [(1, 9, 14), (3, 16, 21), (4, 9, 14)]
x_t shape: torch.Size([3, 8, 4, 4]), dx_t shape: torch.Size([3, 8, 4, 4])
batch_idx_map: {1: 0, 3: 1, 4: 2}
Tensor en posición [1, 10:14] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([3, 8, 4, 4]), flow_idx: 0
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Traceback (most recent call last):
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 634, in forward_flow_matching
    image_latent = self.patch_decoder(image_repr, latent_h * self.params.patch_size, latent_w * self.params.patch_size)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 103, in forward
    B, L, _ = x.reshape
TypeError: cannot unpack non-iterable builtin_function_or_method object
Error al decodificar imagen: cannot unpack non-iterable builtin_function_or_method object
Tensor en posición [3, 17:21] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([3, 8, 4, 4]), flow_idx: 1
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Traceback (most recent call last):
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 634, in forward_flow_matching
    image_latent = self.patch_decoder(image_repr, latent_h * self.params.patch_size, latent_w * self.params.patch_size)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 103, in forward
    B, L, _ = x.reshape
TypeError: cannot unpack non-iterable builtin_function_or_method object
Error al decodificar imagen: cannot unpack non-iterable builtin_function_or_method object
Tensor en posición [4, 10:14] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([3, 8, 4, 4]), flow_idx: 2
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Traceback (most recent call last):
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 634, in forward_flow_matching
    image_latent = self.patch_decoder(image_repr, latent_h * self.params.patch_size, latent_w * self.params.patch_size)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 103, in forward
    B, L, _ = x.reshape
TypeError: cannot unpack non-iterable builtin_function_or_method object
Error al decodificar imagen: cannot unpack non-iterable builtin_function_or_method object
                                                                                                                                                                                Token BOI: 128256, EOI: 128257████████████████████████████████████████████████████████████████████████████████████████████████████████▏         | 12/13 [00:01<00:00,  7.39it/s]
Batch 2 tiene 1 tokens BOI en posiciones tensor([7], device='cuda:0')
Batch 2 tiene 1 tokens EOI en posiciones tensor([12], device='cuda:0')
Posiciones de imágenes encontradas: [(2, 7, 12)]
x_t shape: torch.Size([1, 8, 4, 4]), dx_t shape: torch.Size([1, 8, 4, 4])
batch_idx_map: {2: 0}
Tensor en posición [2, 8:12] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([1, 8, 4, 4]), flow_idx: 0
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Traceback (most recent call last):
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 634, in forward_flow_matching
    image_latent = self.patch_decoder(image_repr, latent_h * self.params.patch_size, latent_w * self.params.patch_size)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 103, in forward
    B, L, _ = x.reshape
TypeError: cannot unpack non-iterable builtin_function_or_method object
Error al decodificar imagen: cannot unpack non-iterable builtin_function_or_method object
Evaluando: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:01<00:00,  6.79it/s]
2025-04-18 16:15:47,879 - INFO - Pérdida de validación: nan, Pérdida LM: nan, Pérdida FM: 0.1000
Época 1/10:   0%|▎                                                                            | 1/238 [00:05<22:22,  5.66s/it, loss=nan, lm_loss=nan, fm_loss=0.02, lr=0.00e+00]Procesando batch con 3 imágenes
BOI índices: [16, 24, 12]
Creado image_latents_dict con 3 entradas
Claves en image_latents_dict: [16, 24, 12]
Imagen 2855667597_bf6ceaef8e.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 2535746605_8124bf4e4f.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 2030781555_b7ff7be28f.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 3374054694_fa56f29267.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Imagen 3000722396_1ae2e976c2.jpg.pt: calculados 4 parches necesarios
Espacio para parches: 4, requerido: 4
Token BOI: 128256, EOI: 128257
Batch 3 tiene 1 tokens BOI en posiciones tensor([16], device='cuda:0')
Batch 3 tiene 1 tokens EOI en posiciones tensor([21], device='cuda:0')
Batch 6 tiene 1 tokens BOI en posiciones tensor([24], device='cuda:0')
Batch 6 tiene 1 tokens EOI en posiciones tensor([29], device='cuda:0')
Batch 7 tiene 1 tokens BOI en posiciones tensor([12], device='cuda:0')
Batch 7 tiene 1 tokens EOI en posiciones tensor([17], device='cuda:0')
Posiciones de imágenes encontradas: [(3, 16, 21), (6, 24, 29), (7, 12, 17)]
x_t shape: torch.Size([3, 8, 4, 4]), dx_t shape: torch.Size([3, 8, 4, 4])
batch_idx_map: {3: 0, 6: 1, 7: 2}
Tensor en posición [3, 17:21] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([3, 8, 4, 4]), flow_idx: 0
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Traceback (most recent call last):
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 634, in forward_flow_matching
    image_latent = self.patch_decoder(image_repr, latent_h * self.params.patch_size, latent_w * self.params.patch_size)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 103, in forward
    B, L, _ = x.reshape
TypeError: cannot unpack non-iterable builtin_function_or_method object
Error al decodificar imagen: cannot unpack non-iterable builtin_function_or_method object
Tensor en posición [6, 25:29] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([3, 8, 4, 4]), flow_idx: 1
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Traceback (most recent call last):
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 634, in forward_flow_matching
    image_latent = self.patch_decoder(image_repr, latent_h * self.params.patch_size, latent_w * self.params.patch_size)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 103, in forward
    B, L, _ = x.reshape
TypeError: cannot unpack non-iterable builtin_function_or_method object
Error al decodificar imagen: cannot unpack non-iterable builtin_function_or_method object
Tensor en posición [7, 13:17] shape: torch.Size([4, 512])
Contenido no cero: 2048
dx_t shape: torch.Size([3, 8, 4, 4]), flow_idx: 2
target_shape: torch.Size([8, 4, 4]), dimensiones: 3
Traceback (most recent call last):
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 634, in forward_flow_matching
    image_latent = self.patch_decoder(image_repr, latent_h * self.params.patch_size, latent_w * self.params.patch_size)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/teamspace/studios/this_studio/tr-fl-local/model.py", line 103, in forward
    B, L, _ = x.reshape
TypeError: cannot unpack non-iterable builtin_function_or_method object
Error al decodificar imagen: cannot unpack non-iterable builtin_function_or_method object
Época 1/10:   0%|▎                                                                          | 1/238 [00:06<22:22,  5.66s/it, loss=nan, lm_loss=nan, fm_loss=0.0667, lr=0.00e+00]2025-04-18 16:15:48,413 - INFO - Generando muestras en paso 0...
2025-04-18 16:15:48,426 - ERROR - Error al generar muestras: too many values to unpack (expected 2)
Época 1/10:   0%|▎                                                                          | 1/238 [00:07<28:21,  7.18s/it, loss=nan, lm_loss=nan, fm_loss=0.0667, lr=0.00e+00]
Traceback (most recent call last):
  File "/teamspace/studios/this_studio/tr-fl-local/train.py", line 650, in <module>
    train_transfusion_model()
  File "/teamspace/studios/this_studio/tr-fl-local/train.py", line 645, in train_transfusion_model
    trained_model = trainer.train(train_dataloader, val_dataloader)
  File "/teamspace/studios/this_studio/tr-fl-local/train.py", line 504, in train
    stop_training = self.train_epoch(train_dataloader, epoch)
  File "/teamspace/studios/this_studio/tr-fl-local/train.py", line 308, in train_epoch
    self.save_checkpoint()
  File "/teamspace/studios/this_studio/tr-fl-local/train.py", line 116, in save_checkpoint
    torch.save(checkpoint, checkpoint_path)
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/serialization.py", line 944, in save
    _save(
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/serialization.py", line 1214, in _save
    storage = storage.cpu()
  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/storage.py", line 267, in cpu
    return torch.UntypedStorage(self.size()).copy_(self, False)
KeyboardInterrupt

```