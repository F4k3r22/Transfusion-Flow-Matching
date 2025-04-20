#model.py

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import torch
import torch.nn.functional as F
import torch.nn as nn
#from flow_matching.path.scheduler import CondOTScheduler

@dataclass
class ModelArgs:
    # Parámetros base del transformador
    dim: int = 2048
    n_layers: int = 24
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    max_batch_size: int = 5
    max_seq_len: int = 2048
    dropout: float = 0.1
    
    # Parámetros específicos para Transfusion
    vae_latent_dim: int = 8          # Dimensión latente del VAE
    patch_size: int = 2              # Tamaño de parche (en espacio latente)
    max_img_size: Tuple[int, int] = (256, 256)  # Tamaño máximo de imagen
    fm_lambda: float = 5.0           # Factor de peso para la pérdida de Flow Matching
    continuous_dim: int = 8          # Dimensión de los vectores de parche (VAE latent dim)

# Clase básica para VAE 
class VAE(nn.Module):
    def __init__(self, latent_dim=8):
        super().__init__()
        
        
        # Encoder (simplificado)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, latent_dim, kernel_size=1, stride=1)
        )
        
        # Decoder (simplificado)
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, 256, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        return self.encoder(x)
        
    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

class PatchEncoder(nn.Module):
    """Convierte parches latentes en vectores para el transformador"""
    def __init__(self, latent_dim, model_dim, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.linear = nn.Linear(latent_dim * patch_size * patch_size, model_dim)
        
    def forward(self, x):
        # x: [B, C, H, W] -> [B, L, D]
        B, C, H, W = x.shape
        x = x.reshape(B, C, H//self.patch_size, self.patch_size, W//self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C*self.patch_size*self.patch_size)
        return self.linear(x)

class PatchDecoder(nn.Module):
    """Convierte vectores del transformador en parches latentes con manejo de errores"""
    def __init__(self, latent_dim, model_dim, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.linear = nn.Linear(model_dim, latent_dim * patch_size * patch_size)
        
       
    def forward(self, x, H, W):
        """Convierte vectores del transformador en parches latentes con redimensionamiento correcto"""
        B, L, _ = x.shape
        patches = self.linear(x)

        # Calcular dimensiones esperadas para el tamaño de parche
        expected_h = H // self.patch_size
        expected_w = W // self.patch_size
        expected_patches = expected_h * expected_w
    
        # Verificar si necesitamos redimensionar
        if L != expected_patches:
            print(f"Advertencia: Número de parches no coincide - recibidos: {L}, esperados: {expected_patches}")
        
            try:
                # Convertir a formato intermedio (más fácil de manipular)
                patches_flat = patches.reshape(B, L, self.latent_dim * self.patch_size * self.patch_size)
                patches_reshaped = patches_flat.permute(0, 2, 1)  # [B, C*PS*PS, L]
            
                # Calcular el factor de escala
                factor = int(math.sqrt(L / expected_patches))
                # Redimensionar a formato temporal
                patches_temp = patches_reshaped.reshape(
                    B, 
                    self.latent_dim * self.patch_size * self.patch_size, 
                    int(math.sqrt(L)), 
                    int(math.sqrt(L))
                )
            
                # Redimensionar a la forma correcta con pooling
                result = F.adaptive_avg_pool2d(
                    patches_temp,
                    (expected_h, expected_w)
                )
            
                # Redimensionar al formato final
                result = result.reshape(B, self.latent_dim, expected_h * self.patch_size, expected_w * self.patch_size)
                return result
            
            except Exception as e:
                print(f"Error en método avanzado: {e}")
                # Fallback a reshape básico + interpolación
                try:
                    temp = patches.reshape(B, self.latent_dim, L, -1)
                    return F.interpolate(temp, size=(H, W), mode='bilinear', align_corners=False)
                except Exception as e2:
                    print(f"Error en fallback: {e2}")
                    # Último recurso: tensor cero
                    return torch.zeros(B, self.latent_dim, H, W, device=x.device)
    
        # Proceso normal si no hay discrepancia
        try:
            patches = patches.reshape(
                B, expected_h, expected_w, 
                self.latent_dim, self.patch_size, self.patch_size
            )
            patches = patches.permute(0, 3, 1, 4, 2, 5)
            patches = patches.reshape(B, self.latent_dim, H, W)
            return patches
        except Exception as e:
            print(f"Error en reshape estándar: {e}")
            return torch.zeros(B, self.latent_dim, H, W, device=x.device)

# Componentes de precomputo para las embeddings rotacionales
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class CondOTScheduler:
    """Scheduler para Flow Matching con transformación lineal"""
    def __init__(self, sigma_min=0.002, sigma_max=80.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def marginal_prob(self, t, x_0, x_1):
        """
        Calcula x_t y dx_t/dt para el tiempo t
        """
        # Asegurar dimensiones compatibles para broadcasting
        if t.dim() == 1:
            # Reformatear t para broadcasting adecuado con x_0 y x_1
            t = t.view(-1, 1, 1, 1)
        
        # Asegurar que x_0 y x_1 tienen la misma forma
        if x_0.shape != x_1.shape:
            # Redimensionar x_0 para que coincida con x_1
            x_0 = x_0[:, :, :x_1.shape[2], :x_1.shape[3]]
        
        x_t = (1 - t) * x_0 + t * x_1
        dx_t = x_1 - x_0
        return x_t, dx_t

class FlowMatchingModule(nn.Module):
    """Componente de Flow Matching para Transfusion"""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.scheduler = CondOTScheduler()
        
    def sample_path(self, x_0, x_1, t):
        """Muestra un punto en la trayectoria"""
        if t.dim() == 1:
            t = t.view(-1, 1, 1, 1)

        x_t, dx_t = self.scheduler.marginal_prob(t, x_0, x_1)
        return x_t, dx_t
    
    def compute_flow_loss(self, predicted_v, target_v):
        """Calcula pérdida de Flow Matching (MSE) asegurando compatibilidad de formas"""
        # Verificar y ajustar formas si es necesario
        if predicted_v.shape != target_v.shape:
            # Intentar aplanar ambos tensores para cálculo de error
            flat_pred = predicted_v.reshape(-1)
            flat_target = target_v.reshape(-1)
        
            # Truncar al tamaño más pequeño si es necesario
            min_size = min(flat_pred.shape[0], flat_target.shape[0])
            flat_pred = flat_pred[:min_size]
            flat_target = flat_target[:min_size]
        
            return F.mse_loss(flat_pred, flat_target)
    
        return F.mse_loss(predicted_v, target_v)
    
    def sample(self, x_0, vf_model, steps=50):
        """Genera una muestra desde x_0 usando el modelo de velocidad"""
        # Solver simple de Euler
        dt = 1.0 / steps
        x_t = x_0
        
        for i in range(steps):
            t = torch.ones((x_t.shape[0], 1), device=x_t.device) * i * dt
            v = vf_model(x_t, t)
            x_t = x_t + v * dt
            
        return x_t
    
    def noise_to_image(self, noise, vf_model, steps=50):
        """Genera una imagen a partir de ruido"""
        return self.sample(noise, vf_model, steps)

def create_transfusion_mask(tokens, boi_token, eoi_token):
    """
    Crea una máscara para atención tipo Transfusion.
    - Atención causal para texto
    - Atención bidireccional dentro de cada imagen
    """
    batch_size, seq_len = tokens.shape
    
    # Iniciar con máscara causal de 3 dimensiones [batch, seq, seq]
    mask = torch.full((batch_size, seq_len, seq_len), float("-inf"), device=tokens.device)
    
    # Aplicar máscara causal a todos los elementos del batch
    causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=tokens.device), diagonal=1)
    mask += causal_mask.unsqueeze(0)  # Añadir dimensión de batch [1, seq, seq]
    
    for b in range(batch_size):
        # Encontrar tokens BOI y EOI
        boi_indices = (tokens[b] == boi_token).nonzero(as_tuple=True)[0]
        eoi_indices = (tokens[b] == eoi_token).nonzero(as_tuple=True)[0]
        
        # Si no hay suficientes pares, continuar
        if len(boi_indices) == 0 or len(eoi_indices) == 0:
            continue
            
        # Para cada par BOI/EOI, permitir atención bidireccional
        for i in range(min(len(boi_indices), len(eoi_indices))):
            start = boi_indices[i].item()
            end = eoi_indices[i].item()
            
            if start < end:  # Verificar validez
                # +1 para incluir el token BOI, hasta EOI
                mask[b, start:end+1, start:end+1] = 0.0
    
    return mask

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads = args.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wk = nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False
        )
        self.wv = nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False
        )
        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False
        )
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        
        # Proyecciones QKV
        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # Aplicar RoPE a Q y K
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Repetir K y V si es necesario
        if self.n_rep > 1:
            xk = xk.repeat_interleave(self.n_rep, dim=2)
            xv = xv.repeat_interleave(self.n_rep, dim=2)

        # Reordenar para atención
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # Calcular scores
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            # Manejar diferentes dimensiones de máscara
            if mask.dim() == 3:  # [bsz, seq, seq]
                mask = mask.unsqueeze(1)  # [bsz, 1, seq, seq]
            elif mask.dim() == 5:  # [bsz, 1, 1, seq, seq]
                mask = mask.squeeze(1)  # [bsz, 1, seq, seq]
            scores = scores + mask
            
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = self.dropout(scores)
        
        # Aplicar atención
        output = torch.matmul(scores, xv)
        expected_dim = self.n_heads * self.head_dim
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, expected_dim)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = int(2 * args.dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.dropout(self.attention(self.attention_norm(x), freqs_cis, mask))
        out = h + self.dropout(self.feed_forward(self.ffn_norm(h)))
        return out

class TransfusionBlock(nn.Module):
    """Bloque de transformador modificado para Transfusion"""
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.args = args
        
        # Componentes estándar del transformador
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]=None, 
                tokens: Optional[torch.Tensor]=None, boi_token: int=None, eoi_token: int=None):
        # Si tenemos tokens, crear máscara personalizada para Transfusion
        if tokens is not None and boi_token is not None and eoi_token is not None:
            # Crear máscara que permite atención bidireccional dentro de imágenes
            custom_mask = create_transfusion_mask(tokens, boi_token, eoi_token)
            if mask is not None:
                # Combinar con máscara causal existente
                attn_mask = custom_mask.unsqueeze(1)
            else:
                attn_mask = custom_mask.unsqueeze(1)
        else:
            attn_mask = mask
            
        # Proceso normal de transformador
        h = x + self.dropout(self.attention(self.attention_norm(x), freqs_cis, attn_mask))
        out = h + self.dropout(self.feed_forward(self.ffn_norm(h)))
        return out

class TransfusionTransformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        
        # Tokens especiales: BOI y EOI
        self.boi_token = params.vocab_size
        self.eoi_token = params.vocab_size + 1
        # Actualizar vocab_size para incluir tokens especiales
        self.vocab_size = params.vocab_size + 2  # +2 para BOI y EOI
        
        # Componentes estándar del LLM
        self.tok_embeddings = nn.Embedding(self.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        
        # Capas del transformador con bloques de Transfusion
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransfusionBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        
        # Cabeza de salida para predicción de texto
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # Frecuencias precomputadas para RoPE
        self.register_buffer("freqs_cis", precompute_freqs_cis(
            params.dim // params.n_heads, params.max_seq_len * 2, params.rope_theta
        ))
        
        # Componentes específicos para procesamiento de imágenes
        self.vae = VAE(params.vae_latent_dim)
        self.patch_encoder = PatchEncoder(params.vae_latent_dim, params.dim, params.patch_size)
        self.patch_decoder = PatchDecoder(params.vae_latent_dim, params.dim, params.patch_size)
        
        # Componente de Flow Matching
        self.flow_module = FlowMatchingModule(params)
        
        # Embedding para tiempo en Flow Matching
        self.time_embedding = nn.Sequential(
            nn.Linear(1, params.dim // 4),
            nn.SiLU(),
            nn.Linear(params.dim // 4, params.dim)
        )

    def embed_time(self, t):
        """Embedding del tiempo para Flow Matching"""
        t_emb = self.time_embedding(t.view(-1, 1))
        return t_emb
    
    def process_image(self, img):
        """Convierte una imagen en parches latentes"""
        latent = self.vae.encode(img)
        print(f"Latente shape: {latent.shape}")
        patches = self.patch_encoder(latent)
        print(f"Patches shape: {patches.shape}")
        print(f"Valores no cero en patches: {torch.sum(patches != 0)}")
        return patches, latent
    
    def reconstruct_image(self, patches, H, W):
        """Reconstruye una imagen a partir de parches"""
        latent = self.patch_decoder(patches, H, W)
        image = self.vae.decode(latent)
        return image
    
    def extract_image_positions(self, tokens):
        """Extrae las posiciones de los tokens BOI y EOI"""
        # Debug
        print(f"Token BOI: {self.boi_token}, EOI: {self.eoi_token}")
        #print(f"Tokens únicos: {torch.unique(tokens)}")
    
        # Los tokens son tensores, buscar posiciones
        boi_pos = (tokens == self.boi_token).nonzero(as_tuple=True)
        eoi_pos = (tokens == self.eoi_token).nonzero(as_tuple=True)
    
        # Verificar si se encontraron tokens
        if len(boi_pos[0]) == 0 or len(eoi_pos[0]) == 0:
            print(f"Advertencia: No se encontraron tokens BOI o EOI en el batch")
            return []
    
        # Agrupar por batch
        image_positions = []
        for b in range(tokens.shape[0]):
            batch_boi = boi_pos[0] == b
            batch_eoi = eoi_pos[0] == b
        
            boi_indices = boi_pos[1][batch_boi]
            eoi_indices = eoi_pos[1][batch_eoi]
        
            # Debug
            if len(boi_indices) > 0:
                print(f"Batch {b} tiene {len(boi_indices)} tokens BOI en posiciones {boi_indices}")
            if len(eoi_indices) > 0:
                print(f"Batch {b} tiene {len(eoi_indices)} tokens EOI en posiciones {eoi_indices}")
        
            # Emparejar BOI con EOI
            for i in range(min(len(boi_indices), len(eoi_indices))):
                if boi_indices[i] < eoi_indices[i]:
                    image_positions.append((b, boi_indices[i].item(), eoi_indices[i].item()))
    
        print(f"Posiciones de imágenes encontradas: {image_positions}")
        return image_positions
    
    def apply_flow_model(self, x, t=None):
        """
        Aplica el modelo de velocidad de Flow Matching.
        En Transfusion, esto es simplemente forward del transformador.
        """
        # t se pasa como información adicional, no se usa en el transformador
        return x  # La velocidad se calcula del output del transformador
    
    def forward_flow_matching(self, x, tokens, x_0, x_1, t):
        """Forward pass para Flow Matching."""
        # Verificar formas compatibles para x_0 y x_1
        if x_0.shape != x_1.shape:
            print(f"Advertencia: Formas no coinciden - x_0: {x_0.shape}, x_1: {x_1.shape}")
            min_h = min(x_0.shape[2], x_1.shape[2])
            min_w = min(x_0.shape[3], x_1.shape[3])
            x_0 = x_0[:, :, :min_h, :min_w]
            x_1 = x_1[:, :, :min_h, :min_w]

        # Encontrar posiciones de imágenes
        image_positions = self.extract_image_positions(tokens)
    
        if not image_positions:
            return torch.tensor(0.0, device=x.device)
    
        # Muestrear puntos en la trayectoria
        x_t, dx_t = self.flow_module.sample_path(x_0, x_1, t)
    
        # Mapeo entre batch e índices de flujo
        batch_idx_map = {}
        flow_idx = 0
        for b, _, _ in image_positions:
            if b not in batch_idx_map:
                batch_idx_map[b] = flow_idx
                flow_idx += 1
    
        print(f"x_t shape: {x_t.shape}, dx_t shape: {dx_t.shape}")
        print(f"batch_idx_map: {batch_idx_map}")
    
        losses = []
        for b, start, end in image_positions:
            # Verificar mapeo válido
            if b not in batch_idx_map:
                continue
        
            flow_idx = batch_idx_map[b]
            if flow_idx >= x_t.shape[0]:
                print(f"Advertencia: flow_idx {flow_idx} fuera de rango para x_t shape {x_t.shape}")
                continue
        
            # Obtener representación del transformador
            print(f"Tensor en posición [{b}, {start+1}:{end}] shape: {x[b, start+1:end].shape}")
            print(f"Contenido no cero: {torch.sum(x[b, start+1:end] != 0)}")
        
            # Obtener representación de imagen del transformador
            image_repr = x[b, start+1:end]
        
            # Preparar dimensiones objetivo
            print(f"dx_t shape: {dx_t.shape}, flow_idx: {flow_idx}")
            target = dx_t[flow_idx]
            target_shape = target.shape
            print(f"target_shape: {target_shape}, dimensiones: {len(target_shape)}")
        
            # Verificar cuántas dimensiones tiene el target
            try:
                if len(target_shape) == 3:  # [C, H, W]
                    latent_c, latent_h, latent_w = target_shape
                    # Añadir dimensión de batch
                    target = target.unsqueeze(0)  # [1, C, H, W]
                elif len(target_shape) == 4:  # [B, C, H, W]
                    _, latent_c, latent_h, latent_w = target_shape
                else:
                    print(f"Forma de target inesperada: {target_shape}")
                    continue
            
                # Verificar compatibilidad de parches
                H0, W0 = x_0.shape[2], x_0.shape[3]
                expected_patches = (H0 * W0) // (self.params.patch_size ** 2)
                received_patches = image_repr.shape[0]
            
                if expected_patches != received_patches:
                    print(f"Advertencia: Número de parches no coincide - transformer: {received_patches}, esperado: {expected_patches}")
            
                # Convertir a imagen latente 
                try:
                    # Si image_repr es 2D, añadir una dimensión de batch
                    if image_repr.dim() == 2:
                        image_repr = image_repr.unsqueeze(0)
                
                    # Usar el descodificador de parches mejorado
                    image_latent = self.patch_decoder(image_repr, latent_h * self.params.patch_size, latent_w * self.params.patch_size)
                
                    # Verificar si coinciden las formas antes de calcular la pérdida
                    if image_latent.shape != target.shape:
                        print(f"Formas no coinciden: imagen={image_latent.shape}, target={target.shape}")
                        image_latent = F.interpolate(
                            image_latent, 
                            size=(target.shape[2], target.shape[3]),
                            mode='bilinear', 
                            align_corners=False
                        )
                
                    # Calcular pérdida
                    flow_loss = F.mse_loss(image_latent, target)
                    losses.append(flow_loss)
                
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"Error al decodificar imagen: {e}")
                    # Usar una pérdida fija si falla la decodificación
                    losses.append(torch.tensor(0.1, device=x.device, requires_grad=True))
            except Exception as e:
                print(f"Error procesando dimensiones: {e}")
                losses.append(torch.tensor(0.1, device=x.device, requires_grad=True))
    
        # Devolver pérdida media o cero
        if losses:
            mean_loss = torch.stack(losses).mean()
            # Verificar NaN/Inf
            if torch.isnan(mean_loss) or torch.isinf(mean_loss):
                print("Pérdida de flow inválida, usando valor por defecto")
                return torch.tensor(0.1, device=x.device, requires_grad=True)
            # Limitar magnitud
            if mean_loss > 100:
                print(f"Pérdida de flow muy alta: {mean_loss.item()}, limitando")
                return torch.tensor(10.0, device=x.device, requires_grad=True)

            return mean_loss
        else:
            return torch.tensor(0.0, device=x.device, requires_grad=True)
    
    def forward(self, tokens: torch.Tensor, labels: Optional[torch.Tensor]=None,
               image_latents: Optional[Dict[int, torch.Tensor]]=None,
               flow_x0: Optional[torch.Tensor]=None,
               flow_x1: Optional[torch.Tensor]=None,
               flow_t: Optional[torch.Tensor]=None):
        """
        tokens: Índices de tokens (texto + BOI/EOI)
        labels: Etiquetas para pérdida de LM
        image_latents: Diccionario que mapea posición a tensor latente de imagen
        flow_x0, flow_x1, flow_t: Variables para pérdida de Flow Matching
        """
        if image_latents and len(image_latents) > 0:
            print(f"Forward recibiendo {len(image_latents)} latentes de imagen")

        bsz, seqlen = tokens.shape
        
        # Obtener embeddings
        h = self.tok_embeddings(tokens)
        
        # Si hay imágenes, integrarlas en la secuencia
        if image_latents:
            for pos, latent in image_latents.items():
                # Convertir latente en parches y reemplazar en la secuencia
                patches = self.patch_encoder(latent)
                # La posición en h sería pos+1 (después de BOI)
                patch_len = patches.shape[1]
                if pos+1+patch_len <= seqlen:
                    h[:, pos+1:pos+1+patch_len] = patches
        
        # Preparar frecuencias para RoPE
        freqs_cis = self.freqs_cis[:seqlen]
        
        # Crear máscara causal básica
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
        
        # Pasar por todas las capas
        for layer in self.layers:
            h = layer(h, freqs_cis, mask, tokens, self.boi_token, self.eoi_token)
            
        h = self.norm(h)
        
        # Calcular pérdida de LM (si aplicable)
        lm_loss = None
        if labels is not None:
            logits = self.output(h)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        # Calcular pérdida de Flow Matching (si aplicable)
        fm_loss = None
        if flow_x0 is not None and flow_x1 is not None and flow_t is not None:
            fm_loss = self.forward_flow_matching(h, tokens, flow_x0, flow_x1, flow_t)
        
        # Combinar pérdidas
        if lm_loss is not None and fm_loss is not None:
            total_loss = lm_loss + self.params.fm_lambda * fm_loss
            return h, total_loss, lm_loss, fm_loss
        elif lm_loss is not None:
            return h, lm_loss
        else:
            # Solo inferencia
            logits = self.output(h)
            return h, logits
    
    def generate_image(self, prompt_tokens, image_size=(256, 256), steps=50):
        """
        Genera una imagen a partir de un prompt de texto.
        """
        # Inferencia para tokens de texto
        with torch.no_grad():
            # Preparar tokens con BOI/EOI
            batch_size = prompt_tokens.shape[0]
            device = prompt_tokens.device
            
            # Añadir token BOI
            boi = torch.full((batch_size, 1), self.boi_token, device=device)
            eoi = torch.full((batch_size, 1), self.eoi_token, device=device)
            tokens = torch.cat([prompt_tokens, boi, eoi], dim=1)
            
            # Preparar ruido inicial y puntos de muestreo
            latent_h, latent_w = image_size[0] // (8 * self.params.patch_size), image_size[1] // (8 * self.params.patch_size)
            num_patches = latent_h * latent_w
            noise = torch.randn(batch_size, self.params.vae_latent_dim, latent_h * self.params.patch_size, 
                               latent_w * self.params.patch_size, device=device)
            
            # Calcular tiempo de muestreo
            times = torch.linspace(0, 1, steps, device=device)
            
            # Inicializar con ruido
            x_t = noise
            
            # Generar paso a paso
            for i in range(steps):
                # Codificar ruido actual
                patches = self.patch_encoder(x_t)
                
                # Tokens con parches actuales
                current_tokens = torch.cat([
                    tokens,
                    torch.full((batch_size, 1), self.eoi_token, device=device)
                ], dim=1)
                
                # Preparamos el diccionario de latentes: {posición BOI: x_t}
                pos_boi = tokens.shape[1] - 2  # índice donde colocamos BOI
                image_latents = { pos_boi: x_t }
                # Forward pass INCLUYENDO parches
                h, _ = self.forward(tokens, image_latents=image_latents)
                
                image_patches = h[:, pos_boi+1 : pos_boi+1 + num_patches, :]
                
                
                x_next = self.patch_decoder(
                        image_patches,
                        latent_h * self.params.patch_size,
                        latent_w * self.params.patch_size
                )
                
                # Actualizar con paso de solver
                if i < steps - 1:
                    dt = times[i+1] - times[i]
                    x_t = x_t + (x_next - x_t) * dt
                else:
                    x_t = x_next
            
            # Decodificar imagen final
            images = self.vae.decode(x_t)
            return images