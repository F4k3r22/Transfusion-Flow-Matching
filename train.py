#train.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
import logging
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
from PIL import Image
import torchvision.transforms as transforms
import os

from transformers.models.owlvit.processing_owlvit import ImagesKwargs

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TransfusionTrainer:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
        
        # Mover modelo a GPU
        self.model = self.model.to(self.device)
        
        # Configurar optimizador
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2)
        )
        
        # Scheduler - Warmup lineal + decay coseno
        def lr_lambda(step):
            if step < args.warmup_steps:
                return step / args.warmup_steps
            else:
                progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
                return 0.5 * (1.0 + np.cos(np.pi * progress))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # Mixed precision para ahorro de memoria
        self.scaler = GradScaler() if args.mixed_precision else None
        
        # Crear directorios de salida
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.sample_dir = self.output_dir / "samples"
        self.sample_dir.mkdir(exist_ok=True)
        
        # Variables de seguimiento
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Si hay checkpoint para reanudar, cargarlo
        if args.resume_checkpoint:
            self._resume_from_checkpoint(args.resume_checkpoint)
    
    def _resume_from_checkpoint(self, checkpoint_step):
        """Reanudar entrenamiento desde checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{checkpoint_step}.pt"
        if checkpoint_path.exists():
            logger.info(f"Reanudando desde checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            if self.scaler and 'scaler' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler'])
            self.global_step = checkpoint['step']
            if 'train_losses' in checkpoint:
                self.train_losses = checkpoint['train_losses']
            if 'val_losses' in checkpoint:
                self.val_losses = checkpoint['val_losses']
            if 'best_val_loss' in checkpoint:
                self.best_val_loss = checkpoint['best_val_loss']
            logger.info(f"Reanudado desde paso {self.global_step}")
        else:
            logger.warning(f"Checkpoint {checkpoint_path} no encontrado. Comenzando desde cero.")
    
    def save_checkpoint(self, is_best=False, save_as_latest=True):
        """Guardar checkpoint del modelo"""
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'step': self.global_step,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        if self.scaler is not None:
            checkpoint['scaler'] = self.scaler.state_dict()
        
        # Guardar checkpoint normal
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{self.global_step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Guardar como latest
        if save_as_latest:
            latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
            torch.save(checkpoint, latest_path)
        
        # Si es el mejor modelo hasta ahora, guardar otra copia
        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
        
        logger.info(f"Checkpoint guardado en paso {self.global_step}")
    
    def train_epoch(self, train_dataloader, epoch):
        """Entrenar durante una época"""
        self.model.train()
        epoch_loss = 0
        epoch_lm_loss = 0
        epoch_fm_loss = 0

        progress_bar = tqdm(train_dataloader, desc=f"Época {epoch+1}/{self.args.num_epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            # Mover batch a device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
    
            # Manejar datos de imagen si están presentes
            flow_x0 = None
            flow_x1 = None
            flow_t = None
            image_latents_dict = {}  # Nuevo: diccionario para mapear posiciones -> latentes
    
            if 'images' in batch and 'boi_indices' in batch:
                images = batch['images'].to(self.device)
                boi_indices = batch['boi_indices']
        
                print(f"Procesando batch con {len(boi_indices)} imágenes")
                print(f"BOI índices: {boi_indices}")

                # CORRECCIÓN 1: Crear mapeo índice → posición BOI
                batch_to_boi = {}
                for i, idx in enumerate(boi_indices):
                    if i < len(images):
                        # CORRECCIÓN 2: Eliminar .item() ya que idx ya es un entero
                        batch_to_boi[i] = idx
            
                # CORRECCIÓN 3: Todo este bloque debe estar FUERA del bucle for
                # Procesar imágenes con VAE para obtener latentes
                with torch.no_grad():
                    batch_latents = self.model.vae.encode(images)
            
                # Crear diccionario de posiciones BOI → latentes
                image_latents_dict = {}
                for batch_idx, boi_pos in batch_to_boi.items():
                    # Asociar posición BOI con latente correspondiente
                    image_latents_dict[boi_pos] = batch_latents[batch_idx:batch_idx+1]
            
                print(f"Creado image_latents_dict con {len(image_latents_dict)} entradas")
                # Imprimir claves para verificar
                print(f"Claves en image_latents_dict: {list(image_latents_dict.keys())}")
            
                # Para Flow Matching
                x_1 = batch_latents  # Latente objetivo
                x_0 = torch.randn_like(x_1) * 0.8  # Ruido inicial
                t = torch.rand(len(images), device=self.device)
            
                flow_x0 = x_0
                flow_x1 = x_1
                flow_t = t
        
            # Forward pass con mixed precision si está activado
            if self.args.mixed_precision:
                with autocast():
                    if flow_x0 is not None:
                        # NUEVO: Pasar image_latents_dict al modelo
                        _, total_loss, lm_loss, fm_loss = self.model(
                            tokens=input_ids,
                            labels=labels,
                            image_latents=image_latents_dict,  # Añadir este parámetro
                            flow_x0=flow_x0,
                            flow_x1=flow_x1,
                            flow_t=flow_t
                        )
                    else:
                        # Batch de solo texto
                        _, total_loss = self.model(tokens=input_ids, labels=labels)
                        lm_loss = total_loss
                        fm_loss = torch.tensor(0.0, device=self.device)
                
                    # Verificar NaNs y estabilizar pérdida
                    if torch.isnan(total_loss):
                        print(f"¡Advertencia! Pérdida NaN detectada en paso {self.global_step}")
                        # Usar valor seguro para evitar que el entrenamiento se detenga
                        total_loss = torch.tensor(10.0, device=self.device, requires_grad=True)
                        lm_loss = torch.tensor(10.0, device=self.device)
                        fm_loss = torch.tensor(0.0, device=self.device)
                
                    # Escalar pérdida para acumulación de gradientes
                    total_loss = total_loss / self.args.gradient_accumulation_steps
                
                    # Backpropagation con mixed precision
                    if not torch.isnan(total_loss) and not torch.isinf(total_loss):
                        self.scaler.scale(total_loss).backward()
    
                        if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                            # Comprobar si hay gradientes inválidos
                            found_inf = False
                            for param in self.model.parameters():
                                if param.grad is not None:
                                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                        found_inf = True
                                        break
        
                            if not found_inf:
                                try:
                                    # Desescalar los gradientes
                                    self.scaler.unscale_(self.optimizer)
                                    # Recortar gradientes
                                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                                    # Actualizar parámetros
                                    self.scaler.step(self.optimizer)
                                    self.scaler.update()
                                except RuntimeError as e:
                                    print(f"Error en el optimizador: {e}")
                                    # Reiniciar el optimizador en caso de error
                                    self.optimizer.zero_grad()
                                    self.scaler.update()
                            else:
                                print("Gradientes inválidos detectados, saltando paso de optimización")
                                self.optimizer.zero_grad()
                                self.scaler.update()
    
                    else:
                        print("Pérdida inválida, saltando backward pass")
                        self.optimizer.zero_grad()
            else:
                # Sin mixed precision
                if flow_x0 is not None:
                    _, total_loss, lm_loss, fm_loss = self.model(
                        tokens=input_ids,
                        labels=labels,
                        flow_x0=flow_x0,
                        flow_x1=flow_x1,
                        flow_t=flow_t
                    )
                else:
                    # Batch de solo texto
                    _, total_loss = self.model(tokens=input_ids, labels=labels)
                    lm_loss = total_loss
                    fm_loss = torch.tensor(0.0, device=self.device)
                
                # Escalar pérdida para acumulación de gradientes
                total_loss = total_loss / self.args.gradient_accumulation_steps
                
                # Backpropagation
                total_loss.backward()
                
                if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                    # Clipping de gradientes
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    
                    # Paso del optimizador
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    
                    self.global_step += 1
            
            # Seguimiento de pérdidas
            epoch_loss += total_loss.item() * self.args.gradient_accumulation_steps
            epoch_lm_loss += lm_loss.item()
            if flow_x0 is not None:
                epoch_fm_loss += fm_loss.item()
            
            # Actualizar barra de progreso
            lr = self.scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': epoch_loss / (batch_idx + 1),
                'lm_loss': epoch_lm_loss / (batch_idx + 1),
                'fm_loss': epoch_fm_loss / (batch_idx + 1) if batch_idx > 0 else 0,
                'lr': f'{lr:.2e}'
            })
            
            # Generar muestras periódicamente
            if self.global_step % self.args.sample_every == 0:
                self.generate_samples()
            
            # Guardar checkpoint periódicamente
            if self.global_step % self.args.checkpoint_every == 0:
                self.save_checkpoint()
            
            # Evaluar periódicamente
            if self.global_step % self.args.eval_every == 0:
                val_loss = self.evaluate(self.val_dataloader)
                self.val_losses.append((self.global_step, val_loss))
                
                # Guardar mejor modelo
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(is_best=True)
                
                # Plotear curvas de pérdida
                self.plot_losses()
                
                # Volver a modo de entrenamiento
                self.model.train()
            
            # Parada temprana basada en pasos
            if self.global_step >= self.args.max_steps:
                logger.info(f"Alcanzado máximo de pasos {self.args.max_steps}. Deteniendo entrenamiento.")
                return True  # Señala que debemos detener el entrenamiento
        
        # Fin de la época
        epoch_loss /= (batch_idx + 1)
        epoch_lm_loss /= (batch_idx + 1)
        epoch_fm_loss /= (batch_idx + 1) if batch_idx > 0 else 1
        
        self.train_losses.append((self.global_step, epoch_loss))
        
        logger.info(f"Época {epoch+1}/{self.args.num_epochs} completada. "
                   f"Pérdida: {epoch_loss:.4f}, Pérdida LM: {epoch_lm_loss:.4f}, "
                   f"Pérdida FM: {epoch_fm_loss:.4f}")
        
        # Si hemos alcanzado el máximo de pasos, indicar que debemos parar
        return self.global_step >= self.args.max_steps
    
    def evaluate(self, dataloader):
        """Evaluar el modelo"""
        self.model.eval()
        total_loss = 0
        total_lm_loss = 0
        total_fm_loss = 0
        num_batches = 0
        
        logger.info("Ejecutando evaluación...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluando"):
                # Mover batch a device
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Manejar datos de imagen si están presentes
                flow_x0 = None
                flow_x1 = None
                flow_t = None
                
                if 'images' in batch:
                    images = batch['images'].to(self.device)
                    
                    # Procesar imágenes con VAE para obtener latentes
                    image_latents = self.model.vae.encode(images)
                    
                    # Para Flow Matching
                    batch_size = images.shape[0]
                    x_1 = image_latents  # Latente objetivo
                    x_0 = torch.randn_like(x_1)  # Ruido inicial
                    
                    # Muestrear pasos de tiempo
                    t = torch.rand(batch_size, device=self.device)
                    
                    # Configurar para flow matching
                    flow_x0 = x_0
                    flow_x1 = x_1
                    flow_t = t
                
                # Forward pass
                if flow_x0 is not None:
                    _, loss, lm_loss, fm_loss = self.model(
                        tokens=input_ids,
                        labels=labels,
                        flow_x0=flow_x0,
                        flow_x1=flow_x1,
                        flow_t=flow_t
                    )
                else:
                    # Batch de solo texto
                    _, loss = self.model(tokens=input_ids, labels=labels)
                    lm_loss = loss
                    fm_loss = torch.tensor(0.0, device=self.device)
                
                total_loss += loss.item()
                total_lm_loss += lm_loss.item()
                if flow_x0 is not None:
                    total_fm_loss += fm_loss.item()
                
                num_batches += 1
        
        # Calcular pérdidas promedio
        avg_loss = total_loss / num_batches
        avg_lm_loss = total_lm_loss / num_batches
        avg_fm_loss = total_fm_loss / num_batches if num_batches > 0 else 0
        
        logger.info(f"Pérdida de validación: {avg_loss:.4f}, Pérdida LM: {avg_lm_loss:.4f}, "
                   f"Pérdida FM: {avg_fm_loss:.4f}")
        
        return avg_loss
    
    def generate_samples(self):
        """Generar y guardar imágenes de muestra a partir de prompts de texto"""
        self.model.eval()
        logger.info(f"Generando muestras en paso {self.global_step}...")
        
        # Definir algunos prompts de muestra
        sample_prompts = [
            "Un gato sentado en un sofá",
            "Una hermosa puesta de sol sobre las montañas",
            "Una manzana roja sobre una mesa de madera",
            "Un pequeño barco navegando en un lago tranquilo"
        ]
        
        # Tokenizar prompts (adaptado al uso real)
        # Nota: En una implementación real, usaríamos un tokenizador adecuado
        # Aquí simulamos tokens para pruebas
        max_len = 20
        tokens = []
        for prompt in sample_prompts:
            # Simular tokenización - en implementación real esto usaría el tokenizador
            token_ids = torch.randint(0, self.model.vocab_size-2, (1, max_len))
            tokens.append(token_ids)
        
        input_tokens = torch.cat(tokens, dim=0).to(self.device)
        
        # Generar imágenes
        with torch.no_grad():
            try:
                images = self.model.generate_image(
                    input_tokens, 
                    image_size=(self.args.image_size, self.args.image_size),
                    steps=self.args.sampling_steps
                )
                
                # Guardar imágenes individuales
                for i, (image, prompt) in enumerate(zip(images, sample_prompts)):
                    # Convertir a imagen PIL
                    img = transforms.ToPILImage()(image.cpu())
                    
                    # Guardar con prompt como nombre de archivo
                    prompt_slug = prompt.lower().replace(" ", "_")[:30]
                    filename = f"paso_{self.global_step}_{i}_{prompt_slug}.png"
                    img.save(self.sample_dir / filename)
                
                # Crear una cuadrícula de todas las imágenes
                grid = make_grid(images, nrow=2)
                save_image(grid, self.sample_dir / f"grid_paso_{self.global_step}.png")
                
            except Exception as e:
                logger.error(f"Error al generar muestras: {e}")
        
        self.model.train()
    
    def plot_losses(self):
        """Plotear pérdidas de entrenamiento y validación"""
        plt.figure(figsize=(10, 6))
        
        if self.train_losses:
            steps, losses = zip(*self.train_losses)
            plt.plot(steps, losses, label='Pérdida de entrenamiento')
        
        if self.val_losses:
            steps, losses = zip(*self.val_losses)
            plt.plot(steps, losses, label='Pérdida de validación')
        
        plt.xlabel('Pasos')
        plt.ylabel('Pérdida')
        plt.title('Pérdida de entrenamiento y validación')
        plt.legend()
        plt.grid(True)
        
        # Guardar plot
        plt.savefig(self.output_dir / 'curva_perdida.png')
        plt.close()
    
    def train(self, train_dataloader, val_dataloader):
        """Entrenamiento completo"""
        self.val_dataloader = val_dataloader  # Guardar para evaluaciones periódicas
        
        logger.info("Comenzando entrenamiento...")
        
        # Mostrar uso de GPU inicial
        if torch.cuda.is_available():
            logger.info(f"GPU inicial: {torch.cuda.memory_allocated(self.device)/1e9:.2f} GB")
        
        for epoch in range(self.args.num_epochs):
            # Entrenar una época
            stop_training = self.train_epoch(train_dataloader, epoch)
            
            # Si se alcanzó el máximo de pasos, detener entrenamiento
            if stop_training:
                break
        
        # Checkpoint final
        self.save_checkpoint()
        
        logger.info("¡Entrenamiento completado!")
        
        # Evaluación final
        final_loss = self.evaluate(val_dataloader)
        logger.info(f"Pérdida de validación final: {final_loss:.4f}")
        
        # Generar muestras finales
        sample_dir_final = self.output_dir / "muestras_finales"
        sample_dir_final.mkdir(exist_ok=True)
        self.generate_samples()
        
        return self.model


def train_transfusion_model():
    parser = argparse.ArgumentParser(description="Entrenar modelo Transfusion")
    
    # Configuración del modelo
    parser.add_argument("--dim", type=int, default=768, help="Dimensión del modelo")
    parser.add_argument("--n-layers", type=int, default=12, help="Número de capas")
    parser.add_argument("--n-heads", type=int, default=12, help="Número de cabezas de atención")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Tamaño del vocabulario")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Longitud máxima de secuencia")
    parser.add_argument("--dropout", type=float, default=0.1, help="Tasa de dropout")
    
    # Configuración específica de Transfusion
    parser.add_argument("--vae-latent-dim", type=int, default=8, help="Dimensión latente VAE")
    parser.add_argument("--patch-size", type=int, default=2, help="Tamaño de parche")
    parser.add_argument("--image-size", type=int, default=64, help="Tamaño de imagen")
    parser.add_argument("--fm-lambda", type=float, default=5.0, help="Peso pérdida Flow Matching")
    parser.add_argument("--sampling-steps", type=int, default=25, help="Pasos para muestreo")
    
    # Configuración de entrenamiento
    parser.add_argument("--batch-size", type=int, default=8, help="Tamaño de batch")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Tasa de aprendizaje")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Norma máxima de gradiente")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Pasos de warmup")
    parser.add_argument("--max-steps", type=int, default=50000, help="Máximo de pasos de entrenamiento")
    parser.add_argument("--num-epochs", type=int, default=10, help="Número de épocas")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Pasos de acumulación de gradiente")
    parser.add_argument("--mixed-precision", action="store_true", help="Usar entrenamiento de precisión mixta")
    parser.add_argument("--text-only-prob", type=float, default=0.5, help="Probabilidad de muestrear batches solo texto")
    
    # Configuración de IO
    parser.add_argument("--data-dir", type=str, required=True, help="Directorio de datos")
    parser.add_argument("--output-dir", type=str, default="output", help="Directorio de salida")
    parser.add_argument("--resume-checkpoint", type=str, help="Reanudar desde checkpoint")
    parser.add_argument("--checkpoint-every", type=int, default=1000, help="Guardar checkpoint cada N pasos")
    parser.add_argument("--eval-every", type=int, default=1000, help="Evaluar cada N pasos")
    parser.add_argument("--sample-every", type=int, default=2000, help="Generar muestras cada N pasos")
    parser.add_argument("--num-workers", type=int, default=4, help="Número de workers del dataloader")
    parser.add_argument("--gpu-id", type=int, default=0, help="ID de GPU")
    
    args = parser.parse_args()
    
    # Este código asume que tienes implementado el dataset y modelo en archivos separados
    # Importar dataset
    from datasetfl import TransfusionDataset, collate_fn
    
    # Cargar datasets
    train_dataset = TransfusionDataset(
        args.data_dir, 
        "train", 
        text_only_prob=args.text_only_prob, 
        max_seq_len=args.max_seq_len
    )
    
    val_dataset = TransfusionDataset(
        args.data_dir, 
        "val", 
        text_only_prob=args.text_only_prob, 
        max_seq_len=args.max_seq_len
    )
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Importar modelo
    from model import ModelArgs, TransfusionTransformer
    text_meta = np.load(os.path.join(args.data_dir, 'text_train_meta.npy'), allow_pickle=True).item()
    real_vocab_size = text_meta['vocab_size']
    boi_token_id = text_meta['boi_token_id']
    eoi_token_id = text_meta['eoi_token_id']

    print(f"Usando tamaño de vocabulario: {real_vocab_size}")
    print(f"BOI token ID: {boi_token_id}, EOI token ID: {eoi_token_id}")
    
    # Inicializar argumentos del modelo
    model_args = ModelArgs(
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        vocab_size=real_vocab_size,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        vae_latent_dim=args.vae_latent_dim,
        patch_size=args.patch_size,
        max_img_size=(args.image_size, args.image_size),
        fm_lambda=args.fm_lambda
    )
    
    # Crear modelo
    model = TransfusionTransformer(model_args)
    model.boi_token = boi_token_id
    model.eoi_token = eoi_token_id
    
    # Contar parámetros
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Modelo tiene {num_params/1e6:.2f}M parámetros")
    
    # Crear trainer
    trainer = TransfusionTrainer(model, args)
    
    # Entrenar modelo
    trained_model = trainer.train(train_dataloader, val_dataloader)
    
    return trained_model

if __name__ == "__main__":
    train_transfusion_model()