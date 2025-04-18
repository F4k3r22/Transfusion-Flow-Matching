# datasetfl.py

import os
import random
import torch
import numpy as np
from datasets import load_dataset
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class TransfusionTokenizer:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B",
        max_seq_length: int = 512,  # Reducido para pruebas iniciales
    ):
        # Inicializar tokenizador base
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_seq_length = max_seq_length
        
        # Añadir tokens especiales para Transfusion
        special_tokens = {
            "additional_special_tokens": ["<BOI>", "<EOI>"]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Guardar IDs para un acceso fácil
        self.boi_token_id = self.tokenizer.convert_tokens_to_ids("<BOI>")
        self.eoi_token_id = self.tokenizer.convert_tokens_to_ids("<EOI>")
        
        print(f"Tamaño de vocabulario: {len(self.tokenizer)}")
        print(f"BOI token ID: {self.boi_token_id}")
        print(f"EOI token ID: {self.eoi_token_id}")
    
    def encode_text(self, text, add_eos=True):
        """Tokeniza texto con truncamiento"""
        return self.tokenizer.encode(
            text,
            add_special_tokens=add_eos,
            padding=False,
            truncation=True,
            max_length=self.max_seq_length
        )
    
    def encode_text_with_image_placeholder(self, text):
        """Tokeniza texto e inserta tokens BOI/EOI"""
        tokens = self.encode_text(text, add_eos=False)
        
        # Añadir tokens especiales
        tokens = tokens + [self.boi_token_id, self.eoi_token_id]
        
        # Asegurar que no exceda la longitud máxima
        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length-1] + [self.tokenizer.eos_token_id]
        
        return tokens


class DataPreparation:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B",
        max_seq_length: int = 512,
        num_proc: int = 8,
        output_dir: str = "data",
        val_size: float = 0.05,  # Mayor validación para pruebas iniciales
        image_size: int = 64,    # Resolución baja para pruebas
        flickr_path: str = "path/to/flickr8k"
    ):
        self.max_seq_length = max_seq_length
        self.num_proc = num_proc
        self.output_dir = Path(output_dir)
        self.val_size = val_size
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.image_size = image_size
        self.flickr_path = Path(flickr_path)
        
        # Inicializar tokenizador personalizado para Transfusion
        self.tokenizer = TransfusionTokenizer(model_name, max_seq_length)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
    
    def prepare_text_dataset(self, sample_size=10000):
        """Preparar OpenWebText para entrenamiento de solo texto"""
        print("Cargando dataset de texto...")
        dataset = load_dataset("openwebtext", num_proc=self.num_proc)
        
        # Seleccionar una muestra para pruebas
        dataset = dataset['train'].select(range(min(sample_size, len(dataset['train']))))
        
        # Split train/val
        split_dataset = dataset.train_test_split(
            test_size=self.val_size,
            seed=2357,
            shuffle=True
        )
        split_dataset['val'] = split_dataset.pop('test')
        
        print("Tokenizando textos...")
        tokenized = split_dataset.map(
            lambda example: {
                'ids': self.tokenizer.encode_text(example['text']),
                'len': len(self.tokenizer.encode_text(example['text']))
            },
            remove_columns=['text'],
            desc="Tokenizando",
            num_proc=self.num_proc
        )
        
        # Guardar en archivos binarios
        for split, dset in tokenized.items():
            arr_len = np.sum(dset['len'], dtype=np.uint64)
            filename = self.output_dir / f'text_{split}.bin'
            
            print(f"Guardando {split}.bin...")
            dtype = np.uint16  # Suficiente para Llama tokens
            arr = np.memmap(str(filename), dtype=dtype, mode='w+', shape=(arr_len,))
            
            idx = 0
            for batch in tqdm(dset.iter(batch_size=100), desc=f'Escribiendo {filename}'):
                arr_batch = np.concatenate(batch['ids'])
                arr[idx:idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            
            arr.flush()
            
            meta = {
                'total_tokens': arr_len,
                'vocab_size': len(self.tokenizer.tokenizer),
                'max_seq_length': self.max_seq_length,
                'boi_token_id': self.tokenizer.boi_token_id,
                'eoi_token_id': self.tokenizer.eoi_token_id
            }
            np.save(str(self.output_dir / f'text_{split}_meta.npy'), meta)
        
        print("Procesamiento de texto completado.")
    
    def prepare_flickr8k(self, num_samples=1000):
        """Preparar dataset Flickr8k para entrenamiento texto-imagen"""
        print(f"Preparando dataset Flickr8k ({num_samples} muestras)...")
        
        # Cargar archivo de captions
        captions_file = self.flickr_path / "captions.txt"
        
        if not captions_file.exists():
            print(f"Error: No se encontró el archivo {captions_file}")
            return
        
        # Leer captions
        captions = []
        with open(captions_file, 'r', encoding='utf-8') as f:
            # Omitir encabezado si existe
            first_line = f.readline().strip()
            if not first_line.startswith("image"):
                captions.append(first_line.split(",", 1))
            
            for line in f:
                parts = line.strip().split(",", 1)
                if len(parts) == 2:
                    captions.append(parts)
        
        # Agrupar por imagen
        image_captions = {}
        for img_name, caption in captions:
            if img_name not in image_captions:
                image_captions[img_name] = []
            image_captions[img_name].append(caption)
        
        # Seleccionar muestra aleatoria
        all_images = list(image_captions.keys())
        random.shuffle(all_images)
        selected_images = all_images[:num_samples]
        
        # Crear directorio para datos procesados
        image_data_dir = self.output_dir / "flickr8k_processed"
        image_data_dir.mkdir(exist_ok=True, parents=True)
        
        # Procesar imágenes y captions
        processed_data = []
        
        for img_name in tqdm(selected_images, desc="Procesando imágenes"):
            img_path = self.flickr_path / "Images" / img_name
            if not img_path.exists():
                continue
            
            # Seleccionar una caption aleatoria
            caption = random.choice(image_captions[img_name])
            
            try:
                # Cargar y transformar imagen
                img = Image.open(img_path).convert('RGB')
                img_tensor = self.transform(img)
                
                # Tokenizar caption con placeholders
                tokens = self.tokenizer.encode_text_with_image_placeholder(caption)
                
                # Guardar datos
                img_torch_path = image_data_dir / f"{img_name}.pt"
                torch.save(img_tensor, img_torch_path)
                
                processed_data.append({
                    'image_file': str(img_torch_path.relative_to(self.output_dir)),
                    'caption': caption,
                    'tokens': tokens
                })
            except Exception as e:
                print(f"Error procesando {img_name}: {e}")
        
        # Dividir en train/val
        random.shuffle(processed_data)
        val_size = int(len(processed_data) * self.val_size)
        train_data = processed_data[val_size:]
        val_data = processed_data[:val_size]
        
        # Guardar metadatos
        torch.save(train_data, self.output_dir / "flickr8k_train.pt")
        torch.save(val_data, self.output_dir / "flickr8k_val.pt")
        
        print(f"Dataset de imágenes procesado: {len(train_data)} train, {len(val_data)} val")
    
    def prepare_all(self):
        """Preparar ambos datasets"""
        self.prepare_text_dataset()
        self.prepare_flickr8k()
        print("Preparación de datos completa.")


class TransfusionDataset(Dataset):
    """Dataset para entrenamiento de Transfusion con texto e imágenes"""
    def __init__(
        self, 
        data_dir,
        split="train", 
        text_only_prob=0.5,  # Probabilidad de muestras solo texto
        max_seq_len=512
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_seq_len = max_seq_len
        self.text_only_prob = text_only_prob
        
        # Cargar metadatos
        self.text_meta = np.load(
            str(self.data_dir / f'text_{split}_meta.npy'), 
            allow_pickle=True
        ).item()
        
        # Cargar datos de imagen (pequeño para memoria)
        self.image_data = torch.load(self.data_dir / f"flickr8k_{split}.pt")
        
        # Cargar memmap para texto
        self.text_data = np.memmap(
            str(self.data_dir / f'text_{split}.bin'),
            dtype=np.uint16,
            mode='r'
        )
        
        # Calcular número total de secuencias
        tokens_per_seq = self.max_seq_len - 1  # -1 para dejar espacio para EOS
        self.num_text_seqs = int(self.text_meta['total_tokens'] // tokens_per_seq)
        
        # Recordar tamaños para muestreo
        self.num_image_samples = len(self.image_data)
        
        print(f"Dataset '{split}' cargado:")
        print(f"  Número de secuencias de texto: {self.num_text_seqs}")
        print(f"  Número de muestras de imagen: {self.num_image_samples}")
    
    def __len__(self):
        # Calcular tamaño equivalente: si text_only_prob=0.5,
        # queremos aproximadamente mitad texto y mitad imagen
        # Limitado por el tipo menos común
        text_equivalent = self.num_text_seqs / self.text_only_prob
        image_equivalent = self.num_image_samples / (1 - self.text_only_prob)
        return int(min(text_equivalent, image_equivalent))
    
    def get_text_sequence(self, idx):
        """Obtener una secuencia de texto desde el memmap"""
        tokens_per_seq = self.max_seq_len - 1
        i = idx % self.num_text_seqs
        start_idx = i * tokens_per_seq
        end_idx = start_idx + tokens_per_seq
        
        # Obtener tokens desde memmap
        tokens = self.text_data[start_idx:end_idx].copy()
        
        # Crear etiquetas (desplazadas uno a la derecha para next-token prediction)
        x = np.zeros(self.max_seq_len, dtype=np.int64)
        y = np.zeros(self.max_seq_len, dtype=np.int64)
        
        x[:len(tokens)] = tokens
        y[:len(tokens)-1] = tokens[1:]
        y[len(tokens)-1] = self.text_meta['eoi_token_id']  # EOS como predicción final
        
        return {
            'input_ids': torch.tensor(x, dtype=torch.long),
            'labels': torch.tensor(y, dtype=torch.long),
            'is_text_only': True
        }
    
    def get_image_text_pair(self, idx):
        """Obtener un par imagen-texto con espacio garantizado para parches de imagen"""
        i = idx % self.num_image_samples
        sample = self.image_data[i]
    
        # Cargar imagen
        img_path = self.data_dir / sample['image_file']
        image = torch.load(img_path)
    
        # Obtener tokens originales
        orig_tokens = sample['tokens']
    
        # Calcular el número de parches necesarios (basado en el tamaño de imagen y de parche)
        # Para una imagen de 64x64 con latentes de 8x8 y patch_size=2, necesitamos 16 parches (4x4)
        # Esto debe coincidir con la configuración usada en tu modelo
        img_height, img_width = 64, 64  # Asume tamaño predeterminado de imagen
        latent_factor = 16  # Factor de compresión de VAE (típico en VAEs para imágenes)
        patch_size = 2  # Debe coincidir con params.patch_size en el modelo
    
        # Calcular tamaño de latente y número de parches
        latent_size = img_height // latent_factor  # Ej: 64/8 = 8
        num_patches = (latent_size // patch_size) ** 2  # Ej: (8/2)^2 = 16
    
        print(f"Imagen {img_path.name}: calculados {num_patches} parches necesarios")
    
        # Encontrar índices de tokens BOI y EOI
        try:
            boi_idx = orig_tokens.index(self.text_meta['boi_token_id'])
            eoi_idx = orig_tokens.index(self.text_meta['eoi_token_id'])
        except ValueError:
            print(f"Error: No se encontraron tokens BOI/EOI en la muestra {i}")
            return self.get_text_sequence(idx)  # Fallback a texto
    
        # Crear una nueva lista de tokens con espacio suficiente para los parches
        tokens = []
    
        # Copiar tokens hasta BOI (inclusive)
        for j in range(boi_idx + 1):
            tokens.append(orig_tokens[j])
    
        # Añadir tokens de relleno para parches (usar 0 o un token especial)
        # Estos serán reemplazados por las representaciones de parches durante el entrenamiento
        pad_token = 0  # Token de relleno, podría ser cualquiera que no interfiera
        tokens.extend([pad_token] * num_patches)
    
        # Añadir EOI y cualquier token restante
        tokens.append(self.text_meta['eoi_token_id'])
        for j in range(eoi_idx + 1, len(orig_tokens)):
            tokens.append(orig_tokens[j])
    
        # Recalcular índice de BOI (debería ser el mismo)
        boi_idx = tokens.index(self.text_meta['boi_token_id'])
    
        # Verificar que hay espacio suficiente entre BOI y EOI
        eoi_idx = tokens.index(self.text_meta['eoi_token_id'])
        patch_space = eoi_idx - boi_idx - 1
    
        print(f"Espacio para parches: {patch_space}, requerido: {num_patches}")
    
        # Truncar si es necesario para ajustar a max_seq_len
        if len(tokens) > self.max_seq_len:
            print(f"Advertencia: Secuencia demasiado larga ({len(tokens)}), truncando a {self.max_seq_len}")
            tokens = tokens[:self.max_seq_len-1] + [self.text_meta['eoi_token_id']]
        
            # Recalcular índices tras truncamiento
            if self.text_meta['boi_token_id'] in tokens:
                boi_idx = tokens.index(self.text_meta['boi_token_id'])
            else:
                print("Error: BOI perdido tras truncamiento")
                return self.get_text_sequence(idx)  # Fallback a texto
    
        # Crear input y labels
        x = np.zeros(self.max_seq_len, dtype=np.int64)
        y = np.zeros(self.max_seq_len, dtype=np.int64)
    
        # Copiar tokens al input
        x[:len(tokens)] = tokens
    
        # Para predicción next-token, desplazar 1 a la derecha
        y[:len(tokens)-1] = tokens[1:]
        y[len(tokens)-1] = self.text_meta['eoi_token_id']  # EOS como predicción final
    
        return {
            'input_ids': torch.tensor(x, dtype=torch.long),
            'labels': torch.tensor(y, dtype=torch.long),
            'image': image,
            'is_text_only': False,
            'boi_idx': boi_idx
        }
    
    def __getitem__(self, idx):
        # Decidir si devolver texto o par imagen-texto
        is_text = random.random() < self.text_only_prob
        
        if is_text:
            return self.get_text_sequence(idx)
        else:
            return self.get_image_text_pair(idx)


def collate_fn(batch):
    """Función de collate personalizada para manejar muestras mixtas"""
    # Separar muestras de texto y de imagen
    text_only = [item for item in batch if item['is_text_only']]
    image_text = [item for item in batch if not item['is_text_only']]
    
    result = {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
        'is_text_only': [item['is_text_only'] for item in batch]
    }
    
    if image_text:
        result['images'] = torch.stack([item['image'] for item in image_text])
        # Guardar posiciones como enteros Python, no como tensores
        result['boi_indices'] = [int(item['boi_idx']) for item in image_text]
        # Aquí está el cambio importante - guardar los índices de batch junto con BOI
        batch_indices = []
        boi_positions = []
        
        for i, item in enumerate(batch):
            if not item.get('is_text_only', True):
                batch_indices.append(i)  # Índice en el batch
                boi_positions.append(item['boi_idx'])  # Posición BOI
                
        result['batch_indices'] = batch_indices
        result['boi_indices'] = boi_positions
    
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Preparar datasets para Transfusion")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B", help="Nombre del modelo pre-entrenado")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Longitud máxima de secuencia")
    parser.add_argument("--num-proc", type=int, default=8, help="Número de procesos")
    parser.add_argument("--output-dir", type=str, default="data", help="Directorio de salida")
    parser.add_argument("--val-size", type=float, default=0.05, help="Tamaño del conjunto de validación")
    parser.add_argument("--image-size", type=int, default=64, help="Tamaño de las imágenes")
    parser.add_argument("--flickr-path", type=str, required=True, help="Ruta al dataset Flickr8k")
    parser.add_argument("--num-images", type=int, default=1000, help="Número de imágenes a procesar")
    
    args = parser.parse_args()
    
    data_prep = DataPreparation(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        num_proc=args.num_proc,
        output_dir=args.output_dir,
        val_size=args.val_size,
        image_size=args.image_size,
        flickr_path=args.flickr_path
    )
    
    # Preparar ambos datasets
    data_prep.prepare_text_dataset(sample_size=10000)
    data_prep.prepare_flickr8k(num_samples=args.num_images)
    
    # Ejemplo de cómo crear dataloader
    dataset = TransfusionDataset(args.output_dir, "train")
    dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn, shuffle=True)
    
    # Mostrar un ejemplo
    print("\nEjemplo de batch:")
    batch = next(iter(dataloader))
    print(f"Input shape: {batch['input_ids'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    print(f"Tipos de muestra: {batch['is_text_only']}")
    if 'images' in batch:
        print(f"Imágenes shape: {batch['images'].shape}")


if __name__ == '__main__':
    main()