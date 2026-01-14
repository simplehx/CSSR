import logging
import datetime
import os
import re
current_file_name = os.path.basename(__file__)
match = re.search(r'v(\d+)', current_file_name, re.IGNORECASE)
logging.getLogger('faiss').setLevel(logging.WARNING)
v=int(re.search(r'v(\d+)', current_file_name, re.IGNORECASE).group(1))
logging_format = "%(asctime)s - %(levelname)s - %(message)s"
date_format = "%Y-%m-%d %H:%M:%S"
dt_converter = lambda *args: (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=8)).timetuple()
logging.Formatter.converter = dt_converter
logging.basicConfig(filename=f"./save/log/train_v{v}.log", level=logging.INFO, format=logging_format, datefmt=date_format)

import random
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import lmdb
from model_utils import *
import math
from haversine import haversine, haversine_vector, Unit
import pickle
from st_moe_pytorch import MoE, SparseMoEBlock
import faiss

cuda_indices = list(range(torch.cuda.device_count()))
valid_cuda = []
for cuda_index in cuda_indices:
    cuda_memory_usage = torch.cuda.mem_get_info(cuda_index)
    memory_usage = (cuda_memory_usage[1] - cuda_memory_usage[0]) // 1024 // 1024
    if memory_usage < 1024:
        valid_cuda.append(cuda_index)

if valid_cuda:
    cuda_idx = valid_cuda[0]
    print(f"{current_file_name} in cuda:{cuda_idx}")
else:
    sys.exit(1)
    print(f"not valid device")

random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed)
else:
    torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

img_env = lmdb.open(f"./save/dataset/StreetSem/", readonly=True)
img_txn = img_env.begin()

img_df = pd.read_csv(f"./meta_data/img_df.csv", header=0)
img_df = img_df[img_df["city"] == city_name]
img_list = [{"key": f"{index}_{fov}", "index": index, "img_emb_id": img_emb_id} for img_emb_id, index in enumerate(img_df.index) for fov in train_fovs]

random.shuffle(img_list)

key2idx = {item["key"]: idx for idx, item in enumerate(img_list)}
train_count, val_count = int(len(img_list) * 0.8), int(len(img_list) * 0.1)
test_count = len(img_list) - train_count - val_count
train_img_list = img_list[:train_count]
train_fov_dict = dict()
for item in train_img_list:
    filename = item["key"]
    img_idx = int(filename.split("_")[0])
    fov = int(filename.split("_")[1])
    if img_idx not in train_fov_dict:
        train_fov_dict[img_idx] = []
    train_fov_dict[img_idx].append(fov)

val_img_list = img_list[train_count: train_count+val_count]
test_img_list = img_list[train_count+val_count:]

class PretrainDataset(Dataset):
    def __init__(self, img_list, mode):
        super().__init__()
        self.img_df = img_df
        self.img_txn = img_txn
        self.img_list = img_list
        self.mode = mode
        self.fov_padding_idx = len(fov2idx)
        self.fov_padding = [[0] * 1536]
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        data_dict = self.img_list[index]
        img_key = data_dict["key"]
        img_index = data_dict["index"]
        img_emb_idx = torch.LongTensor([data_dict["img_emb_id"]])
        fov_idx = torch.LongTensor([fov2idx[int(img_key.split("_")[1])]])

        img_data = self.img_df.loc[img_index]
        geo = torch.Tensor(img_data[["lat", "lng"]].tolist())
        img_geocode = img_data["geohash"][:10]
        img_geocode_tensor = torch.LongTensor([hash2index[hash_] for hash_ in start_token + img_geocode + end_token])
        img_vec, content_add_vec, creative_query, geo_specific_query, object_query, activity_query, atmosphere_query = pickle.loads(self.img_txn.get(img_key.encode()))
        
        img_vec = torch.Tensor(img_vec)
        content_add_vec = torch.Tensor(content_add_vec)

        query_vec = torch.Tensor([creative_query, geo_specific_query, object_query, activity_query, atmosphere_query])
        index = torch.LongTensor([index])
        
        img_id = torch.LongTensor([key2idx[img_key]])
        
        if self.mode == "train":
            fov_pos_list = list(set([f"{img_index}_{fov}" for fov in train_fov_dict[img_index]]) - set([img_key]))

            fov_id = [key2idx[item] for item in fov_pos_list]
            padding_len = 3 - len(fov_id)
            fov_id = torch.LongTensor(fov_id + [-1] * padding_len)

            fov_idx_list = [fov2idx[int(fov_pos.split("_")[1])] for fov_pos in fov_pos_list]
            fov_idx_list = fov_idx_list + [self.fov_padding_idx] * (3 - len(fov_idx_list))
            fov_idx_list = torch.LongTensor(fov_idx_list)

            img_vec_list, content_vec_list = [], []
            if fov_pos_list:
                for fov_pos in fov_pos_list:
                    fov_img_vec, fov_content_add_vec, _, _, _, _, _ = pickle.loads(self.img_txn.get(fov_pos.encode()))
                    img_vec_list.append(fov_img_vec)
                    content_vec_list.append(fov_content_add_vec)

            fov_padding_mask = [1] * len(img_vec_list) + [0] * padding_len
            img_vec_list = img_vec_list + self.fov_padding * padding_len
            content_vec_list = content_vec_list + self.fov_padding * padding_len
            
            fov_img_pos = torch.Tensor(img_vec_list)
            fov_content_pos = torch.Tensor(content_vec_list)
            fov_padding_mask = torch.LongTensor(fov_padding_mask)

            return {"img_vec": img_vec, "content_vec": content_add_vec, "query_vec": query_vec, "geocode": img_geocode_tensor, "index": index,
                    "fov_img_pos": fov_img_pos, "fov_content_pos": fov_content_pos, "fov_padding_mask": fov_padding_mask, "geo": geo, 
                    "img_emb_idx": img_emb_idx, "fov_idx": fov_idx, "fov_idx_list": fov_idx_list, "img_id": img_id, "fov_id": fov_id}
        else:
            return {"img_vec": img_vec, "content_vec": content_add_vec, "query_vec": query_vec, "geocode": img_geocode_tensor, 
                    "index": index, "geo": geo, "img_emb_idx": img_emb_idx, "fov_idx": fov_idx, "img_id": img_id}
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.geo_char_len = len(geo_code_char)
        self.emb_dim = 1536
        self.dim = EMB_DIM
        self.pos_encoding = PositionalEncoding(d_model=self.dim)
        self.geocode_embedding = nn.Embedding(num_embeddings=self.geo_char_len, embedding_dim=self.dim)
        geo_encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim, nhead=4, batch_first=True, dropout=0)
        self.geo_transformer_encoder = nn.TransformerEncoder(geo_encoder_layer, num_layers=4)

        img_moe = MoE(dim = self.emb_dim, num_experts = 4, gating_top_n = 2, threshold_train = 0.2, threshold_eval = 0.2, capacity_factor_train = 1.25, capacity_factor_eval = 2., balance_loss_coef = 1e-2, router_z_loss_coef = 1e-3)
        self.img_moe_block = SparseMoEBlock(img_moe, add_ff_before = True, add_ff_after = True)

        text_moe = MoE(dim = self.emb_dim, num_experts = 4, gating_top_n = 2, threshold_train = 0.2, threshold_eval = 0.2, capacity_factor_train = 1.25, capacity_factor_eval = 2., balance_loss_coef = 1e-2, router_z_loss_coef = 1e-3)
        self.text_moe_block = SparseMoEBlock(text_moe, add_ff_before = True, add_ff_after = True)

        self.img_head = nn.Sequential(
            nn.LayerNorm(self.emb_dim),
            nn.Linear(self.emb_dim, 1024),
            nn.GELU(),
            torch.nn.Linear(1024, self.dim)
        )
        self.content_head = nn.Sequential(
            nn.LayerNorm(self.emb_dim),
            nn.Linear(self.emb_dim, 1024),
            nn.GELU(),
            torch.nn.Linear(1024, self.dim)
        )
        self.query_head = nn.Sequential(
            nn.LayerNorm(self.emb_dim),
            nn.Linear(self.emb_dim, 1024),
            nn.GELU(),
            torch.nn.Linear(1024, self.dim)
        )

        self.geo_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim * 2),
            nn.GELU(),
            torch.nn.Linear(self.dim * 2, self.dim)
        )

        self.head_sem = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim * 2),
            nn.GELU(),
            torch.nn.Linear(self.dim * 2, self.dim)
        )

        self.head_geo = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim * 2),
            nn.GELU(),
            torch.nn.Linear(self.dim * 2, self.dim)
        )

        self.head_fov = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim * 2),
            nn.GELU(),
            # torch.nn.Linear(self.dim * 2, self.dim)
            torch.nn.Linear(self.dim * 2, 4)
        )

        merge_encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim, nhead=4, batch_first=True, dropout=0)
        self.merge_encoder = nn.TransformerEncoder(merge_encoder_layer, num_layers=4)

        self.img_embedding = nn.Embedding(num_embeddings=len(img_df), embedding_dim=self.emb_dim)
        self.fov_embedding = nn.Embedding(num_embeddings=len(fov2idx)+1, embedding_dim=self.emb_dim, padding_idx=len(fov2idx))

        self.modality_emb = nn.Embedding(3, self.dim)

    def forward(self, data, mode):
        img_emb = self.img_embedding(data["img_emb_idx"])
        fov_emb = self.fov_embedding(data["fov_idx"])
        
        img_out, img_total_aux_loss, _, _ = self.img_moe_block(data["img_vec"].unsqueeze(1) + img_emb + fov_emb)
        img_vec = self.img_head(img_out)

        content_out, content_total_aux_loss, _, _ = self.text_moe_block(data["content_vec"].unsqueeze(1))
        content_vec = self.content_head(content_out)

        geo_vec = self.geocode_embedding(data["geocode"])
        geo_vec = self.pos_encoding(geo_vec)
        geo_vec = self.geo_transformer_encoder(geo_vec).mean(dim=1, keepdim=True)
        geo_vec = self.geo_head(geo_vec)

        query_out, query_total_aux_loss, _, _ = self.text_moe_block(data["query_vec"])
        query_vec = self.query_head(query_out)
        
        img_vec = img_vec + self.modality_emb.weight[0].view(1, 1, -1)
        content_vec = content_vec + self.modality_emb.weight[1].view(1,1,-1)
        geo_vec = geo_vec + self.modality_emb.weight[2].view(1,1,-1)
        img_geo_content_vec = torch.concat([img_vec, content_vec, geo_vec], dim=1)
        merged = self.merge_encoder(img_geo_content_vec).mean(dim=1, keepdim=True)

        z_sem = self.head_sem(merged)
        z_geo = self.head_geo(merged)
        z_fov = self.head_fov(merged)

        if mode == "train":
            fov_fov_emb = self.fov_embedding(data["fov_idx_list"])

            fov_img_pos = data["fov_img_pos"] + fov_fov_emb + img_emb.expand(-1, 3, -1)
            fov_img_out, fov_img_total_aux_loss, _, _ = self.img_moe_block(fov_img_pos.view(-1, 1, fov_img_pos.size(-1)))
            fov_img_vec = self.img_head(fov_img_out)

            fov_content_pos = data["fov_content_pos"]
            fov_content_out, fov_content_total_aux_loss, _, _ = self.text_moe_block(fov_content_pos.view(-1, 1, fov_content_pos.size(-1)))
            fov_content_vec = self.content_head(fov_content_out)

            fov_geo_vec = geo_vec.expand(-1, 3, -1).reshape(-1, 1, geo_vec.size(-1))

            fov_img_vec = fov_img_vec + self.modality_emb.weight[0].view(1, 1, -1)
            fov_content_vec = fov_content_vec + self.modality_emb.weight[1].view(1, 1, -1)
            fov_geo_vec = fov_geo_vec + self.modality_emb.weight[2].view(1, 1, -1)

            fov_img_geo_content_vec = torch.concat([fov_img_vec, fov_content_vec, fov_geo_vec], dim=1)

            fov_merged = self.merge_encoder(fov_img_geo_content_vec).mean(dim=1, keepdim=True)

            fov_z_sem = self.head_sem(fov_merged).view(-1, 3, self.dim)
            fov_z_fov = self.head_fov(fov_merged).view(-1, 3, 4)
            
            aux_loss = (img_total_aux_loss + content_total_aux_loss + query_total_aux_loss + fov_img_total_aux_loss + fov_content_total_aux_loss) / 5

            return {"z_sem": z_sem, "z_geo": z_geo, "z_fov": z_fov, "query_vec": query_vec, "aux_loss": aux_loss, 
                    "fov_z_sem": fov_z_sem, "fov_z_fov": fov_z_fov}
        else:
            return {"z_sem": z_sem, "z_geo": z_geo, "z_fov": z_fov, "query_vec": query_vec}
        

class OurLossFunc(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.sigma = torch.nn.Parameter(torch.ones(3, device=device))
        self.fov_loss = nn.CrossEntropyLoss(ignore_index=len(fov2idx))

    def forward(self, outputs, geo_score, fov_mask, geo_mask, fov_idx, fov_idx_list):
        z_sem = outputs["z_sem"].squeeze(1)
        z_geo = outputs["z_geo"].squeeze(1)
        z_fov = outputs["z_fov"].squeeze(1)

        query_vec = outputs["query_vec"]

        fov_z_sem = outputs["fov_z_sem"]
        fov_z_fov = outputs["fov_z_fov"]

        anchor_loss = self.constraint_loss(z_sem, query_vec, neg_image_embeddings_global=fov_z_sem[fov_mask])
        geo_score = torch.tensor(geo_score).to(z_sem.device)[geo_mask][:, geo_mask]
        geo_loss = self.constraint_loss(z_geo[geo_mask], z_geo[geo_mask].unsqueeze(1), geo_score=geo_score)

        fov_inputs = torch.concat([z_fov.unsqueeze(1), fov_z_fov], dim=1).view(-1, 4)
        fov_targets = torch.concat([fov_idx, fov_idx_list], dim=-1).view(-1)
        fov_loss = self.fov_loss(fov_inputs, fov_targets)
        
        weighted_loss = 0.5 * torch.stack([anchor_loss, geo_loss, fov_loss]) / self.sigma ** 2
        weighted_loss = weighted_loss.sum() + torch.log(self.sigma.prod())

        return weighted_loss + outputs["aux_loss"]
    
    def constraint_loss(self, image_embeddings, text_embeddings, temperature: float = 0.03, geo_score: torch.Tensor = None, mask: torch.Tensor = None, neg_text_embeddings_global: torch.Tensor = None, neg_image_embeddings_global: torch.Tensor = None):
        device = image_embeddings.device
        B, P, D = text_embeddings.shape
        assert image_embeddings.shape[-1] == D

        if mask is None:
            mask = torch.ones((B, P), device=device, dtype=torch.bool)

        num_valid_pairs = torch.sum(mask)
        if num_valid_pairs == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        img_q = F.normalize(image_embeddings, dim=-1)      # (B, D)
        txt = F.normalize(text_embeddings, dim=-1)   # (B, P, D)

        txt_keys = txt.reshape(-1, D)

        if neg_text_embeddings_global is not None:
            neg_txt = F.normalize(neg_text_embeddings_global, dim=-1)
            all_txt_keys = torch.cat([txt_keys, neg_txt], dim=0)  # ((B*P)+Nt, D)
        else:
            all_txt_keys = txt_keys

        if geo_score is None:
            logits_i2t = (img_q @ all_txt_keys.t()) * (1.0 / temperature)   # (B, B*P)

            rows = torch.arange(B, device=device)              # (B,)

            col_idx = rows[:, None] * P + torch.arange(P, device=device)[None, :]  # (B, P)

            log_prob_i2t = logits_i2t.log_softmax(dim=1)               # (B, B*P)
            pos_log_prob  = log_prob_i2t.gather(1, col_idx)            # (B, P)
            loss_i2t = -torch.sum(pos_log_prob * mask) / num_valid_pairs

            flat_mask = mask.view(-1)  # (B*P,)
            valid_txt_q = txt.reshape(B * P, D)[flat_mask]  # (num_valid_pairs, D)

            if neg_image_embeddings_global is not None:
                neg_img = F.normalize(neg_image_embeddings_global, dim=-1)
                all_img_keys = torch.cat([img_q, neg_img], dim=0)               # (B+Ni, D)
            else:
                all_img_keys = img_q

            logits_t2i = (valid_txt_q @ all_img_keys.t()) * (1.0 / temperature)
            full_labels_t2i = torch.arange(B, device=device).repeat_interleave(P) # (B*P,)
            valid_labels_t2i = full_labels_t2i[flat_mask] # (num_valid_pairs,)

            loss_t2i = F.cross_entropy(logits_t2i, valid_labels_t2i)
        else:
            logits_i2t = (img_q @ all_txt_keys.t()) * (1.0 / temperature)
            soft_targets_i2t = F.normalize(geo_score, p=1, dim=1)
            loss_i2t = F.kl_div(F.log_softmax(logits_i2t, dim=1), soft_targets_i2t, reduction='batchmean')
            
            flat_mask = mask.view(-1)  # (B*P,)
            valid_txt_q = txt.reshape(B * P, D)[flat_mask]  # (num_valid_pairs, D)
            logits_t2i = (valid_txt_q @ img_q.t()) * (1.0 / temperature)
            full_labels_t2i = torch.arange(B, device=device).repeat_interleave(P) # (B*P,)
            valid_labels_t2i = full_labels_t2i[flat_mask] # (num_valid_pairs,)
            soft_targets_t2i = F.normalize(geo_score, p=1, dim=1)
            loss_t2i = F.kl_div(F.log_softmax(logits_t2i, dim=1), soft_targets_t2i, reduction='batchmean')
            
        return 0.5 * (loss_i2t + loss_t2i)

class Trainner:
    def __init__(self):
        self.device = torch.device(f"cuda:{cuda_idx}")
        self.model = Model().to(self.device)

        all_dataset = PretrainDataset(img_list, mode="faiss")
        train_dataset = PretrainDataset(train_img_list, mode="train")
        val_dataset = PretrainDataset(val_img_list, mode="eval")
        test_dataset = PretrainDataset(test_img_list, mode="test")

        self.all_loader = DataLoader(all_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, shuffle=False, persistent_workers=True, prefetch_factor=4)
        self.train_loader = DataLoader(train_dataset, batch_size=TBATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, shuffle=True, persistent_workers=True, prefetch_factor=4)
        self.val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, shuffle=False, persistent_workers=True, prefetch_factor=4)
        self.test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, shuffle=False, persistent_workers=True, prefetch_factor=4)

        self.our_loss = OurLossFunc(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.use_amp = True
        
        
        self.scheduler1 = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
        self.scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=1e-6)
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, schedulers=[self.scheduler1, self.scheduler2], milestones=[WARMUP_EPOCHS])
        self.scaler = torch.amp.GradScaler()

        self.best_checkpoint = None

    def train(self, epoch):
        self.model.train()
        self.iteration(epoch, self.train_loader, mode="train")
        self.scheduler.step()
        
    
    def eval(self, epoch):
        self.model.eval()
        acc_dict = self.iteration(epoch, self.val_loader, mode="eval")
        return acc_dict
    
    def test(self, epoch):
        self.model.eval()
        acc_dict = self.iteration(epoch, self.test_loader, mode="test")
        return acc_dict

    def trans_data(self, data):
        for key, value in data.items():
            data[key] = value.to(self.device)
        return data

    def save_checkpoint(self):
        torch.save(self.best_checkpoint, os.path.join(model_save_path, f"latest.pth"))
    
    def build_faiss(self):
        img_index = None
        for data in tqdm(self.all_loader, desc="faiss load"):
            data = self.trans_data(data)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                with torch.no_grad():
                    outputs = self.model(data, mode="test")
            
            img_vec = F.normalize(outputs["z_sem"].squeeze(1), dim=-1).cpu().detach().numpy()
            index_list = data["index"].squeeze(-1).cpu().numpy()
            
            if img_index is None:
                img_index = faiss.IndexFlatIP(img_vec.shape[-1])
                img_index = faiss.IndexIDMap(img_index)
            img_index.add_with_ids(img_vec, index_list)
        resource = faiss.StandardGpuResources()
        self.gpu_index = faiss.index_cpu_to_gpu(resource, cuda_idx, img_index)

    def iteration(self, epoch, data_loader, mode):
        epoch_train_loss_list = []
        correct_dict = {K: 0 for K in K_list}
        query_correct_dict = {i: 0 for i in range(5)}
        total_count, query_total_count = 0, 0
        distance_result = []
        mrr_list = []

        for index, data in enumerate(tqdm(data_loader, desc=f"{mode.upper()} Epoch {epoch}")):
            data = self.trans_data(data)
            if mode == "train":
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                    outputs = self.model(data, mode)
                    
                geo_list = data["geo"].tolist()
                geo_distance = haversine_vector(geo_list, geo_list, Unit.KILOMETERS, comb=True)
                geo_score = 1 - np.tanh(np.arcsinh(geo_distance))
                
                fov_mask = data["fov_padding_mask"].bool()
                unique_values, inverse_indices, counts = torch.unique(data["img_emb_idx"].view(-1), return_inverse=True, return_counts=True)
                element_counts = counts[inverse_indices]
                geo_mask = (element_counts == 1)

                fov_idx = data["fov_idx"]
                fov_idx_list = data["fov_idx_list"]
                loss = self.our_loss(outputs, geo_score, fov_mask, geo_mask, fov_idx, fov_idx_list)
                epoch_train_loss_list.append(loss.item())
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            
            elif mode == "test" or mode == "eval":
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                    with torch.no_grad():
                        outputs = self.model(data, mode)

                test_index_list = data["index"].squeeze().tolist()
                if mode == "test":
                    img_df_index_list = [key2idx[test_img_list[index]["key"]] for index in test_index_list]
                elif mode == "eval":
                    img_df_index_list = [key2idx[val_img_list[index]["key"]] for index in test_index_list]
                ground_truth = [item for item in img_df_index_list for _ in range(5)]

                query_vec = F.normalize(outputs["query_vec"], dim=-1)
                query_vec_numpy = query_vec.view(-1, query_vec.size(-1)).detach().cpu().numpy()

                search_distance, search_indices = self.gpu_index.search(query_vec_numpy, max(K_list))
                
                search_loc = [img_df.loc[img_list[item]["index"]][["lat", "lng"]].tolist() for item in search_indices[:, 0]]
                target_loc = data["geo"].unsqueeze(1).expand(-1, 5, -1).reshape(-1, 2).tolist()
                distance_list = haversine_vector(search_loc, target_loc, Unit.KILOMETERS).tolist()
                distance_result.extend(distance_list)

                search_result = [[key2idx[img_list[item]["key"]] for item in row] for row in search_indices]
                ground_truth = torch.LongTensor(ground_truth).unsqueeze(-1)
                match_index = np.where(ground_truth.numpy() == np.array(search_result))[1]
                mrr = [0] * len(ground_truth)
                if len(match_index) != 0:
                    mrr_ = (1 / (match_index + 1)).tolist()
                    mrr = mrr_ + [0] * (len(ground_truth) - len(mrr_))
                mrr_list.extend(mrr)

                for i in range(5):
                    query_correct = (ground_truth[i::5, :] == torch.LongTensor(search_result)[i::5, :min(K_list)]).any(dim=-1).sum()
                    query_correct_dict[i] += query_correct.item()
                for K in K_list:
                    k_correct = (ground_truth == torch.LongTensor(search_result)[:, :K]).any(dim=-1).sum()
                    correct_dict[K] += k_correct.item()

                total_count += len(ground_truth)
                query_total_count += (len(ground_truth) // 5)
                
        if mode=="train" and epoch_train_loss_list:
            logging.info(f"epoch: {epoch}, mode: {mode}, epoch_loss: {np.mean(epoch_train_loss_list)}")
        elif mode == "test" or mode == "eval":
            final_acc_dict = {K: val / total_count * 100 for K, val in correct_dict.items()}
            logging.info(f"epoch: {epoch}, mode: {mode}, total: {total_count}, acc: {final_acc_dict}, mrr: {np.mean(mrr_list)}")

            query_final_acc_dict = {i: val / query_total_count * 100 for i, val in query_correct_dict.items()}
            logging.info(f"epoch: {epoch}, mode: {mode}, total: {total_count}, query_acc: {query_final_acc_dict}")

            acc1 = (np.array(distance_result) < 1).sum() / total_count * 100
            acc5 = (np.array(distance_result) < 5).sum() / total_count * 100
            logging.info(f"epoch: {epoch}, mode: {mode}, total: {total_count}, acc1: {acc1}, acc5: {acc5}")

            return final_acc_dict
        
    def start(self):
        logging.info(f"\n\n\n{'-' * 100}")
        best_eval_acc = 0
        stop_count = 0
        for epoch in range(EPOCHS):
            if stop_count > 5: break
            self.train(epoch)
            # with torch.no_grad():
            self.build_faiss()
            acc_dict = self.eval(epoch)
            if acc_dict[10] > best_eval_acc:
                best_eval_acc = acc_dict[10]
                self.test(epoch)
                stop_count = 0
                # self.save_checkpoint(epoch)
                best_checkpoint = {
                    "model": {k: v.cpu() for k, v in self.model.state_dict().items()},
                }
                self.best_checkpoint = best_checkpoint
            else:
                stop_count += 1
        self.save_checkpoint()


if __name__ == '__main__':
    model_save_path = os.path.join(model_save_base, f"v{v}")
    
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    trainner = Trainner()
    trainner.start()