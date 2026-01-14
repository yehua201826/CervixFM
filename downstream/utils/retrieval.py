import torch
from torch.utils.data import DataLoader
from torch.nn.functional import normalize

import numpy as np
import faiss

from downstream.utils.metrics import caculate_retrieval_acc, save_result

def retrieval(key_feature_path, query_feature_path, key_dataset, query_dataset, device, save_dir, label_level=None):

    key_features = torch.load(key_feature_path)
    query_features = torch.load(query_feature_path)

    key_dataloader = DataLoader(key_dataset, batch_size=1, num_workers=4, shuffle=False, drop_last=False)
    query_dataloader = DataLoader(query_dataset, batch_size=1, num_workers=4, shuffle=False, drop_last=False)

    key_imgs = []
    key_embeddings = []
    key_labels = []
    for img_name, _, img_label in key_dataloader:
        img_name = img_name[0]
        img_label = img_label[label_level].item()
        key_imgs.append(img_name)
        key_embeddings.append(key_features[img_name])
        key_labels.append(img_label)

    query_imgs = []
    query_embeddings = []
    query_labels = []
    for img_name, _, img_label in query_dataloader:
        img_name = img_name[0]
        img_label = img_label[label_level].item()
        query_imgs.append(img_name)
        query_embeddings.append(query_features[img_name])
        query_labels.append(img_label)

    key_embeddings = torch.stack(key_embeddings)
    query_embeddings = torch.stack(query_embeddings)

    key_embeddings_mean = key_embeddings.mean(dim=0, keepdim=True)

    key_embeddings = key_embeddings - key_embeddings_mean
    query_embeddings = query_embeddings - key_embeddings_mean

    key_embeddings = normalize(key_embeddings, dim=-1, p=2)
    query_embeddings = normalize(query_embeddings, dim=-1, p=2)

    key_embeddings_np = key_embeddings.numpy().astype(np.float32)

    faiss_index = faiss.IndexFlatL2(key_embeddings_np.shape[1])
    faiss_index.add(key_embeddings_np)

    res = faiss.StandardGpuResources()
    gpu_faiss_index = faiss.index_cpu_to_gpu(res, device, faiss_index)

    img_result = dict()
    acc_result = dict()
    for i, query_embedding in enumerate(query_embeddings):
        query_embedding = query_embedding.numpy().reshape(1, -1)

        distances, indices = gpu_faiss_index.search(query_embedding, 10)

        img_result[query_imgs[i]] = [key_imgs[k] for k in indices[0]]

        indices_labels = [key_labels[k] for k in indices[0]]
        query_label = query_labels[i]
        acc_result[query_imgs[i]] = caculate_retrieval_acc(indices_labels, query_label)

    acc = np.array(list(acc_result.values()))
    average_acc = acc.sum(axis=0) / acc.shape[0]
    save_result(img_result, acc_result, average_acc, save_dir)

