import os
import numpy as np

import torch
from torch.utils.data import DataLoader

from downstream.utils.metrics import caculate_metrics, save_metrics


def test_nofreeze(model, dataset, save_dir, num_class, efficiency, device, label_level=None):

    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, drop_last=True, pin_memory=True)

    model.eval()
    all_labels = []
    all_probs = []
    all_preds =[]

    model.eval()
    with torch.no_grad():
        for i, (img_names, imgs, labels) in enumerate(dataloader):
            imgs = imgs.to(device)
            if label_level is not None:
                labels = np.array(labels[label_level])
            else:
                labels = np.array(labels)

            output = model(imgs)
            probs = torch.softmax(output, dim=1).cpu().numpy()
            output_index = torch.argmax(output, dim=1).cpu().numpy()

            all_labels.extend(labels)
            all_probs.extend(probs)
            all_preds.extend(output_index)

    balanced_accuracy, top_accuracy, F1, auroc, roc, cm= caculate_metrics(np.array(all_labels), np.array(all_probs), np.array(all_preds), num_class)

    if efficiency * 100 >= 1:
        result_save_path = os.path.join(save_dir, f'result{int(efficiency * 100)}')
    else:
        result_save_path = os.path.join(save_dir, f'result{str(efficiency * 100)}')
    os.makedirs(result_save_path, exist_ok=True)

    save_metrics(result_save_path, balanced_accuracy, top_accuracy, F1, auroc, roc, cm)


def test_freeze(model, dataset, feature_path, save_dir, num_class, efficiency, device, label_level=None):

    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, drop_last=True, pin_memory=True)

    model.eval()
    all_labels = []
    all_probs = []
    all_preds =[]

    if os.path.exists(feature_path):
        print('loading freeze backbone feature ! ! !')
        features_dict = torch.load(feature_path)

        model.eval()
        with torch.no_grad():
            for i, (img_names, imgs, labels) in enumerate(dataloader):
                img_features = torch.stack([features_dict[img_name] for img_name in img_names])
                img_features = img_features.to(device)

                if label_level is not None:
                    labels = np.array(labels[label_level])
                else:
                    labels = np.array(labels)

                output = model.forward_classifier(img_features)
                probs = torch.softmax(output, dim=1).cpu().numpy()
                output_index = torch.argmax(output, dim=1).cpu().numpy()

                all_labels.extend(labels)
                all_probs.extend(probs)
                all_preds.extend(output_index)

    else:
        print('recording freeze backbone feature ! ! !')
        features_dict = dict()

        model.eval()
        with torch.no_grad():
            for i, (img_names, imgs, labels) in enumerate(dataloader):
                imgs = imgs.to(device)

                if label_level is not None:
                    labels = np.array(labels[label_level])
                else:
                    labels = np.array(labels)

                features = model.forward_backbone(imgs)
                output = model.forward_classifier(features)

                probs = torch.softmax(output, dim=1).cpu().numpy()
                output_index = torch.argmax(output, dim=1).cpu().numpy()

                all_labels.extend(labels)
                all_probs.extend(probs)
                all_preds.extend(output_index)

                features = features.cpu()
                for i, img_name in enumerate(img_names):
                    features_dict[img_name] = features[i]

        os.makedirs(os.path.dirname(feature_path), exist_ok=True)
        torch.save(features_dict, feature_path)

    balanced_accuracy, top_accuracy, F1, auroc, roc, cm= caculate_metrics(np.array(all_labels), np.array(all_probs), np.array(all_preds), num_class)

    if efficiency * 100 >= 1:
        result_save_path = os.path.join(save_dir, f'result{int(efficiency * 100)}')
    else:
        result_save_path = os.path.join(save_dir, f'result{str(efficiency * 100)}')
    os.makedirs(result_save_path, exist_ok=True)

    save_metrics(result_save_path, balanced_accuracy, top_accuracy, F1, auroc, roc, cm)