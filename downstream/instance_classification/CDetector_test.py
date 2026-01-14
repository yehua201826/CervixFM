import argparse
import os

import torch

from downstream.instance_classification.dataset.instance_dataset import Instance_Dataset
from downstream.load_model import ViT_classifier, ViT_classifier_DINOV2, ResNet50_classifier, ViT_classifier_UNI, ViT_classifier_Phikon, \
    ViT_classifier_CTranspath,ViT_classifier_HIPT
from downstream.utils.test import test_nofreeze, test_freeze


def init_args():
    parser = argparse.ArgumentParser(description="CDetector_test")

    parser.add_argument("--data_path", type=str, default="../../data/instance_classification_dataset/CDetector", help="CDetector dataset path")
    parser.add_argument("--feature_path", type=str, default="../../data/instance_classification_feature/CDetector_test", help="save freeze backbone feature")
    parser.add_argument("--save_path", type=str, default="../../data/instance_classification_output/CDetector", help="save model parameters, test results and logs")

    parser.add_argument("--model", type=str, default="ViT_classifier", help="model name")
    # parser.add_argument("--model_path", type=str, default="../../data/instance_classification_output/CDetector/checkpoint100/epoch9_fold4_model.pth", help="")
    parser.add_argument("--model_path", type=str, default="../../data/instance_classification_output/CDetector/freeze/checkpoint100/epoch9_fold4_model.pth", help="")
    parser.add_argument("--num_class", type=int, default=11, help="class num")
    parser.add_argument("--ViT_freeze", action="store_true", help="freeze vit model parameters")

    parser.add_argument("--label_efficiency", type=float, default=1., help='Label efficiency rate')

    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--device", type=int, default=6, help="device index")

    args = parser.parse_args()
    return args


if __name__=="__main__":
    args = init_args()
    
    args.save_path = os.path.join(args.save_path, args.model)
    if args.ViT_freeze:
        args.save_path = os.path.join(args.save_path, 'freeze')
    os.makedirs(args.save_path, exist_ok=True)

    test_dataset = Instance_Dataset(data_path=args.data_path, stage='test', seed=args.seed)

    model = eval(args.model)(num_class=args.num_class, ViT_freeze=args.ViT_freeze)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu')['model'])
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
    model.to(device)

    if args.ViT_freeze:
        feature_path = os.path.join(args.feature_path, args.model + '.pth')
        test_freeze(model=model, dataset=test_dataset, feature_path=feature_path, save_dir=args.save_path, num_class=args.num_class, efficiency=args.label_efficiency, device=device)
    else:
        test_nofreeze(model=model, dataset=test_dataset, save_dir=args.save_path, num_class=args.num_class, efficiency=args.label_efficiency, device=device)
