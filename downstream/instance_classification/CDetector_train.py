import argparse
import os

import torch
import torch.nn as nn

from downstream.instance_classification.dataset.instance_dataset import Instance_Dataset
from downstream.load_model import ViT_classifier, ViT_classifier_DINOV2, ResNet50_classifier, ViT_classifier_UNI, ViT_classifier_Phikon, \
    ViT_classifier_CTranspath,ViT_classifier_HIPT
from downstream.utils.train import train_freeze, train_nofreeze


def init_args():
    parser = argparse.ArgumentParser(description="CDetector_train")

    parser.add_argument("--epochs", type=int, default=1000, help="train epoch")
    parser.add_argument("--batch_size", type=int, default=64, help="train batch size")

    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate for classifier, if vit is learnable, set its learning rate = learning_rate*0.01")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="weight decay for vit and classifier")

    parser.add_argument("--model", type=str, default="ViT_classifier", help="model name")
    parser.add_argument("--data_path", type=str, default="../../data/instance_classification_dataset/CDetector", help="CDetector dataset path")
    parser.add_argument("--feature_path", type=str, default="../../data/instance_classification_feature/CDetector", help="save freeze backbone feature")
    parser.add_argument("--save_path", type=str, default="../../data/instance_classification_output/CDetector", help="save model parameters, test results and logs")

    parser.add_argument("--pretrained_path", type=str, default="../../data/DINOv2_checkpoints/2024-11-08_03-41-40/ddp_weight_iteration_652580.pth", help="pretrained vit model")
    parser.add_argument("--num_class", type=int, default=11, help="class num")
    parser.add_argument("--ViT_freeze", action="store_true", help="freeze vit model parameters")

    parser.add_argument("--label_efficiency", type=float, default=1., help='Label efficiency rate')

    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--device", type=int, default=0, help="device index")

    args = parser.parse_args()
    return args


if __name__=="__main__":
    args = init_args()
    
    args.save_path = os.path.join(args.save_path, args.model)
    if args.ViT_freeze:
        args.save_path = os.path.join(args.save_path, 'freeze')
    os.makedirs(args.save_path, exist_ok=True)

    train_dataset = Instance_Dataset(data_path=args.data_path,  efficiency=args.label_efficiency, stage='train', seed=args.seed)

    model = eval(args.model)(pretrained_path=args.pretrained_path, num_class=args.num_class, ViT_freeze=args.ViT_freeze)
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
    model.to(device)

    if args.ViT_freeze:
        print("only classifier train !")
        optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        print("vit and classifier both train !")
        optimizer = torch.optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': args.learning_rate*0.01, 'weight_decay': args.weight_decay*0.01},
            {'params': model.classifier.parameters(), 'lr': args.learning_rate, 'weight_decay': args.weight_decay}
        ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_dataset) // args.batch_size, eta_min=1e-6)

    criterion = nn.CrossEntropyLoss()

    if args.ViT_freeze:
        feature_path = os.path.join(args.feature_path, args.model + '.pth')
        train_freeze(model=model, dataset=train_dataset, optimizer=optimizer, scheduler=scheduler, criterion=criterion, feature_path=feature_path,
                     epochs=args.epochs, batch_size=args.batch_size, save_dir=args.save_path, efficiency=args.label_efficiency, device=device)
    else:
        train_nofreeze(model=model, dataset=train_dataset, optimizer=optimizer, scheduler=scheduler, criterion=criterion,
                       epochs=args.epochs, batch_size=args.batch_size, save_dir=args.save_path, efficiency=args.label_efficiency, device=device)
