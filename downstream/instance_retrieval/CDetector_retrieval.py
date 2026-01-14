import argparse
import os

from downstream.instance_retrieval.dataset.retrieval_dataset import Retrieval_Dataset

from downstream.utils.retrieval import retrieval


def init_args():
    parser = argparse.ArgumentParser(description="CDetector_retrieval")

    parser.add_argument("--model", type=str, default="ViT_classifier", help="model name")

    parser.add_argument("--data_path", type=str, default="../../data/instance_classification_dataset/CDetector", help="CDetector dataset path")
    parser.add_argument("--key_feature_path", type=str, default="../../data/instance_classification_feature/CDetector", help="save freeze backbone feature")
    parser.add_argument("--query_feature_path", type=str, default="../../data/instance_classification_feature/CDetector_test", help="save freeze backbone feature")
    parser.add_argument("--save_path", type=str, default="../../data/instance_retrieval_output/CDetector", help="save model parameters, test results and logs")

    parser.add_argument("--device", type=int, default=3, help="device index")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = init_args()

    args.key_feature_path = os.path.join(args.key_feature_path, args.model+'.pth')
    args.query_feature_path = os.path.join(args.query_feature_path, args.model+'.pth')

    args.save_path = os.path.join(args.save_path, args.model)
    os.makedirs(args.save_path, exist_ok=True)

    key_dataset = Retrieval_Dataset(data_path=args.data_path, stage='train')
    query_dataset = Retrieval_Dataset(data_path=args.data_path, stage='test')

    retrieval(key_feature_path=args.key_feature_path, query_feature_path=args.query_feature_path, key_dataset=key_dataset, query_dataset=query_dataset,
              device=args.device, save_dir=args.save_path)