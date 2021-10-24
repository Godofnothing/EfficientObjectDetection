import os
import torch
import pickle
import argparse
import pandas as pd

from datasets.policy_dataset import PolicyDataset
from utils.metrics import DatasetMetricEvaluator
from utils.reward import DetectionReward
from utils.agent import get_model 
from trainer import TrainerSP

def get_metrics_csv(args, dataset='train', n_flops='400MF'):
    path = f'{args.results_dir}/mask_rcnn_regnetx-{n_flops}/{dataset}_metrics.csv'
    return pd.read_csv(path).iloc[:, 1:]

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, help='data dir')
    parser.add_argument('-r', '--results_dir', type=str, help='dir with metrics results')
    parser.add_argument('-o', '--output_dir', type=str, help='dir with the experiment output')
    parser.add_argument('--agent_model', type=str, default='resnet18', help='CNN model used for the agent')
    parser.add_argument('--num_epochs', type=int, default=5, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of agent training')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='optimizer lr')
    parser.add_argument('--flops_weight', type=float, default=1e-11, help='weight of flops in the reward')
    parser.add_argument('--alpha', type=float, default=0.85, help='probaility bounding factor')
    parser.add_argument('--opt', type=str, default='adam', help='optimizer type [Adam or SGD]')
    
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # agent initialization
    agent = get_model(1, _type='resnet18').to(device=device) 
    # load metric files
    train_metric_wk = get_metrics_csv(args, 'train', '400MF')
    train_metric_sg = get_metrics_csv(args, 'train', '4GF')
    val_metric_wk = get_metrics_csv(args, 'val', '400MF')
    val_metric_sg = get_metrics_csv(args, 'val', '4GF')
    # train dataset
    train_dataset = PolicyDataset(
        train_metric_wk,
        train_metric_sg,
        args.data_dir,
        coco_annotations_path = 'annotations/instances_train2017.json', 
        images_path = 'train2017'
    )
    # val dataset 
    val_dataset = PolicyDataset(
        val_metric_wk,
        val_metric_sg,
        args.data_dir,
        coco_annotations_path = 'annotations/instances_val2017.json', 
        images_path = 'val2017'
    )
    
    wk_detection_path = f'{args.results_dir}/mask_rcnn_regnetx-400MF/val_results.pkl'
    sg_detection_path = f'{args.results_dir}/mask_rcnn_regnetx-4GF/val_results.pkl'
    gt_detection_path = f'{args.results_dir}/val_gt.pkl'

    dme = DatasetMetricEvaluator(
        val_dataset, 
        wk_detection_path, 
        sg_detection_path, 
        gt_detection_path
    ) 
    # set the coefficient
    detection_reward = DetectionReward(4*(10**8), 4*(10**9), args.flops_weight)

    my_trainer = TrainerSP(
        agent, train_dataset, detection_reward, 
        val_dataset=val_dataset, val_dataset_metric_evaluator=dme, 
        device=device, alpha=args.alpha
    )

    if args.opt == 'adam':
        optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate)
    else:
        optimizer = torch.optim.SGD(agent.parameters(), lr=args.learning_rate, momentum=0.9)

    my_trainer.configure_optimizer(optimizer)

    ssm = my_trainer.train(args.num_epochs, batch_size=args.batch_size, validate=True, verbose=30)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/ssm.pkl","wb") as ssm_file:
        pickle.dump(ssm, ssm_file)
