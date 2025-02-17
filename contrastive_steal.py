import torch
import argparse
from surrogate_model import Surrogate_model
from utils import load_dataset
from Loss import ContrastiveLoss
from train_representation import train_representation,train_represnetation_linear
from train_posteriors import train_posterior
from test_target import test_for_target
from test_last import test_onehot
from train_onehot import train_onehot
from train_posteriors import train_posterior
from test_target import test_for_target
import numpy as np
from utils import load_target_model,load_dataset
import dataloader
from test_target import test_for_target
import torchvision
from Linear import linear
import os
from PIL import Image
import requests
import timm # A couple of these imports are unused, same as normal steal.

def main():
    torch.set_num_threads(1)   
    parser = argparse.ArgumentParser() # All of these parsers define how the surrogate model will be trained.
    parser.add_argument('--model_type',default='simclr',type=str)
    parser.add_argument('--pretrain',default='cifar10',type=str)
    parser.add_argument('--target_dataset',default='cifar10',type=str)
    parser.add_argument('--surrogate_dataset',default='cifar10',type=str)
    parser.add_argument('--augmentation',default=2,type=int) # No difference between stealing target, but these is an augmentation argument for modifying appearence of surrogate data.
    parser.add_argument('--surrogate_model',default='resnet18',type=str)
    parser.add_argument('--epoch',default= 100, type = int)


    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    catagory_num = 10
    surrogate_model = Surrogate_model(catagory_num,args.surrogate_model).to(device)
    target_encoder,target_linear = load_target_model(args.model_type,args.pretrain,args.target_dataset)
    train_dataset,test_dataset,linear_dataset = load_dataset(args.pretrain,args.target_dataset,args.surrogate_dataset,args.augmentation,args.split) # Prepares dataset for training and testing.
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True,
    )
    linear_loader = torch.utils.data.DataLoader(
        linear_dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True,
    )
    # This is where things start getting different. Instead of writing different if statements depending on what part of the target model you are stealing, everything has been summarized to one stealing method.
    
    criterion = ContrastiveLoss(32) # The key difference that steals target encoders using embeddings.
    optimizer = torch.optim.Adam(surrogate_model.encoder.parameters(), lr=3e-4)
    criterion2 = torch.nn.CrossEntropyLoss() # Criterion for loss between classifications of target and surrogate model.
    optimizer2 = torch.optim.Adam(surrogate_model.linear.parameters(), lr=3e-4)
    for i in range(args.epoch):
        train_representation(target_encoder,surrogate_model.encoder,train_loader,criterion,optimizer,device) # Trains the surrogate model's encoder using specified criterion and optimizer.
    for i in range(args.epoch):
        train_represnetation_linear(surrogate_model,target_encoder,linear_loader,criterion2,optimizer2,device) # Trains linear layer to predict data using the surrogate model's new encoder.
        agreement,accuracy = test_onehot(target_encoder,surrogate_model,test_loader) # Compares results between surrogate and target model.
    os.makedirs("new_surrogate_model/", exist_ok=True)
    # torch.save(surrogate_model,'new_surrogate_model/surrogate_model_'+args.model_type + '_'+args.steal+'_'+args.pretrain+'_'+args.target_dataset+'_'+args.surrogate_dataset+'_'+args.surrogate_model+'.pkl')
    # Less lines than the standard method of stealing a model, which leads to less computational effort.

if __name__ == "__main__":
    main()
    
