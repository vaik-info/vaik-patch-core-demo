import os.path

import torch
import argparse
import multiprocessing
import tqdm
from datetime import datetime
import pytz

import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from timm.scheduler import StepLRScheduler
from torch.utils.tensorboard import SummaryWriter
from model.pretrain_classification import WideResnet502Model
from data.classification_dataset import ClassificationDataset

import random
import numpy as np
seed = 777
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)

def pretrain(device, epochs, batch_size, step_size, test_step_size, train_dir_path, test_dir_path, output_dir_path, pretrain_model_path):
    train_transform = [
                        T.Resize(256),
                        T.RandomResizedCrop(224),
                        T.RandomHorizontalFlip(),
                        T.ToTensor()
    ]
    train_dataset = ClassificationDataset(train_dir_path, batch_size * step_size, T.Compose(train_transform))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=multiprocessing.cpu_count() // 4, worker_init_fn=seed_worker, generator=g)

    test_transform = [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor()
    ]
    test_dataset = ClassificationDataset(test_dir_path, batch_size * test_step_size, T.Compose(test_transform))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=multiprocessing.cpu_count() // 4, worker_init_fn=seed_worker, generator=g)

    # Extract feature
    if os.path.exists(pretrain_model_path):
        checkpoint = torch.load(pretrain_model_path)
        num_classes = checkpoint['num_classes']
        pretrain_model = WideResnet502Model(num_classes)
        pretrain_model.load_state_dict(checkpoint['state_dict'])
        print(f'Load: at {pretrain_model_path}')
    else:
        pretrain_model = WideResnet502Model(num_classes=len(train_dataset.classes))
        print(f'Load: at None')
    pretrain_model.to(device)

    # prepare loss
    criterion = nn.CrossEntropyLoss()

    # prepare optimizer
    optimizer = optim.SGD(pretrain_model.parameters(), lr=0.0005)
    scheduler = StepLRScheduler(optimizer,
                                decay_t=epochs//6,
                                decay_rate=0.5,
                                t_in_epochs=True)

    # prepare metrics
    train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=len(train_dataset.classes))
    val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=len(train_dataset.classes))

    # prepare logs
    save_model_dir_path = os.path.join(output_dir_path,
                                       f'{datetime.now(pytz.timezone("Asia/Tokyo")).strftime("%Y-%m-%d-%H-%M-%S")}')
    os.makedirs(save_model_dir_path, exist_ok=True)
    summary_writer = SummaryWriter(os.path.join(save_model_dir_path, 'logs'))

    # train
    for epoch in tqdm.tqdm(range(epochs), desc='epoch'):
        pretrain_model.train()
        train_acc.reset(), val_acc.reset()
        train_epoch_loss = 0
        val_epoch_loss = 0

        for x, y in tqdm.tqdm(train_dataloader, desc='train'):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            y_pred = pretrain_model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            train_acc(torch.argmax(y_pred, -1).cpu(), y.cpu())
            train_epoch_loss += y_pred.shape[0] * loss.item()

        pretrain_model.eval()
        for x, y in tqdm.tqdm(test_dataloader, desc='test'):
            with torch.no_grad():
                x, y = x.to(device), y.to(device)
                y_pred = pretrain_model(x)
                loss = criterion(y_pred, y)
                val_acc(torch.argmax(y_pred, -1).cpu(), y.cpu())
                val_epoch_loss += y_pred.shape[0] * loss.item()

        train_epoch_loss = train_epoch_loss/(len(train_dataloader)*batch_size)
        val_epoch_loss = val_epoch_loss/(len(test_dataloader)*batch_size)
        print(f"Epoch{epoch+1}, Loss/train:{train_epoch_loss}, Loss/test:{val_epoch_loss},"
              f" Accuracy/train:{train_acc.compute()}, Accuracy/test:{val_acc.compute()}")
        torch.save({'state_dict': pretrain_model.state_dict(), 'num_classes': len(train_dataset.classes)}, os.path.join(save_model_dir_path, f'epoch-{epoch}_step-{step_size}_batch-{batch_size}_train_loss-{train_epoch_loss:.4f}_val_loss-{val_epoch_loss:.4f}_train_acc-{train_acc.compute():.4f}_val_acc-{val_acc.compute():.4f}.pth'))
        torch.save({'state_dict': pretrain_model.state_dict(), 'num_classes': len(train_dataset.classes)}, os.path.join(output_dir_path, 'latest.pth'))
        summary_writer.add_scalar("Loss/train", train_epoch_loss, epoch+1)
        summary_writer.add_scalar("Loss/test", val_epoch_loss, epoch+1)
        summary_writer.add_scalar("Accuracy/train", train_acc.compute(), epoch+1)
        summary_writer.add_scalar("Accuracy/test", val_acc.compute(), epoch+1)
        scheduler.step(epoch+1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pretrain')
    parser.add_argument('--device', type=str, default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--step_size', type=int, default=1000)
    parser.add_argument('--test_step_size', type=int, default=1000)
    parser.add_argument('--train_dir_path', type=str, default=os.path.join(os.path.dirname(__file__), 'sample_dataset/vaik-mnist-classification-dataset/train'))
    parser.add_argument('--test_dir_path', type=str, default=os.path.join(os.path.dirname(__file__), 'sample_dataset/vaik-mnist-classification-dataset/valid'))
    parser.add_argument('--output_dir_path', type=str, default='/tmp/pretrain_output')
    parser.add_argument('--pretrain_model_path', type=str, default='', help='Path to the pre-trained weights')
    args = parser.parse_args()

    args.train_dir_path = os.path.expanduser(args.train_dir_path)
    args.test_dir_path = os.path.expanduser(args.test_dir_path)
    args.output_dir_path = os.path.expanduser(args.output_dir_path)
    args.pretrain_model_path = os.path.expanduser(args.pretrain_model_path)

    pretrain(args.device, args.epochs, args.batch_size, args.step_size, args.test_step_size, args.train_dir_path, args.test_dir_path, args.output_dir_path, args.pretrain_model_path)
