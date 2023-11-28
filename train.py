import torch
import os
import argparse
import torch.nn as nn
from data import SpeechCommandsDataset, collate_fn
from model import SpeechClassifierModel, SpeechClassifierModelTransformer
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from tabulate import tabulate
from tqdm import tqdm
import numpy as np

def save_checkpoint(model, optimizer, scheduler,model_params, filename):
    state = {
        'model_params':model_params,
        'model_state_dict': model.state_dict(),
        'optimizer_sdict': optimizer.state_dict(),
        'scheduler_sdict': scheduler.state_dict(),
    }
    torch.save(state, filename)

def data_loader(args, data, **kwargs):
    train_loader = DataLoader(data, batch_size=args.batch_size, shuffle = 1,
                                    collate_fn=collate_fn, **kwargs)
    return train_loader


def train(args, model, device, train_loader, optimizer, loss_fn, epoch):
    loss_list = []
    pred_list = []
    label_list = []
    model.train()  # Set the model to training mode
    for idx, (data, target, pholders) in enumerate(tqdm(train_loader, desc="Training")):
        data, target = data.to(device), target.to(device)
        output = model(data)
        target = target.float()
        loss = loss_fn(torch.flatten(output), target)
        loss.backward()
        #loss_rec += loss.detach()
        optimizer.step()
        optimizer.zero_grad()
        loss_list.append(loss.item())
        pred = torch.sigmoid(output)
        #pred_list.extend(torch.flatten(torch.round(pred)).cpu().numpy())
        label_list.extend(target.cpu().numpy())
    return epoch, loss_list
    '''    log = f'| epoch = {epoch} | loss_asr = {np.mean(loss_list)} | lr = {self.scheduler.get_last_lr()} |'
        return 1, log, loss_list
        print("epoch: {}, Iter: {}/{}, loss: {}".format(epoch, idx, len(train_loader), loss.item()), end="\r")
    mean_loss = sum(loss_list) / len(loss_list)
    #round_pred = torch.round(torch.tensor(pred_list))
    #correct = round_pred.eq(torch.tensor(label_list).view_as(round_pred)).sum().item()
    #accuracy = correct / len(torch.tensor(label_list))
    #print('Average train loss:', mean_loss, "Average train Accuracy", accuracy)
    report = classification_report(torch.Tensor(label_list).numpy(), torch.Tensor(pred_list).numpy())
    print(report)
    return 1, report, loss_list'''

def main(args):
    local_rank = 0
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda:{:d}'.format(local_rank))

    train_dataset = SpeechCommandsDataset(args.data_path, args.model_type)
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if local_rank else {}
    train_loader = data_loader(args, train_dataset, **kwargs)
    model_params = {'num_classes': 1, 'feature_size': 40, 'hidden_size': args.hidden_size, 
                    'num_layers': 3, 'dropout': 0.2, 'bidirectional':True, 'device': device}  
    model = SpeechClassifierModel(**model_params)
    #model = SpeechClassifierModelTransformer(**model_params)
    if (args.load_pretrain_model):
        checkpoint = torch.load(args.load_pretrain_model)
        model.load_state_dict(checkpoint['model_state_dict'])
    model=model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #loss_fn = nn.BCELoss()
    loss_fn = nn.BCEWithLogitsLoss()
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5, verbose=True)
    path = os.path.join('results/', (str(args.model_type)+'_'+str(args.model_name))+".txt")
    f = open(path,'a')
    for epoch in range(1, args.epochs + 1):
        print("\nstarting training with learning rate", optimizer.param_groups[0]['lr'])
        epoch, loss_list = train(args, model, device, train_loader, optimizer, loss_fn, epoch)
        log = f"| epoch = {epoch} | loss_vad = {np.mean(loss_list)} | lr = {optimizer.param_groups[0]['lr']} |"
        scheduler.step(np.mean(loss_list))
        if epoch % 1 == 0:
            checkpoint_path = os.path.join(args.save_checkpoint_path, (str(args.model_type)+'_'+str(args.model_name)+'_'+str(epoch))+".pt")
            save_checkpoint(model, optimizer, scheduler, model_params, checkpoint_path)
        '''table = tabulate([['Best train accuracy', best_train_accuracy], 
                          ['Best train report', best_train_report],
                          ['Best epoch', best_epoch],
                          ['Current epoch', epoch],
                          ['Current train accuracy', train_accuracy],
                          ['Current train report', train_report],
                          ['Current train loss', sum(loss_list)/len(loss_list)],
                          ['Current learning rate', optimizer.param_groups[0]['lr']]
                          ],
                          headers=['Metric', 'Value']) 
        print(table + "\n")'''
        print(log)
        f.write(log + "\n")
        print("Finished training")
        print("Model saved at", checkpoint_path)
    f.close()

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Speech Commands Training for Wake Word Detection')
    parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--model_name',default="KeyWD", type=str,required=False, help='Name of the saved model')
    parser.add_argument('--data_path', default=None, type=str, help='Path to data')
    parser.add_argument('--num_classes', default=1, type=int, help='Number of classes')
    parser.add_argument('--save_checkpoint_path', type=str, default=None, help='Path to save the best checkpoint')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--num_workers', type=int, default=1, help='number of data loading workers')
    parser.add_argument('--hidden_size', type=int, default=128, help='lstm hidden size')
    parser.add_argument('--load_pretrain_model', type=str, default=None, required=False, help='path to load a pretrain model to continue training')
    parser.add_argument('--model_type', type=str, default=None, help='Type of data sent to the model')
    
    args = parser.parse_args()

    main(args)
