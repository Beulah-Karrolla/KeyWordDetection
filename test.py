import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import numpy as np
from model import SpeechClassifierModel, SpeechClassifierModelTransformer
from data import SpeechCommandsDataset, collate_fn

def main(args):
    local_rank = args.device
    #torch.cuda.set_device(local_rank)
    device = torch.device('cpu')
    #device = torch.device('cuda:{:d}'.format(local_rank))

    # Load the trained model from the saved checkpoint
    checkpoint_path = args.checkpoint_path
    checkpoint = torch.load(checkpoint_path)
    model_state_dict = checkpoint["model_state_dict"]
    #import ipdb;ipdb.set_trace()
    model_params = {'num_classes': checkpoint['model_params']['num_classes'],
                    'feature_size': checkpoint['model_params']['feature_size'], 
                    'hidden_size': checkpoint['model_params']['hidden_size'], 
                    'num_layers': checkpoint['model_params']['num_layers'], 
                    'dropout': checkpoint['model_params']['dropout'], 
                    'bidirectional': checkpoint['model_params']['bidirectional'], 
                    'device': device}  
    model = SpeechClassifierModel(**model_params)
    model = SpeechClassifierModelTransformer(**model_params)
    #model = SpeechClassifierModel()
    model.load_state_dict(model_state_dict)
    model.eval()

    # Define the test dataset and data loader
    test_dataset = SpeechCommandsDataset(args.dataset_path, 'testing')
    test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=collate_fn)

    # Iterate over the test data and pass it through the model to get predictions
    correct = 0
    total = 0
    count = 0
    mistakes = []
    loss_list = []
    loss_fn = nn.BCEWithLogitsLoss()
    f = open(args.results_path ,'a')
    with torch.no_grad():
        for inputs, labels, pholders in test_dataloader:
            input, label, pholder = inputs.to(device), labels.to(device), pholders
            output = model(input)
            #import ipdb;ipdb.set_trace()
            label_float = label.float()
            loss = loss_fn(torch.flatten(output), label_float)
            loss_list.append(loss.item())
            best = np.where(output < 0.5, 0, 1)
            predicted = best[0][0]
            label = label.item()
            #_, predicted = torch.max(output.data, 1)
            total += 1
            correct += (predicted == label)
            #import ipdb;ipdb.set_trace()    
            mistakes.append([pholder, predicted, label]) if predicted != label else None
            print(count)
            count +=1

    
    #import ipdb;ipdb.set_trace()
    print("Total number of mistakes: {}".format(len(mistakes)))
    print("Total number of test examples: {}".format(total))
    print("Total number of correct predictions: {}".format(correct))
    print(mistakes)
    print("Total number of mistakes: {}".format(len(mistakes)))
    print("Total number of test examples: {}".format(total))
    print("Total number of correct predictions: {}".format(correct))
    f.write(str(mistakes))
    f.close()
            

    # Calculate the accuracy of the model on the test data
    accuracy = correct / total

    # Print the accuracy
    print("Accuracy: {:.2f}%".format(accuracy * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Speech Commands Testing for Wake Word Detection')
    parser.add_argument('--dataset_path', default=None, type=str, help='Path to dataset')
    parser.add_argument('--device', default=None, type=int, help='Cuda device')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to the best checkpoint')
    parser.add_argument('--results_path', type=str, default=None, help='Path to the save the results')
    args = parser.parse_args()

    main(args)
