import torch
import torch.nn as nn
import torch.optim as optim
from models.base import VGG
import wandb
import argparse
from data import load_dataset, compute_accuracy, CLASSES


def train(args):

    if args.wandb:
        run = wandb.init(
            project='vgg',            
            config={
                'network': args.model,
                'dataset': args.dataset,
                'epochs': args.epochs,
                'batch' : args.batch,
                'loss' : 'sgd',
                'learning_rate': args.learning_rate,
                'momentum' : args.momentum,
                'weight_decay' : args.weight_decay,
            })

    trainloader, valloader = load_dataset(batch_size = args.batch)

    model = VGG(classes = CLASSES,  network = args.model)
    model.cuda()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay = args.weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[18, 36, 53], gamma=0.1)

    for epoch in range(args.epochs):
        train_rolling_accuracy = train_rolling_loss = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_accuracy = compute_accuracy(outputs, labels)
            train_rolling_accuracy += train_accuracy
            train_rolling_loss += loss.item()
            print(f'Epoch: {epoch + 1} / {args.epochs}, Iter: {(i + 1)} / {len(trainloader)} Loss: {loss.item():.3f} Accuracy: {train_accuracy:.2f}')
        train_average_accuracy = train_rolling_accuracy / len(trainloader)
        train_average_loss = train_rolling_loss / len(trainloader)
        print(f'Epoch: {epoch + 1} Avg. Loss: {train_average_loss:.2f} Avg. Accuracy: {train_average_accuracy:.2f}')

        with torch.no_grad():
            val_rolling_accuracy = val_rolling_loss = 0
            for i, data in enumerate(valloader, 0):
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_accuracy = compute_accuracy(outputs, labels)
                val_rolling_accuracy += val_accuracy
                val_rolling_loss += loss.item()
                print(f'Epoch: {epoch + 1} / {args.epochs}, Iter: {(i + 1)} / {len(valloader)} Loss: {loss.item():.3f} Accuracy: {val_accuracy:.2f}')
            val_average_accuracy = val_rolling_accuracy / len(valloader)
            val_average_loss = val_rolling_loss / len(valloader)
            print(f'Epoch: {epoch + 1} Avg. Loss: {val_average_loss:.2f} Avg. Accuracy: {val_average_accuracy:.2f}')

        if args.wandb:
            wandb.log({'train_average_accuracy': train_average_accuracy,
                        'val_average_accuracy': val_average_accuracy,
                        'train_average_loss': train_average_loss,
                        'val_average_loss': val_average_loss,
                    })

        if not (epoch + 1) % 10:
            weights_filepath = f'weights/checkpoint_{args.dataset}_{args.model}_' + str(epoch + 1).zfill(5) + '.pt'
            torch.save(model.state_dict(), weights_filepath)

        scheduler.step()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Argument parser")
    parser.add_argument("--model", type = str, default = 'vgg_A')
    parser.add_argument("--dataset", type = str, default = 'cifar10')
    parser.add_argument("--batch", type = int, default = 64)
    parser.add_argument("--epochs", type = int, default = 74)
    parser.add_argument("--learning_rate", type = float, default = 10e-4)
    parser.add_argument("--weight_decay", type = float, default = 5 * 10e-4)
    parser.add_argument("--momentum", type = str, default = 0.9)
    parser.add_argument("--wandb", action="store_true", default = False)
    args = parser.parse_args()

    train(args)