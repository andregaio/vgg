import torch
import torch.nn as nn
import torch.optim as optim
from models.base import VGG
import wandb
from dataset import load_dataset


WANDB_LOG = True
BATCH_SIZE = 224
EPOCHS = 74
CLASSES = 10
LEARNING_RATE = 10e-2
MOMENTUM = 0.9
WEIGHT_DECAY = 5 * 10e-4
NETWORK = "vgg_A"
DATASET = "cifar10"


if WANDB_LOG:
    run = wandb.init(
        project="vgg",            
        config={
            "learning_rate": LEARNING_RATE,
            "network": NETWORK,
            "dataset": DATASET,
            "epochs": EPOCHS,
            "loss" : 'sgd',
            "batch_size" : BATCH_SIZE,
            "momentum" : MOMENTUM,
        })


trainloader, valloader = load_dataset('cifar10', batch_size = BATCH_SIZE)



model = VGG(classes = CLASSES,  network = NETWORK)
model.cuda()


criterion = nn.CrossEntropyLoss()


optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay = WEIGHT_DECAY)


scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[18, 36, 54], gamma=0.1)


for epoch in range(EPOCHS):
    train_rolling_accuracy = train_rolling_loss = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, max_indices = torch.max(outputs, dim=1)
        num_matches = torch.sum(max_indices == labels)
        train_accuracy = (num_matches.item() / labels.numel()) * 100.0
        train_rolling_accuracy += train_accuracy
        train_rolling_loss += loss.item()
        print(f'Epoch: {epoch + 1} / {EPOCHS}, Iter: {(i + 1)} / {len(trainloader)} Loss: {loss.item():.3f} Accuracy: {train_accuracy:.2f}')
    train_average_accuracy = train_rolling_accuracy / len(trainloader)
    train_average_loss = train_rolling_loss / len(trainloader)
    print(f'Epoch: {epoch + 1} Avg. Loss: {train_average_loss:.2f} Avg. Accuracy: {train_average_accuracy:.2f}')



    with torch.no_grad():
        val_rolling_accuracy = val_rolling_loss = 0
        for i, data in enumerate(valloader, 0):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, max_indices = torch.max(outputs, dim=1)
            num_matches = torch.sum(max_indices == labels)
            val_accuracy = (num_matches.item() / labels.numel()) * 100.0
            val_rolling_accuracy += val_accuracy
            val_rolling_loss += loss.item()
            print(f'Epoch: {epoch + 1} / {EPOCHS}, Iter: {(i + 1)} / {len(valloader)}] Loss: {loss.item():.3f} Accuracy: {val_accuracy:.2f}')
        val_average_accuracy = val_rolling_accuracy / len(valloader)
        val_average_loss = val_rolling_loss / len(valloader)
        print(f'Epoch: {epoch + 1} Avg. Loss: {val_average_loss:.2f} Avg. Accuracy: {val_average_accuracy:.2f}')


    if WANDB_LOG:
        wandb.log({"train_average_accuracy": train_average_accuracy,
                    "val_average_accuracy": val_average_accuracy,
                    "train_average_loss": train_average_loss,
                    "val_average_loss": val_average_loss,
                })


    if not (epoch + 1) % 10:
        weights_filepath = f'weights/checkpoint_' + str(epoch + 1).zfill(5) + '.pt'
        torch.save(model.state_dict(), weights_filepath)


    scheduler.step()

print('Finished Training')