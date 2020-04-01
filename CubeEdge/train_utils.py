import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import math

device = 'cuda:0'

def train(net, train_data, test_data, lr=1e-3, num_epochs=200, batch_size=32, weight_decay=1e-5, print_loss=False, num_cls=4):
    print(net)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1)

    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    all_train_loss = []
    all_eval_loss = []
    all_train_acc = []
    all_eval_acc = []

    tic = time.time()
    for epoch in range(1, num_epochs + 1):
        net.train()
        correct = 0
        total = 0
        running_loss = 0.
        num_iters = 0
        for input, target in train_loader:
            input = input.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            pred = net(input)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            running_loss += loss
            _, predicted = torch.max(pred.data, 1)
            total += target.size(0) 
            correct += (predicted == target).sum().item()
            num_iters += 1
        
        train_acc = correct / total
        train_loss = running_loss / num_iters
        eval_loss, eval_acc = evaluate(net, test_loader, criterion, num_cls)
        all_train_loss.append(train_loss)
        all_eval_loss.append(eval_loss)
        all_train_acc.append(train_acc)
        all_eval_acc.append(eval_acc)

        if print_loss:
            print('Epoch {}, train loss {:.3f}, eval loss {:.3f}, train acc {:.3f}, eval acc {:.3f}'.format(epoch, train_loss, eval_loss, train_acc, eval_acc))
    print('Finish training, time {:.3f}'.format(time.time() - tic))

    return all_train_loss, all_eval_loss, all_train_acc, all_eval_acc

        
def evaluate(net, data_loader, criterion, num_cls=0):
    net.eval()
    correct = 0
    total = 0
    running_loss = 0.
    num_iters = 0
    cls_acc = [0.] * num_cls
    cls_num = [0.] * num_cls
    std = 0.
    with torch.no_grad():
        for input, target in data_loader:
            input = input.to(device)
            target = target.to(device)
            pred = net(input)
            loss = criterion(pred, target)

            running_loss += loss
            
            _, predicted = torch.max(pred.data, 1)
            total += target.size(0) 
            correct += (predicted == target).sum().item()
            pred = predicted.flatten()
            targ = target.flatten()
            
            num_iters += 1
            eval_acc = correct / total
            for i in range(len(cls_acc)):
                cls_acc[i] /= (cls_num[i] + 1e-6)
                std += (cls_acc[i] - eval_acc)**2
   
        return running_loss / num_iters, eval_acc


if __name__ == "__main__":
    # Regress a function to check training process
    from dataset import RDiff
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = [20, 10]
    
    train_data = RDiff('datasets/rfunc_train.pkl')
    test_data = RDiff('datasets/rfunc_test.pkl')

    real_mlp = nn.Sequential(nn.Linear(1, 256),
                        nn.ReLU(),
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        nn.Linear(256, 512),
                        nn.ReLU(),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, 1),
                        )

    all_loss = train(real_mlp, train_data, test_data, lr=1e-3, num_epochs=10, batch_size=200)
    fig1 = plt.figure("Loss")
    fig1.gca().plot(all_loss)
    # Test & plot
    input, target = test_data[:-1]
    pred = real_mlp(torch.FloatTensor(input).unsqueeze(0).to(device))
    pred = pred.squeeze().cpu().detach().numpy()
    fig2 = plt.figure("Regression")
    fig2.gca().scatter(input.squeeze(), pred)
    plt.show()