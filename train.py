import sys
import os
import torch
import torch.optim as optim
import yaml
import importlib
from time import time
import datetime
import numpy as np
from tqdm import tqdm
from test import random_rotate
from utils.logger import Logger

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def train(config, logger):
    num_epochs = int(config['num_epochs'])
    batch_size = int(config['batch_size'])
    learning_rate = float(config['learning_rate'])
    weight_decay = float(config['weight_decay'])
    # Data
    logger.log_string("Loading dataset...")

    data_dir = config[config['dataset']]
    val_data_dir = data_dir
    if config['dataset'] == 'ntu':
        from data.ntu.feeder import Feeder
        num_joints = 25
        num_cls = 60
    elif config['dataset'] == 'fpha':
        from data.fpha.feeder import Feeder
        num_joints = 21
        num_cls = 45
    else:
        raise ValueError
    logger.log_string('Data dir: {}, num_joints: {}, num_cls: {}'.format(data_dir, num_joints, num_cls))

    # Get model
    module, model_name = config['net'].rsplit('.', 1)
    logger.backup_files([os.path.join(*module.split('.')) + '.py'])
    module = importlib.import_module(module)
    model = getattr(module, model_name)
    print('model name', model_name)
    net = model(config['in_channels'], num_joints, config['data_param']['num_frames'], num_cls, config)
    device_ids = config['device_ids']
    print('device_ids', device_ids) 
    if config['resume'] is not '':
        logger.log_string('Resume from' + config['resume'])
        net.load_state_dict(torch.load(config['resume']))
    device = device_ids[0]
    net = net.to(device)

    def count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    logger.log_string('Model total number of params:' + str(count_params(net)))
    
    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
 
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5, last_epoch=config['start_epoch']-2) 
    
    train_label_path = os.path.join(data_dir, 'train_label.pkl')
    val_label_path = os.path.join(val_data_dir, 'val_label.pkl')
    train_edge_path = os.path.join(data_dir, 'train_data_rel.npy') if config['use_edge'] else None
    test_edge_path = os.path.join(val_data_dir, 'val_data_rel.npy') if config['use_edge'] else None

    if 'edge_only' in config and config['edge_only']:
        print(os.path.join(data_dir, 'train_data_rel.npy'))
        traindata = Feeder(os.path.join(data_dir, 'train_data_rel.npy'), train_label_path, None, num_samples=-1,
                                        mmap=True, num_frames=config['data_param']['num_frames'])
        testdata = Feeder(os.path.join(val_data_dir, 'val_data_rel.npy'), val_label_path, None, num_samples=-1,
                                        mmap=True, num_frames=config['data_param']['num_frames'])
    else:
        traindata = Feeder(os.path.join(data_dir, 'train_data.npy'), train_label_path, train_edge_path, num_samples=-1,
                                        mmap=True, num_frames=config['data_param']['num_frames'])
        testdata = Feeder(os.path.join(val_data_dir, 'val_data.npy'), val_label_path, test_edge_path, num_samples=-1,
                                        mmap=True, num_frames=config['data_param']['num_frames'])
    logger.log_string('Train samples %d'  % len(traindata))
    logger.log_string('Test samples %d' % len(testdata))
    
    trainloader = torch.utils.data.DataLoader(traindata, batch_size=batch_size,
                            shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    testloader = torch.utils.data.DataLoader(testdata, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    best_acc = 0.

    # Whether use schedular
    change_lr = True

    for epoch in range(config['start_epoch'], num_epochs + 1):
        np.random.seed() # reset seed
        tic = time()
        net.train()
        correct = 0
        total = 0
        running_loss = 0.0
        num_iters = 0
        # Train
        if torch.__version__  == '1.0.0':
            if change_lr:
                scheduler.step()  # Adjust learning rate
                logger.log_scalar_train('Learning rate', scheduler.get_lr()[0], epoch)
                print(scheduler.get_lr()[0])

        for data in tqdm(trainloader, total=len(trainloader), disable=not config['tqdm'],ascii=True):
        # for data in trainloader:
            inputs, labels = data
            if config['padding_input']:
                pad = torch.zeros([inputs.shape[0], 1, inputs.shape[2], inputs.shape[3],inputs.shape[4]])
                inputs = torch.cat([pad, inputs.type_as(pad)], dim=1)
            # Data Augmentation
            if config['data_augmentation']:
                inputs = random_rotate(inputs, y_only=True)
            if config['use_edge']:
                inputs[0], inputs[1], labels = inputs[0].to(device), inputs[1].to(device), labels.to(device)
            else:
                inputs, labels = inputs.to(device), labels.to(device)

            # Freeze ADJ matrix for some epochs
            if config['net'] in ['models.dgnn.Model', 'models.qdgnn.Model']:
                for name, params in net.named_parameters():
                    if 'source_M' in name or 'target_M' in name:
                        params.requires_grad = epoch > 10
                        
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = net.get_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            num_iters = num_iters + 1

        if torch.__version__ in ['1.1.0', '1.2.0']:
            if change_lr:
                scheduler.step()  # Adjust learning rate

        # Eval and metrics
        acc_train = correct / total
        loss_train = running_loss / num_iters
        acc_eval, loss_eval = evaluate(config, net, testloader)
        if acc_eval > best_acc:
            best_acc = acc_eval
            # Save trained model
            torch.save(net.state_dict(), os.path.join(config['logdir'], 'model.pkl'))
        logger.log_string('Epoch %d: train loss: %.5f, eval loss: %.5f, train acc: %.5f, eval acc: %.5f, time: %.5f' % (epoch, 
                                                                    loss_train, loss_eval, acc_train, acc_eval, time() - tic))
        logger.log_scalar_train('Loss', loss_train, epoch)
        logger.log_scalar_train('Accuracy', acc_train, epoch)
        logger.log_scalar_eval('Loss', loss_eval, epoch)
        logger.log_scalar_eval('Accuracy', acc_eval, epoch)

    logger.log_string('Best eval acc: %.5f' % (best_acc))
    logger.log_string('Finished Training')
    logger.close()
    

def evaluate(config, net, dataloader):
    device = config['device_ids'][0]
    np.random.seed() # reset seed
    num_iters = 0
    correct = 0
    total = 0
    running_loss = 0.0
    net.eval()
    with torch.no_grad():
        for data in tqdm(dataloader,ascii=True,disable=not config['tqdm']):
            inputs, labels = data
            if config['padding_input']:
                pad = torch.zeros([inputs.shape[0], 1, inputs.shape[2], inputs.shape[3], inputs.shape[4]])
                inputs = torch.cat([pad, inputs.type_as(pad)], dim=1)
            if config['use_edge']:
                inputs[0], inputs[1], labels = inputs[0].to(device), inputs[1].to(device), labels.to(device)
            else:
                inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = net.get_loss(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            num_iters += 1
    return correct / total, running_loss / num_iters


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError('Please enter config file path.')
    
    # Read configs
    configFile = sys.argv[1]
    with open(configFile, 'r') as f:
        config = yaml.safe_load(f)

    # Create a logger
    logger = Logger(config['logdir'])
    logger.log_string(datetime.datetime.now())
    logger.log_string(config)
    logger.backup_files(['train.py', 'qpu_ops.py', 'qpu_layers.py'])

    if config['dataset'] == 'ntu':
        logger.backup_files(['data/ntu/feeder.py'])
    if config['dataset'] == 'fpha':
        logger.backup_files(['data/fpha/feeder.py'])

    # Train
    train(config, logger)
