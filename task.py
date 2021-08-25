import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import os
import random
import numpy as np
from datetime import datetime
from google.cloud import storage


def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def evaluate(model, device, test_loader):

    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return accuracy

def main():

    model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

    # Each process runs on GPU devices specified by the local_rank argument.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.", default=100)
    parser.add_argument("--batch_size", type=int, help="Training batch size for one process.", default=1024)
    parser.add_argument("--learning_rate", dest='learning_rate', type=float, help="Learning rate.", default=0.1)
    parser.add_argument("--random_seed", type=int, help="Random seed.", default=0)
    parser.add_argument("--model_dir", type=str, help="Directory for saving models.", default=os.environ['AIP_MODEL_DIR'] if 'AIP_MODEL_DIR' in os.environ else "")
    parser.add_argument("--model_filename", type=str, help="Model filename.", default="resnet_distributed.pth")
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
    parser.add_argument('--local_training', dest='local_training', action='store_true',
                    help='use local machine for training')
    parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
    parser.add_argument("--resume", action="store_true", help="Resume training from saved checkpoint.")
    parser.add_argument('--world-size', default=int(os.getenv('WORLD_SIZE', -1)), type=int,
                    help='number of nodes for distributed training')
    parser.add_argument('--dist-url', default='http://localhost:8082', type=str,
                    help='url used to set up distributed training') # From https://cloud.google.com/ai-platform/training/docs/distributed-pytorch#updating_your_training_code
    parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')  
    parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')  
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    argv = parser.parse_args()

    if argv.dist_url == "env://" and argv.world_size == -1:
        argv.world_size = int(os.environ["WORLD_SIZE"])

    argv.distributed = argv.world_size > 1 or argv.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    
    # debugging
    print (f"os WORLD_SIZE={os.getenv('WORLD_SIZE', -1)}")
    print (f"os RANK={os.getenv('RANK', 0)}")
    print (f"os MASTER_ADDR={os.getenv('MASTER_ADDR', 'localhost')}")
    print (f"os MASTER_PORT={os.getenv('MASTER_PORT', '8082')}")
    print (f'Arg - distributed={argv.distributed}')
    print (f'Arg - multiprocessing_distributed={argv.multiprocessing_distributed}')
    print (f'Arg - dist_backend={argv.dist_backend}')
    print (f'Arg - dist_url={argv.dist_url}')
    print (f'ngpus_per_node={ngpus_per_node}')

    start = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    print (f'Starting training: {start}')       
    if argv.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        argv.world_size = ngpus_per_node * argv.world_size
        print ('GPU x WORLD SIZE = {}'.format(argv.world_size))
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, argv))
    else:
        # Simply call main_worker function
        main_worker(argv.gpu, ngpus_per_node, argv)

    end = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    print (f'Training complete: {end}')

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    
    if args.gpu is not None:
        print("Use GPU: {args.gpu} for training")

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
            print (f"Distributed and getting rank from os.environ: rank={args.rank}")
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
            print (f"Distributed and Multiprocesing. Setting rank for each worker. rank={args.rank}")
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        print ("Process group initialized")
    
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    #learning_rate = args.learning_rate
    random_seed = args.random_seed
    model_dir = args.model_dir
    model_filename = args.model_filename
    resume = args.resume
    model_filepath = os.path.join(model_dir, model_filename)

    # We need to use seeds to make sure that the models initialized in different processes are the same
    set_random_seeds(random_seed=random_seed)
    print ("random seeds")
    
    # create model
    if args.pretrained:
        print(f"=> using pre-trained model '{args.arch}'")
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print(f"=> creating model '{args.arch}'")
        model = models.__dict__[args.arch]()   

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

            # Encapsulate the model on the GPU assigned to the current process
            device = torch.device("cuda:{}".format(args.rank))
            print (f"Distributed GPU device={device}")
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            print (f"Distributed CPU device used")
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        device = torch.device(f"cuda:{args.rank}")
        print (f"Non-distributed GPU device id ={args.gpu}")
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()  
        print (f"Non distributed CPU device used")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    # We only save the model who uses device "cuda:0"
    # To resume, the device for the saved model would also be "cuda:0"
    #if resume == True:
    #    map_location = {"cuda:0": "cuda:{}".format(local_rank)}
    #    ddp_model.load_state_dict(torch.load(model_filepath, map_location=map_location))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = f'cuda:{args.gpu}'
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print(f"=> no checkpoint found at '{args.resume}'")        
    
    cudnn.benchmark = True    
    
    # Prepare dataset and dataloader
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Data should be prefetched
    # Download should be set to be False, because it is not multiprocess safe
    train_set = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=transform) 
    test_set = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

    # Restricts data loading to a subset of the dataset exclusive to the current process
    train_sampler = DistributedSampler(dataset=train_set)

    # Load training data - set num_workers to turn multi-process data loading
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, sampler=train_sampler, num_workers=8)
    
    # Test loader does not have to follow distributed sampling strategy
    # Load training data - set num_workers to turn multi-process data loading
    test_loader = DataLoader(dataset=test_set, batch_size=128, shuffle=False, num_workers=8)

    #criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)

    # Loop over the dataset multiple times
    for epoch in range(num_epochs):

        epoch_start = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        print("Rank: {}, Epoch: {}, Training start: {}".format(args.rank, epoch, epoch_start))
        
        # Evaluate model routinely
        if epoch % 10 == 0:
            if args.rank == 0:
                accuracy = evaluate(model=model, device=device, test_loader=test_loader)
                if args.local_training:
                    torch.save(model.state_dict(), model_filepath)
                    print ('saving model to local folders')
                else:
                    # Save locally, then copy to GCS - https://cloud.google.com/vertex-ai/docs/training/exporting-model-artifacts
                    torch.save(model.state_dict(), model_filename)
                    # Upload model artifact to Cloud Storage
                    model_directory = os.environ['AIP_MODEL_DIR']
                    print (f"AIP_MODEL_DIR={model_directory}")
                    storage_path = os.path.join(model_directory,'model', model_filename)
                    print (f"storage_path={storage_path}")
                    blob = storage.blob.Blob.from_string(storage_path, client=storage.Client())
                    blob.upload_from_filename(model_filename)
                print("-" * 75)
                epoch_middle = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                print(f"Epoch: {epoch}, Accuracy: {accuracy}, Time: {epoch_middle}")
                print("-" * 75)

        # Switch to training mode
        model.train()    
            
        for data in train_loader:
            if args.gpu is not None:
                inputs = data[0].cuda(args.gpu, non_blocking=True)
                print(f'training with gpu {args.gpu}')
            labels = data[1].cuda(args.gpu, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    epoch_end = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    print (f'Epoch complete: {epoch_end}')
            
if __name__ == "__main__":
    main()
