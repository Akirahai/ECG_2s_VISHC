from libs import *
from model import resnet
from model import resnet_1d
from model import seresnet_1d


from utils import ECG_DB
from utils import train
from utils import validation
from utils import static
from model import seresnet

from model import resnet_orignal
from model import AlexNet
from model import SqueezeNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/chapman/CPSC/', help='Directory for data dir')
    parser.add_argument('--leads', type=str, default='all', help='ECG leads to use')
    parser.add_argument('--seed', type=int, default=42, help='Seed to split data') #42
    parser.add_argument('--num-classes', type=int, default=int, help='Num of diagnostic classes')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.00009, help='Learning rate') #0.0001
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Num of workers to load data')
    parser.add_argument('--phase', type=str, default='train', help='Phase: train or test')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume')
    parser.add_argument('--use-gpu', default=[1,2], action='store_true', help='Use GPU')
    parser.add_argument('--path', type=str, default="/home/ubuntu/hai.lv/IECG_new/ecg_2seconds/result/paper/") #Fix to your path to save model
    
    
    
    return parser.parse_args()



if __name__== "__main__":
    args = parse_args()
    args.best_metric = 0
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:1') # Change to your suitable GPU device
    data_dir = os.path.normpath(args.data_dir)
    
    label_csv = os.path.join('data/labels_6.csv') # Chang to your prefered label file (which inlude 6 classes)
    # The above label file has been divided into 10 folds, each fold has 5000 samples
    
    train_folds=[ 9 , 2 , 6 , 1 , 8,  3 ,10 , 5]
    val_folds=[4,7] 
    
    train_dataset = ECG_DB.ECG_DB('train', data_dir, label_csv, train_folds)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    val_dataset = ECG_DB.ECG_DB('val', data_dir, label_csv, val_folds)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    
    dic_model = {'resnet18.ptl': resnet_1d.ResNet18(num_classes=6).to(device), 
                'resnet34.ptl': resnet_1d.ResNet34(num_classes=6).to(device),
                'SqueezeNet.ptl': SqueezeNet.SqueezeNet().to(device), 
                'AlexNet.ptl': AlexNet.AlexNet().to(device),
                'resnet152.ptl': resnet_orignal.ResNet152(num_classes=6).to(device),
                'se_resnet18.ptl': seresnet_1d.se_resnet18(num_classes=6).to(device),
                'se_resnet34.ptl': seresnet_1d.se_resnet34(num_classes=6).to(device),
                'se_resnet152.ptl': seresnet.se_resnet152().to(device)}
    # Choose model
    
    bestmodel = 'se_resnet34.ptl'
    net = dic_model[bestmodel]
    
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr,betas=(0.1, 0.999)) # Define your optimizer (Adam with beta1=0.1, beta2=0.999)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=40) 
    # Define learning rate scheduler - dcay (CosineAnnealingLR with T_max=40)
    # This means that the learning rate will decrease from the initial value to zero over 40 epochs.
    criterion = nn.BCEWithLogitsLoss()
    
    print(bestmodel)
    if args.resume:
        net.load_state_dict(torch.load(args.path, map_location=device))
        
    for epoch in range(args.epochs):
        # Train and validation from utils
        train.train(train_loader, net, args, criterion, epoch, scheduler, optimizer, device)
        validation.validation(val_loader, net, args, criterion, device, bestmodel)
        
        
        