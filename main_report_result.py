from libs import *

from utils import ECG_DB
from utils import static
import os
import torch
from utils import test


from model import seresnet
from model import resnet
from model import resnet_1d
from model import seresnet_1d
from model import resnet_orignal
# from model import ResCNN
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

# Function to report the result
def table_acc(file,net,args,val_loader,criterion,device):
    data=[]      
    net.load_state_dict(torch.load(args.path+file, map_location=device))
    size= static.get_size(args.path+file ,'mb')
    
    start = 1
    f1_arr, f1_mean =test.test(val_loader, net, criterion, device)
    stop = 2
    time=stop - start
    if len(f1_arr)==6:
        data = [file.split(".")[0],f1_mean,size,time,f1_arr[0],f1_arr[1], f1_arr[2], f1_arr[3], f1_arr[4],  f1_arr[5]]
            
    # SNR,AF,IAVB,LBBB,RBBB,STD        
    col_names = ["Model", "F1_mean",'Size(MB)','Time(s)', 'F1_SNR','F1_AF', 'F1_IAVB', 'F1_LBBB', 'F1_RBBB','F1_STD']
    return data,col_names



if __name__== "__main__":
    args = parse_args()
    args.best_metric = 0
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda') # Change to your suitable GPU device  
    data_dir = os.path.normpath(args.data_dir)
    
    label_csv = os.path.join('data/labels_6.csv') # Chang to your prefered label file (which inlude 6 classes)
    # The above label file has been divided into 10 folds, each fold has 5000 samples

    val_folds=[4,7] 
    val_dataset = ECG_DB.ECG_DB('val', data_dir, label_csv, val_folds)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    criterion = nn.BCEWithLogitsLoss()
    
    net = resnet_1d.ResNet18(num_classes=6).to(device) 
    file = 'resnet18.ptl'
    
    dic_model = {'resnet18.ptl': resnet_1d.ResNet18(num_classes=6).to(device), 
            'resnet34.ptl': resnet_1d.ResNet34(num_classes=6).to(device),
            'SqueezeNet.ptl': SqueezeNet.SqueezeNet().to(device), 
            'AlexNet.ptl': AlexNet.AlexNet().to(device),
            'resnet152.ptl': resnet_orignal.ResNet152(num_classes=6).to(device),
            'se_resnet18.ptl': seresnet_1d.se_resnet18(num_classes=6).to(device),
            'se_resnet34.ptl': seresnet_1d.se_resnet34(num_classes=6).to(device),
            'se_resnet152.ptl': seresnet.se_resnet152().to(device)}
    
    data_report = []
    # Go through all saved models in the folder
    for file in os.listdir(args.path):
        print(file)
        net = dic_model[file]
        data, col_names = table_acc(file,net,args,val_loader,criterion,device)
        data_report.append(data)
    
    print(tabulate(data_report, headers=col_names, tablefmt="fancy_grid", showindex="always"))
    
    
"""
Final result:

╒════╤═════════════╤═══════════╤════════════╤═══════════╤══════════╤══════════╤═══════════╤═══════════╤═══════════╤══════════╕
│    │ Model       │   F1_mean │   Size(MB) │   Time(s) │   F1_SNR │    F1_AF │   F1_IAVB │   F1_LBBB │   F1_RBBB │   F1_STD │
╞════╪═════════════╪═══════════╪════════════╪═══════════╪══════════╪══════════╪═══════════╪═══════════╪═══════════╪══════════╡
│  0 │ AlexNet     │  0.746857 │     91.12  │         1 │ 0.68984  │ 0.750533 │  0.480712 │ 0.877193  │  0.918499 │ 0.764368 │
├────┼─────────────┼───────────┼────────────┼───────────┼──────────┼──────────┼───────────┼───────────┼───────────┼──────────┤
│  1 │ resnet34    │  0.89537  │     63.491 │         1 │ 0.842105 │ 0.931677 │  0.893471 │ 0.945455  │  0.931174 │ 0.828338 │
├────┼─────────────┼───────────┼────────────┼───────────┼──────────┼──────────┼───────────┼───────────┼───────────┼──────────┤
│  2 │ SqueezeNet  │  0.355099 │      1.382 │         1 │ 0.272586 │ 0.359467 │  0.228435 │ 0.0977702 │  0.893048 │ 0.279286 │
├────┼─────────────┼───────────┼────────────┼───────────┼──────────┼──────────┼───────────┼───────────┼───────────┼──────────┤
│  3 │ se_resnet18 │  0.903815 │     33.774 │         1 │ 0.857143 │ 0.929577 │  0.892086 │ 0.963636  │  0.937748 │ 0.842697 │
├────┼─────────────┼───────────┼────────────┼───────────┼──────────┼──────────┼───────────┼───────────┼───────────┼──────────┤
│  4 │ resnet18    │  0.89533  │     33.434 │         1 │ 0.840909 │ 0.926829 │  0.886525 │ 0.944444  │  0.940397 │ 0.832877 │
├────┼─────────────┼───────────┼────────────┼───────────┼──────────┼──────────┼───────────┼───────────┼───────────┼──────────┤
│  5 │ se_resnet34 │  0.899856 │     64.105 │         1 │ 0.865922 │ 0.921162 │  0.888087 │ 0.944444  │  0.935948 │ 0.843575 │
╘════╧═════════════╧═══════════╧════════════╧═══════════╧══════════╧══════════╧═══════════╧═══════════╧═══════════╧══════════╛

"""