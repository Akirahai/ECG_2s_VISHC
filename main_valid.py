from libs import *
import collections

from utils import ECG_DB
from utils import train
from utils import validation
from utils import static
from utils import static
from utils import test

from model import seresnet
from model import resnet
from model import resnet_1d
from model import seresnet_1d
from model import resnet_orignal
# from model import ResCNN
from model import AlexNet

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

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

# Creat function to load each saved model for confusion matrix

def model_confusion_matrix(file,net,args,val_loader,criterion,device):
    net.load_state_dict(torch.load(args.path+file, map_location=device))
    f1_arr, f1_mean, y_true, y_scores = test.test(val_loader, net, criterion, device)
    
    y_pred = np.zeros(y_scores.shape)
    for i in range(len(y_true)):
        counter_number = [idx for idx, val in enumerate(y_true[i])if val > 0.0]#   1 --- 4
        sorted_score= sorted(y_scores[i], reverse=True)
        # print(sorted_score) 
        for j in range(len(y_pred[i])):
            # print(len(y_scores[i]))
            if y_scores[i][j] > sorted_score[len(counter_number)]:
                y_pred[i][j]=1
            else:
                y_pred[i][j]=0

    conf_matrix = confusion_matrix(np.argmax(y_pred, axis=1), np.argmax(y_true, axis=1))

    # Plot confusion matrix
    plt.figure(figsize = (20,14))
    sns.set(font_scale = 2)

    sn=sns.heatmap(conf_matrix, annot=True,cmap="OrRd", square=True,fmt='g',annot_kws={'size': 20}, linewidths=2.6, xticklabels=['SNR','AF', 'IAVB', 'LBBB', 'RBBB','STD'], yticklabels=['SNR','AF', 'IAVB', 'LBBB', 'RBBB','STD'])
    sn.set_xlabel('Predicted Diseases', fontsize=20)
    sn.set_ylabel('True Diseases', fontsize=20)
    plt.title('Confusion Matrix for ' + file )
    plt.show()
    plt.savefig('/home/ubuntu/hai.lv/IECG_new/ecg_2seconds/result/images/Confusion_Matrix_' + file[:-4] + '.png')

    
    


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
        model_confusion_matrix(file,net,args,val_loader,criterion,device)

