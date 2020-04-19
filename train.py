import torch
import torch.nn as nn
import argparse
import os
from model import Mymodel
from model_trainer import model_trainer
from torch.utils.data import DataLoader, random_split
from load_data import load_data
#annfile = './Data/instances_val2017.json'
#imgDir = './Data/val2017'
annfile = '../annotations/instances_val2017.json'
imgDir = '../val2017'
parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, help='the train batch size', default=1)
parser.add_argument('--lr', type=float, help='the learning rate', default=2e-3)
parser.add_argument('--epochs', type=int, help='training epochs', default=50)
parser.add_argument('--continue_train', action='store_true', help='load checkpoint and continue training')
parser.add_argument("--mode", default='client')
parser.add_argument("--port", default=50093)
args = parser.parse_args()

class train_model():
    def __init__(self,
                 imgDir,
                 annfile,
                 batch_size,
                 learning_rate,
                 num_epochs,
                 continue_train,
                 valid_percent):

        p = load_data(annFile= annfile, imgDir = imgDir)
        Data = p.load()
        self.batch_size = batch_size

        Num_valid = int(len(Data) * valid_percent)
        train_data, valid_data = random_split(Data, [len(Data) - Num_valid, Num_valid])
        del Data
        self.train_data = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=8,
                                pin_memory=True, drop_last=False)
        self.val_data = DataLoader(valid_data, batch_size=self.batch_size, shuffle=False, num_workers=8,
                              pin_memory=True, drop_last=False)
        if not os.path.exists(os.getcwd() + '/result'):
            os.mkdir(os.getcwd() + '/result')
        if not os.path.exists(os.getcwd() + '/checkpoint'):
            os.mkdir(os.getcwd() + '/checkpoint')
        torch.save(self.val_data.dataset.indices, os.getcwd() + '/result/val_ind.pt')
        self.model = Mymodel()
        checkpoint_dir = os.getcwd() + '/checkpoint/'
        checkpoint_path = checkpoint_dir + 'last_checkpoint.pytorch'
        if continue_train:
            try:
                self.model.load_state_dict(torch.load(checkpoint_path))
                print('load pre-trained model successful, continue training!')
            except:
                raise IOError(f"Checkpoint '{checkpoint_path}' load failed! ")
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(torch.cuda.is_available())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                print("There are", torch.cuda.device_count(), "GPUs!")
                print("But Let's use the first two GPUs!")
                self.model = nn.DataParallel(self.model, device_ids=[0, 1])
        self.model = self.model.to(self.device)

        self.trainer = model_trainer(model=self.model,
                                     learning_rate=learning_rate,
                                     num_epochs=num_epochs,
                                     batch_size=batch_size,
                                     device = self.device
                                     )

    def train(self):
            self.trainer.train(self.train_data, self.val_data)
def main():
    My_train = train_model(imgDir = imgDir,
                            annfile = annfile,
                            batch_size= args.batch,
                            learning_rate= args.lr,
                            num_epochs= args.epochs,
                            continue_train= args.continue_train,
                            valid_percent= 0.2)
    My_train.train()

if __name__ == '__main__':
    main()