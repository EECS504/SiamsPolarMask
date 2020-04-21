from model import Mymodel
from load_data import load_data
import torch
import torch.nn as nn
import os
import numpy as np
import time
import argparse
from torch.utils.data import DataLoader
annfile = '../annotations/instances_val2017.json'
imgDir = '../val2017'
save_dir = 'result'
parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, help='the train batch size', default=1)
parser.add_argument('--is_best', action='store_true', help='load best / last checkpoint')
parser.add_argument("--mode", default='client')
parser.add_argument("--port", default=50093)
args = parser.parse_args()

class test_model():
    def __init__(self,
                 imgDir,
                 annfile,
                 batch_size,
                 is_best=False):

        p = load_data(annFile= annfile, imgDir = imgDir)
        Data = p.load()
        self.batch_size = batch_size

        self.test_data = DataLoader(Data, batch_size=self.batch_size, shuffle=True, num_workers=8,
                                pin_memory=True, drop_last=False)

        self.model = Mymodel()
        checkpoint_dir = './checkpoint/'

        if is_best:
            checkpoint_path = checkpoint_dir + 'best_checkpoint.pytorch'
            print('Now loading the best checkpoint!')
        else:
            checkpoint_path = checkpoint_dir + 'last_checkpoint.pytorch'
            print('Now loading the last checkpoint!')
        try:
            self.model.load_state_dict(torch.load(checkpoint_path))
            print('load pre-trained model successful!')
        except:
            raise IOError(f"load Checkpoint '{checkpoint_path}' failed! ")

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                print("There are", torch.cuda.device_count(), "GPUs!")
                print("But Let's use the first two GPUs!")
                self.model = nn.DataParallel(self.model, device_ids=[0, 1])
        self.model.to(self.device)

    def test(self):
        self.model.eval()
        with torch.no_grad():
            pred = {'cls': [], 'mask_reg': [], 'centerness': []}
            image_num = len(self.test_data.dataset)
            niter = np.int(np.ceil(image_num / self.batch_size))
            start_t = time.time()
            for iter_num, batch in enumerate(self.test_data):
                for key in batch:
                    assert isinstance(batch[key], type(batch['template'])) or isinstance(batch[key],
                                                                                         type(batch['targets']))
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(self.device)
                    else:
                        for sub_key in batch[key]:
                            batch[key][sub_key] = batch[key][sub_key].to(self.device)
                batch_template = batch['template']
                batch_detection = batch['detection']

                cls, mask_reg, centerness = self.model.forward(batch_template, batch_detection)
                pred['cls'].append(cls)
                pred['mask_reg'].append(mask_reg)
                pred['centerness'].append(centerness)
                print('(test {} / {})'.format(iter_num + 1, niter))
            end_t = time.time()
            print('total inference time: ', end_t - start_t)
        return pred

def main():
    My_test = test_model(imgDir = imgDir,
                            annfile = annfile,
                            batch_size= 32,
                            is_best= args.is_best)
    results = My_test.test()
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    torch.save(results, save_dir + '/network_output.pt')


if __name__ == '__main__':
    main()