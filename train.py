import warnings
warnings.filterwarnings('ignore')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.utils.data import DataLoader
from utils.dataset import QaTa
import utils.config as config
from torch.optim import lr_scheduler
from engine.wrapper import UA_VLSWrapper

import pytorch_lightning as pl    
from torchmetrics import Accuracy,Dice
from torchmetrics.classification import BinaryJaccardIndex
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        description='Language-guide Medical Image Segmentation')
    parser.add_argument('--config',
                        default='./config/training.yaml',
                        type=str,
                        help='config file')

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)

    return cfg

from pytorch_lightning.callbacks import Callback

class TestAfterEachEpochCallback(Callback):
    def __init__(self, test_dataloader, save_path):
        super().__init__()
        self.test_dataloader = test_dataloader
        self.best_test_dice = -1.0
        self.save_path = os.path.join(save_path, "best_test_model.ckpt")

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        
        pl_module.eval()
        device = pl_module.device
        all_outputs = []

        print(f"\nRunning test after epoch: {trainer.current_epoch}")

        with torch.no_grad():
            for batch in self.test_dataloader:
                x, y = batch
                x, y = [x[0].to(device), [i.to(device) for i in x[1]]], y.to(device)
                batch = (x, y)
                output = pl_module.test_step(batch, 0)
                output = pl_module.test_step_end(output)
                all_outputs.append(output)

        final_metrics = pl_module.shared_epoch_end(all_outputs, stage="test")
        print(final_metrics, '\n')


        test_dice = final_metrics.get("test_dice", None)
        if test_dice is not None and test_dice > self.best_test_dice:
            self.best_test_dice = test_dice
            print(f"<<<<<< Saving new best model with test_dice = {test_dice:.4f} >>>>>>")
            ckpt = {
                'epoch': trainer.current_epoch,
                'state_dict': pl_module.state_dict(),
                'optimizer_states': [opt.state_dict() for opt in trainer.optimizers],
                'test_dice': test_dice
            }
            torch.save(ckpt, self.save_path)

if __name__ == '__main__':

    args = get_parser()
    print("cuda:", torch.cuda.is_available())

    ds_train = QaTa(csv_path=args.train_csv_path,
                    root_path=args.train_root_path,
                    tokenizer=args.bert_type,
                    image_size=args.image_size,
                    mode='train')

    ds_valid = QaTa(csv_path=args.train_csv_path,
                    root_path=args.train_root_path,
                    # csv_path=args.test_csv_path,
                    # root_path=args.test_root_path,
                    tokenizer=args.bert_type,
                    image_size=args.image_size,
                    mode='valid')
    
    ds_test = QaTa(csv_path=args.test_csv_path,
                    root_path=args.test_root_path,
                    tokenizer=args.bert_type,
                    image_size=args.image_size,
                    mode='test')
    

    dl_train = DataLoader(ds_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.train_batch_size)
    dl_valid = DataLoader(ds_valid, batch_size=args.valid_batch_size, shuffle=False, num_workers=args.valid_batch_size)
    dl_test = DataLoader(ds_test, batch_size=args.valid_batch_size, shuffle=False, num_workers=args.valid_batch_size)

    model = UA_VLSWrapper(args)
    
    os.makedirs(args.model_save_path, exist_ok=True)

    ## 1. setting recall function
    model_ckpt = ModelCheckpoint(
        dirpath=os.path.join(args.model_save_path, args.data_name),
        filename=args.model_save_filename,
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        verbose=True,
    )

    early_stopping = EarlyStopping(monitor = 'val_loss',
                            patience=args.patience,
                            mode = 'min'
    )

    os.makedirs(os.path.join(args.model_save_path, args.data_name), exist_ok=True)
    test_after_each_epoch = TestAfterEachEpochCallback(dl_test, os.path.join(args.model_save_path, args.data_name))

    ## 2. setting trainer
    trainer = pl.Trainer(logger=True,
                        min_epochs=args.min_epochs,max_epochs=args.max_epochs,
                        accelerator='gpu', 
                        devices=args.device,
                        callbacks=[model_ckpt, early_stopping , test_after_each_epoch],
                        enable_progress_bar=False,
                        ) 

    ## 3. start training
    print('start training')
    trainer.fit(model, dl_train, dl_valid)
    print('done training')
    ## 4. test
    model.eval()
    trainer.test(model, dl_test, ckpt_path='best')