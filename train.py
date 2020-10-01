from model.UNet_comb_reverse.system import UNetSystem
from model.UNet_comb_reverse.modelCheckpoint import BestAndLatestModelCheckpoint
import os
import pytorch_lightning as pl
import json
import argparse
from pathlib import Path

def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("image_path_1", help="/home/vmlab/Desktop/data/patch/Abdomen/28-44-44/image")
    parser.add_argument("image_path_2", help="/home/vmlab/Desktop/data/patch/Abdomen/28-44-44/image")
    parser.add_argument("label_path", help="/home/vmlab/Desktop/data/patch/Abdomen/28-44-44/image")
    parser.add_argument("model_savepath", help="/home/vmlab/Desktop/data/modelweight/Abdomen/28-44-44/mask")
    parser.add_argument("--train_list", help='["00", "01"]', nargs="*", default= ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19"])
    parser.add_argument("--val_list", help='["20", "21"]', nargs="*", default=["20", "21", "22", "23", "24", "25", "26", "27", "28", "29"])
    parser.add_argument("--log", help="/home/vmlab/Desktop/data/log/Abdomen/28-44-44/mask", default="log")
    parser.add_argument("--in_channel_main", help="Input channlel", type=int, default=1)
    parser.add_argument("--in_channel_final", help="Input channlel", type=int, default=64)
    parser.add_argument("--num_class", help="The number of classes.", type=int, default=14)
    parser.add_argument("--lr", help="Default 0.001", type=float, default=0.001)
    parser.add_argument("--batch_size", help="Default 2", type=int, default=2)
    parser.add_argument("--num_workers", help="Default 6.", type=int, default=6)
    parser.add_argument("--epoch", help="Default 50.", type=int, default=50)
    parser.add_argument("--gpu_ids", help="Default 0.", type=int, default=0, nargs="*")

    parser.add_argument("--api_key", help="Your comet.ml API key.")
    parser.add_argument("--project_name", help="Project name log is saved.")
    parser.add_argument("--experiment_name", help="Experiment name.", default="3DU-Net")


    args = parser.parse_args()

    return args

def main(args):
    criteria = {
            "train" : args.train_list, 
            "val" : args.val_list
            }

    image_path_list = [args.image_path_1, args.image_path_2]
    system = UNetSystem(
            image_path_list = image_path_list,
            label_path = args.label_path,
            criteria = criteria,
            in_channel_main = args.in_channel_main,
            in_channel_final = args.in_channel_final,
            num_class = args.num_class,
            learning_rate = args.lr,
            batch_size = args.batch_size,
            num_workers = args.num_workers, 
            checkpoint = BestAndLatestModelCheckpoint(args.model_savepath)
            )

    if args.api_key != "No": 
        from pytorch_lightning.loggers import CometLogger
        comet_logger = CometLogger(
                api_key = args.api_key,
                project_name =args. project_name,  
                experiment_name = args.experiment_name,
                save_dir = args.log
        )

        trainer = pl.Trainer(
                num_sanity_val_steps = 0, 
                max_epochs = args.epoch,
                checkpoint_callback = None, 
                logger = comet_logger,
                gpus = args.gpu_ids
            )
 
    else:
        trainer = pl.Trainer(
                num_sanity_val_steps = 0, 
                max_epochs = args.epoch,
                checkpoint_callback = None, 
                gpus = args.gpu_ids
            )
 
    trainer.fit(system)

if __name__ == "__main__":
    args = parseArgs()
    main(args)
