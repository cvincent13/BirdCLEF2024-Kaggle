import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from functools import partial
import yaml
import wandb

# https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, pos_weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, pos_weight=self.pos_weight, reduction="none")
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = bce_loss * ((1 - p_t) ** self.gamma)

        #if self.pos_weight is not None:
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * loss

        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss
    

# https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/213075
class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, preds, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=self.pos_weight)(preds, targets)
        probas = torch.sigmoid(preds)
        loss = targets * self.alpha * \
            (1. - probas)**self.gamma * bce_loss + \
            (1. - targets) * probas**self.gamma * bce_loss
        loss = loss.mean()
        return loss

class BCEFocal2WayLoss(nn.Module):
    def __init__(self, weights=[1, 1], alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.focal = BCEFocalLoss(alpha=alpha, gamma=gamma, pos_weight=pos_weight)
        self.weights = weights

    def forward(self, input, target):
        input_ = input["logit"]
        target = target.float()

        framewise_output = input["framewise_logit"]
        clipwise_output_with_max, _ = framewise_output.max(dim=1)

        loss = self.focal(input_, target)
        aux_loss = self.focal(clipwise_output_with_max, target)

        return self.weights[0] * loss + self.weights[1] * aux_loss
    

import math
def _get_cosine_schedule_with_warmup_lr_lambda(current_step, num_warmup_steps, num_training_steps, num_cycles, start_lr, final_lr):
    if current_step < num_warmup_steps:
        progress = float(current_step) / float(max(1, num_warmup_steps))
        return start_lr + progress * (1 - start_lr)
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(final_lr, final_lr + (1 - final_lr) * 0.5 * (1. + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1, start_lr=0, final_lr=0):
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        start_lr=start_lr,
        final_lr=final_lr
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def wandb_init(fold, config_class):
    """
    Initializes the W&B run by creating a config file and initializing a W&B run.
    """
    # Create a dictionary of configuration parameters
    config = {k:v for k,v in dict(vars(config_class)).items() if '__' not in k}
    config.update({"fold":int(fold)}) # int is to convert numpy.int -> int
    # Dump the configuration dictionary to a YAML file
    yaml.dump(config, open(f'{config_class.base_dir}config/{config_class.run_name}.yaml', 'w'),)
    # Load the configuration dictionary from the YAML file
    config = yaml.load(open(f'{config_class.base_dir}config/{config_class.run_name}.yaml', 'r'), Loader=yaml.FullLoader)
    # Initialize a W&B run with the given configuration parameters
    run = wandb.init(project="birdclef-2024",
                     name=config_class.run_name,
                     config=config,
                     group=config_class.wandb_group,
                     save_code=True,)
    return run

    
def log_wandb(valid_df):
    """Log and save validation results with missclassified examples as audio in W&B"""
    # Query only the rows with miss predictions
    #save_df = valid_df.query("miss==True")
    # Map the predicted and target labels to their corresponding names
    #save_df.loc[:, 'pred_name'] = save_df.pred.map(Config.label2name)
    #save_df.loc[:, 'target_name'] = save_df.target.map(Config.label2name)
    # Trim the dataframe for debugging purposes
    #if Config.debug:
    #    save_df = save_df.iloc[:Config.replicas*Config.batch_size*Config.infer_bs]
    # Get the columns to be included in the wandb table
    #noimg_cols = [*Config.tab_cols, 'target', 'pred', 'target_name','pred_name']
    # Retain only the necessary columns
    #save_df = save_df.loc[:, noimg_cols]

    #data = []
    # Load audio files for each miss prediction
    #for idx, row in tqdm(save_df.iterrows(), total=len(save_df), desc='wandb ', position=0, leave=True):
    #    filepath = '/kaggle/input/birdclef-2023/train_audio/'+row.filename
    #    audio, sr = librosa.load(filepath, sr=None)
        # Add the audio file to the data list along with the other relevant information
    #    data+=[[*row.tolist(), wandb.Audio(audio, caption=row.filename, sample_rate=sr)]]
    # Create a wandb table with the audio files and other relevant information
    #wandb_table = wandb.Table(data=data, columns=[*noimg_cols, 'audio'])
    # Log the scores and wandb table to wandb
    #wandb.log({'best': scores,
    #           'table': wandb_table,
    #           })
