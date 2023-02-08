import os
import torch
import json


def save_model(out_dir, run_name, model):
    # Save model
    print('Saving model')
    run_dir = os.path.join(out_dir, run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    torch.save(model.state_dict(), os.path.join(run_dir, 'vae.pth'))
    print('Saving done!')


def save_train_config(out_dir, run_name, config):
    # Save training config
    run_dir = os.path.join(out_dir, run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    file_dir = os.path.join(run_dir, "config.json")
    print("Saving config to: " + file_dir)
    with open(file_dir, 'w') as f:
        json.dump(config, f, indent=2)