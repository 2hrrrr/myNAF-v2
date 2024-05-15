import argparse, yaml, torch, gc
from trainer import Trainer, runUNet
from network import MIMOUNet

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="./config/chest_25.yaml", help="configs file path")
args = parser.parse_args()

with open(args.config, "r") as f:
    cfg = yaml.load(f, Loader=yaml.Loader)

print(f"[Start] exp: {cfg['exp']['expname']}, net: Basic network")
trainer = Trainer(cfg)
trainer.start()
gc.collect()
torch.cuda.empty_cache()
trainer.net.eval()

# run the U-Net
with torch.no_grad():
    args.model_name = 'MIMO-UNet'
    args.test_model = 'data/MIMO-UNet.pkl'
    args.view, mx = trainer.view_render()

model = MIMOUNet()
model.cuda()
retrain = runUNet(model, args)

for i in range(len(retrain)):
    retrain[i] = retrain[i] / torch.max(retrain[i]) * mx[i]

retrainer = Trainer(cfg, retrain)
retrainer.start()