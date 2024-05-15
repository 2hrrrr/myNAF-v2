import os, json, torch
import imageio.v2 as iio
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from tqdm import tqdm


from render import run_network, render
from data_load import TIGREDataset as Dataset
from data_load import test_dataloader
from network import DensityNetwork
from util import get_psnr, get_mse, get_psnr_3d, get_ssim_3d, cast_to_image, calc_mse_loss, get_encoder

def runUNet(model, args):
    state_dict = torch.load(args.test_model)
    model.load_state_dict(state_dict['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(view=args.view, batch_size=1, num_workers=0)
    torch.cuda.empty_cache()
    model.eval()
    raw = []
    with torch.no_grad():
        for _, data in enumerate(dataloader):
            input_img = data.to(device)
            pred = model(input_img)[2]
            pred_clip = torch.clamp(pred, 0, 1)
            raw.append(pred_clip.squeeze(0).squeeze(0))
    
    ret = torch.stack(raw, dim=0)
    return ret

class Trainer:
    def __init__(self, cfg, retrain=None, device="cuda"):

        # Args
        self.global_step = 0
        self.conf = cfg
        self.epochs = cfg["train"]["epoch"]
        self.i_eval = cfg["log"]["i_eval"]
        self.i_save = cfg["log"]["i_save"]
        self.netchunk = cfg["render"]["netchunk"]
        self.n_rays = cfg["train"]["n_rays"]
  
        # Log direcotry
        if retrain is None:
            self.expdir = os.path.join(cfg["exp"]["expdir"], cfg["exp"]["expname"])
        else:
            self.expdir = os.path.join(cfg["exp"]["expdir"], 'retrain', cfg["exp"]["expname"])
        self.ckptdir = os.path.join(self.expdir, "ckpt.tar")
        self.ckptdir_backup = os.path.join(self.expdir, "ckpt_backup.tar")
        self.evaldir = os.path.join(self.expdir, "eval")
        os.makedirs(self.evaldir, exist_ok=True)

        # Dataset
        self.train_dset = Dataset(cfg["exp"]["datadir"], cfg["train"]["n_rays"], "train", retrain, device)
        self.eval_dset = Dataset(cfg["exp"]["datadir"], cfg["train"]["n_rays"], "val", device)
        self.train_dloader = torch.utils.data.DataLoader(self.train_dset, batch_size=cfg["train"]["n_batch"])
        self.voxels = self.eval_dset.voxels
    
        # Network
        network = DensityNetwork
        encoder = get_encoder(**cfg["encoder"])
        self.net = network(encoder, **cfg["network"]).to(device)
        grad_vars = list(self.net.parameters())

        # Optimizer
        self.optimizer = torch.optim.Adam(params=grad_vars, lr=cfg["train"]["lrate"], betas=(0.9, 0.999))
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optimizer, step_size=cfg["train"]["lrate_step"], gamma=cfg["train"]["lrate_gamma"])

        # Load checkpoints
        self.epoch_start = 0
        if cfg["train"]["resume"] and os.path.exists(self.ckptdir) and (retrain is None):
            print(f"Load checkpoints from {self.ckptdir}.")
            ckpt = torch.load(self.ckptdir)
            self.epoch_start = ckpt["epoch"] + 1
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.global_step = self.epoch_start * len(self.train_dloader)
            self.net.load_state_dict(ckpt["network"])

        # Summary writer
        self.writer = SummaryWriter(self.expdir)
        self.writer.add_text("parameters", self.args2string(cfg), global_step=0)

    def args2string(self, hp):
        json_hp = json.dumps(hp, indent=2)
        return "".join("\t" + line for line in json_hp.splitlines(True))

    def start(self):
        def fmt_loss_str(losses):
            return "".join(", " + k + ": " + f"{losses[k].item():.3g}" for k in losses)
        
        
        iter_per_epoch = len(self.train_dloader)
        pbar = tqdm(total= iter_per_epoch * self.epochs, leave=True)
        if self.epoch_start > 0:
            pbar.update(self.epoch_start*iter_per_epoch)

        for idx_epoch in range(self.epoch_start, self.epochs+1):

            # Evaluate
            if (idx_epoch % self.i_eval == 0 or idx_epoch == self.epochs):
                self.net.eval()
                with torch.no_grad():
                    loss_test = self.eval_step(global_step=self.global_step, idx_epoch=idx_epoch)
                self.net.train()
                tqdm.write(f"[EVAL] epoch: {idx_epoch}/{self.epochs}{fmt_loss_str(loss_test)}")
            
            # Train
            for data in self.train_dloader:
                self.global_step += 1
                # Train
                self.net.train()
                loss_train = self.train_step(data, global_step=self.global_step, idx_epoch=idx_epoch)
                pbar.set_description(f"epoch={idx_epoch}/{self.epochs}, loss={loss_train:.3g}, lr={self.optimizer.param_groups[0]['lr']:.3g}")
                pbar.update(1)
            
            # Save
            if (idx_epoch % self.i_save == 0 or idx_epoch == self.epochs) and self.i_save > 0 and idx_epoch > 0:
                if os.path.exists(self.ckptdir):
                    copyfile(self.ckptdir, self.ckptdir_backup)
                tqdm.write(f"[SAVE] epoch: {idx_epoch}/{self.epochs}, path: {self.ckptdir}")
                torch.save(
                    {
                        "epoch": idx_epoch,
                        "network": self.net.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                    },
                    self.ckptdir,
                )

            # Update lrate
            self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)
            self.lr_scheduler.step()

        tqdm.write(f"Training complete! See logs in {self.expdir}")

    def train_step(self, data, global_step, idx_epoch):
        """
        Training step
        """
        self.optimizer.zero_grad()
        loss = self.compute_loss(data, global_step)
        loss.backward()
        self.optimizer.step()
        return loss.item()
        
    def compute_loss(self, data, global_step):
        rays = data["rays"].reshape(-1, 8)
        projs = data["projs"].reshape(-1)
        ret = render(rays, self.net, **self.conf["render"])
        projs_pred = ret["acc"]

        loss = {"loss": 0.}
        calc_mse_loss(loss, projs, projs_pred)

        # Log
        for ls in loss.keys():
            self.writer.add_scalar(f"train/{ls}", loss[ls].item(), global_step)

        return loss["loss"]


    def eval_step(self, global_step, idx_epoch):
        # Evaluate projection
        select_ind = np.random.choice(len(self.eval_dset))
        projs = self.eval_dset.projs[select_ind]
        rays = self.eval_dset.rays[select_ind].reshape(-1, 8)
        H, W = projs.shape
        projs_pred = []
        for i in range(0, rays.shape[0], self.n_rays):
            projs_pred.append(render(rays[i:i+self.n_rays], self.net, **self.conf["render"])["acc"])
        projs_pred = torch.cat(projs_pred, 0).reshape(H, W)

        # Evaluate density
        image = self.eval_dset.image
        image_pred = run_network(self.eval_dset.voxels, self.net, self.netchunk)
        image_pred = image_pred.squeeze()
        loss = {
            "proj_mse": get_mse(projs_pred, projs),
            "proj_psnr": get_psnr(projs_pred, projs),
            "psnr_3d": get_psnr_3d(image_pred, image),
            "ssim_3d": get_ssim_3d(image_pred, image),
        }

        # Logging
        show_slice = 5
        show_step = image.shape[-1]//show_slice
        show_image = image[...,::show_step]
        show_image_pred = image_pred[...,::show_step]
        show = []
        for i_show in range(show_slice):
            show.append(torch.concat([show_image[..., i_show], show_image_pred[..., i_show]], dim=0))
        show_density = torch.concat(show, dim=1)
        show_proj = torch.concat([projs, projs_pred], dim=1)

        self.writer.add_image("eval/density (row1: gt, row2: pred)", cast_to_image(show_density), global_step, dataformats="HWC")
        self.writer.add_image("eval/projection (left: gt, right: pred)", cast_to_image(show_proj), global_step, dataformats="HWC")

        for ls in loss.keys():
            self.writer.add_scalar(f"eval/{ls}", loss[ls], global_step)
            
        # Save
        eval_save_dir = os.path.join(self.evaldir, f"epoch_{idx_epoch:05d}")
        os.makedirs(eval_save_dir, exist_ok=True)
        np.save(os.path.join(eval_save_dir, "image_pred.npy"), image_pred.cpu().detach().numpy())
        np.save(os.path.join(eval_save_dir, "image_gt.npy"), image.cpu().detach().numpy())
        iio.imwrite(os.path.join(eval_save_dir, "slice_show_row1_gt_row2_pred.png"), (cast_to_image(show_density)*255).astype(np.uint8))
        iio.imwrite(os.path.join(eval_save_dir, "proj_show_left_gt_right_pred.png"), (cast_to_image(show_proj)*255).astype(np.uint8))
        with open(os.path.join(eval_save_dir, "stats.txt"), "w") as f: 
            for key, value in loss.items(): 
                f.write("%s: %f\n" % (key, value.item()))

        return loss
    
    def view_render(self):
        H, W = self.train_dset.projs[0].shape
        raw = []
        mx = []
        for i in range(self.train_dset.n_samples):
            rays = self.train_dset.rays2[i].reshape(-1, 8)
            projs_pred = []
            for j in range(0, rays.shape[0], self.n_rays):
                projs_pred.append(render(rays[j:j+self.n_rays], self.net, **self.conf["render"])["acc"])
            projs_pred = torch.cat(projs_pred, 0).reshape(H, W)
            
            mx.append(projs_pred.max())
            projs_pred = projs_pred / projs_pred.max()
            raw.append(projs_pred)
        
        ret = torch.stack(raw, dim=0)
        return ret, mx
