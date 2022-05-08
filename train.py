import os
import torch
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from config import config
from dataset import CXRDataset, MammogramDataset
from models import VAE, J, LSGANDiscriminator
from loss_functions import L1Loss, MSELoss, total_variation
from utils import save_loss_graph, save_image

# Initialize Datasets
train_data = MammogramDataset(config.train_metadata_path, transform=config.augmentation)
# train_data = CXRDataset('CXR_data/stage_2_train_labels.csv', transform=config.augmentation)
normal_data = MammogramDataset(config.normal_metadata_path, transform=config.augmentation)


# Initialize DataLoader
train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
normal_dataloader = DataLoader(normal_data, batch_size=config.batch_size, shuffle=True)

# Initialize Networks
G = VAE(input_size=512, encoder_norm='Instance', decoder_norm='Layer').to(config.device)
J = J().to(config.device)
F = VAE(input_size=512, encoder_norm=None, decoder_norm='Layer').to(config.device)
D = LSGANDiscriminator().to(config.device)

# Initialize Optimizers
optimizer_G = torch.optim.Adam(G.parameters(), lr=config.learning_rate, betas=(config.b1, config.b2), weight_decay=0.0001)
optimizer_J = torch.optim.Adam(J.parameters(), lr=config.learning_rate, betas=(config.b1, config.b2), weight_decay=0.0001)
optimizer_F = torch.optim.Adam(F.parameters(), lr=config.learning_rate, betas=(config.b1, config.b2), weight_decay=0.0001)
optimizer_D = torch.optim.Adam(D.parameters(), lr=config.learning_rate, betas=(config.b1, config.b2), weight_decay=0.0001)

# Initialize Lists for Saving Loss Function 
loss_D_list = []
loss_A_list = []
loss_R1_list = []
loss_R2_list = []
loss_R3_list = []
loss_TV_list = []
loss_G_list = []

# Initialize Schedulers
scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=10, gamma=0.5)
scheduler_J = torch.optim.lr_scheduler.StepLR(optimizer_J, step_size=10, gamma=0.5)
scheduler_F = torch.optim.lr_scheduler.StepLR(optimizer_F, step_size=10, gamma=0.5)
scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=10, gamma=0.5)

# Train networks
for epoch in tqdm(range(config.n_epoch)):
    for i, (z, normal) in enumerate(train_dataloader):
        # Train Discriminator
        z = z.to(config.device)
        syn_img = G(z)
        y = next(iter(normal_dataloader))
        y = y.to(config.device)

        valid_label = torch.ones((config.batch_size, 1), device=config.device, dtype=torch.float32)
        fake_label = torch.zeros((config.batch_size, 1), device=config.device, dtype=torch.float32)

        real_loss = MSELoss(D(y), valid_label)
        fake_loss = MSELoss(D(syn_img.detach()), fake_label)
        loss_D = (real_loss + fake_loss) * 0.5

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # Train Generators
        # Normal patients -> residue_map = 0
        if normal:
            residue_map = torch.zeros(z.shape, device=config.device, dtype=torch.float32)
        else:
            residue_map = F(z)

        z1 = syn_img + residue_map
        z2 = J(G.encoder(z), F.encoder(z))
        
        loss_A = 0.5 * MSELoss(D(syn_img), valid_label)
        loss_R1 = L1Loss(z1, z)
        loss_R2 = L1Loss(z2, z)
        loss_R3 = L1Loss(residue_map, torch.zeros(residue_map.shape, device=config.device, dtype=torch.float32))
        loss_TV = total_variation(syn_img)
        loss_G = config.lambda_A*loss_A + config.lambda_R*(loss_R1 + loss_R2 + loss_R3) + config.lambda_TV*loss_TV
        
        optimizer_G.zero_grad()
        optimizer_J.zero_grad()
        optimizer_F.zero_grad()
        
        loss_G.backward()
        
        optimizer_G.step()
        optimizer_J.step()
        optimizer_F.step()

        # Logging
        if (i + 1) % config.log_interval == 0:
            loss_D_list.append(loss_D.item())
            loss_A_list.append(loss_A.item())
            loss_R1_list.append(loss_R1.item())
            loss_R2_list.append(loss_R2.item())
            loss_R3_list.append(loss_R3.item())
            loss_TV_list.append(loss_TV.item())
            loss_G_list.append(loss_G.item())
            print(f"Epoch [{epoch+1}/{config.n_epoch}] Batch [{i+1}/{len(train_dataloader)}] Learning Rate: {scheduler_G.get_last_lr()}\nloss_D: {loss_D.item():.4f}\nloss_A: {loss_A.item():.4f} loss_R1: {loss_R1.item():.4f} loss_R2: {loss_R2.item():.4f} loss_R3: {loss_R3.item():.4f} loss_TV: {loss_TV.item():.4f} loss_G: {loss_G.item():.4f}\n")

    scheduler_G.step()
    scheduler_J.step()
    scheduler_F.step()
    scheduler_D.step()

    # Save synthesized images
    if (epoch + 1) % config.save_interval == 0:
        normal_img_save_path = os.path.join(config.normal_img_save_path, f"normal_img[{epoch+1}].png")
        residue_map_save_path = os.path.join(config.residue_map_save_path, f"residue_map_[{epoch+1}].png")
        r1_save_path = os.path.join(config.r1_save_path, f"reconstructed_1_[{epoch+1}].png")
        r2_save_path = os.path.join(config.r2_save_path, f"reconstructed_2_[{epoch+1}].png")
        
        normal_img = config.denormalize(syn_img)
        residue_map = config.denormalize(residue_map)
        r1 = config.denormalize(z1)
        r2 = config.denormalize(z2)
        
        save_image(normal_img, normal_img_save_path)
        save_image(residue_map, residue_map_save_path)
        save_image(r1, r1_save_path)
        save_image(r2, r2_save_path)

save_loss_graph(config.graph_path, loss_D_list, "loss of Discriminator", "loss_D")
save_loss_graph(config.graph_path, loss_G_list, "loss of Generators", "loss_G")
save_loss_graph(config.graph_path, loss_A_list, "Adversarial loss", "loss_A")
save_loss_graph(config.graph_path, loss_R1_list, "Reconsturction loss 1", "loss_R1")
save_loss_graph(config.graph_path, loss_R2_list, "Reconsturction loss 2", "loss_R2")
save_loss_graph(config.graph_path, loss_R3_list, "Reconsturction loss 3", "loss_R3")
save_loss_graph(config.graph_path, loss_TV_list, "Total variation loss", "loss_TV")