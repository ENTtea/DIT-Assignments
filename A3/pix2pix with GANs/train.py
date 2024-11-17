import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from torch.optim.lr_scheduler import StepLR
from model import G, D

def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    # Move tensor to CPU, detach from graph, and convert to NumPy array
    image = tensor.cpu().detach().numpy()
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    # Scale to [0, 255] and convert to uint8
    image = (image * 255).astype(np.uint8)
    return image

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Current epoch number.
        num_images (int): Number of images to save from the batch.
    """
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    for i in range(num_images):
        # Convert tensors to images
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        # Concatenate the images horizontally
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        # Save the comparison image
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)

def train_one_epoch(G_net, D_net, dataloader, optimizer_G, optimizer_D, criterion, criterion_L1, device, epoch, num_epochs):
    """
    Train the model for one epoch.

    Args:
        G_net (nn.Module): The neural network model of Generator.
        D_net (nn.Module): The neural network model of Discriminator.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer_G (Optimizer): Optimizer for updating G_net parameters.
        optimizer_D (Optimizer): Optimizer for updating D_net parameters.
        criterion (Loss): BCE Loss function.
        criterion_L1 (Loss): L1 Loss function.
        device (torch.device): Device to run the training on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    G_net.train()
    D_net.train()

    for i, (image_rgb, image_semantic) in enumerate(dataloader):
        # Move data to the device
        realA = image_semantic.to(device)
        realB = image_rgb.to(device)

        optimizer_D.zero_grad()

        # Forward pass
        fakeB = G_net(realA).to(device)

        realAB = torch.cat([realA, realB], dim=1).to(device)
        fakeAB = torch.cat([realA, fakeB], dim=1).to(device)
        
        # D_net loss
        realABD = D_net(realAB)
        label_real_D = torch.ones_like(realABD).to(device)
        err_real_D = criterion(realABD, label_real_D)
        # err_real_D.backward()

        fakeABD = D_net(fakeAB.detach())
        label_fake_D = torch.zeros_like(fakeABD).to(device)
        err_fake_D = criterion(fakeABD, label_fake_D)
        # err_fake_D.backward()

        err_D = (err_real_D+err_fake_D) / 2
        err_D.backward()

        optimizer_D.step()

        # G_net loss
        optimizer_G.zero_grad()

        fakeABG = D_net(fakeAB)
        label_real_G = torch.ones_like(fakeABG).to(device)
        err_fake_G = 100 * criterion_L1(fakeB, realB) + criterion(fakeABG, label_real_G)

        err_fake_G.backward()
        optimizer_G.step()
        
        with torch.no_grad():
            g = G_net(realA).to(device)
            gx = torch.cat([realA, g], dim=1).to(device)
            loss = 100*criterion_L1(g, realB) + criterion(D_net(gx), label_fake_D) +\
                  criterion(D_net(realAB), label_real_D)

            # Save sample images every 5 epochs
            if epoch % 50 == 0 and i == 0:
                save_images(image_semantic, image_rgb, g, 'train_results', epoch)

            # Print loss information
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')

def validate(G_net, D_net, dataloader, criterion, criterion_L1, device, epoch, num_epochs):
    """
    Validate the model on the validation dataset.

    Args:
        G_net (nn.Module): The neural network model of Generator.
        D_net (nn.Module): The neural network model of Discriminator.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer_G (Optimizer): Optimizer for updating G_net parameters.
        optimizer_D (Optimizer): Optimizer for updating D_net parameters.
        criterion (Loss): BCE Loss function.
        criterion_L1 (Loss): L1 Loss function.
        device (torch.device): Device to run the validation on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    G_net.eval()
    D_net.eval()

    val_loss = 0.0

    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(dataloader):
            # Move data to the device
            realB = image_rgb.to(device)
            realA = image_semantic.to(device)

            # Forward pass
            fakeB = G_net(realA).to(device)
            realAB = torch.cat([realA, realB], dim=1).to(device)
            fakeAB = torch.cat([realA, fakeB], dim=1).to(device)

            fakeABD = D_net(fakeAB)
            realABD = D_net(realAB)

            label_real_D = torch.ones_like(realABD).to(device)
            label_fake_D = torch.zeros_like(fakeABD).to(device)

            # Compute the loss
            loss = 100*criterion_L1(fakeB, realB) + criterion(fakeABD, label_fake_D) +\
                  criterion(realABD, label_real_D)
            val_loss += loss.item()

            # Save sample images every 5 epochs
            if epoch % 50 == 0 and i == 0:
                save_images(image_semantic, image_rgb, fakeB, 'val_results', epoch)

    # Calculate average validation loss
    avg_val_loss = val_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')

def main():
    """
    Main function to set up the training and validation processes.
    """
    # Set device to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize datasets and dataloaders
    train_dataset = FacadesDataset(list_file='train_list.txt')
    val_dataset = FacadesDataset(list_file='val_list.txt')

    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=4)

    # Initialize model, loss function, and optimizer
    G_net = G().to(device)
    D_net = D().to(device)
    criterion_L1 = nn.L1Loss()
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(G_net.parameters(), lr=0.005, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D_net.parameters(), lr=0.01, betas=(0.5, 0.999))

    # Add a learning rate scheduler for decay
    scheduler_G = StepLR(optimizer_G, step_size=100, gamma=0.2)
    scheduler_D = StepLR(optimizer_D, step_size=100, gamma=0.1)

    # Training loop
    num_epochs = 600
    for epoch in range(num_epochs):
        train_one_epoch(G_net, D_net, train_loader, optimizer_G, optimizer_D, criterion, criterion_L1, device, epoch, num_epochs)
        validate(G_net, D_net, val_loader, criterion, criterion_L1, device, epoch, num_epochs)

        # Step the scheduler after each epoch
        scheduler_G.step()
        scheduler_D.step()

if __name__ == '__main__':
    main()
