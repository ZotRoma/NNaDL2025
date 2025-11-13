import os
from PIL import Image
import torch
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.datasets import VOCSegmentation
import torch.nn as nn

from torchmetrics import JaccardIndex
from torchmetrics.segmentation import DiceScore

from tqdm import tqdm


def dice_coefficient(prediction, target, epsilon=1e-07):
    prediction_copy = prediction.clone()

    prediction_copy[prediction_copy < 0] = 0
    prediction_copy[prediction_copy > 0] = 1

    intersection = abs(torch.sum(prediction_copy * target))
    union = abs(torch.sum(prediction_copy) + torch.sum(target))
    dice = (2. * intersection + epsilon) / (union + epsilon)

    return dice

class CarvanaDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_transform=None, mask_transform=None):
        """
        images_dir: папка с jpg-картинками,
        masks_dir: папка с png-масками.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_ids = os.listdir(images_dir)
        self.img_transform = img_transform
        self.mask_transform = mask_transform


    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        #img_path = os.path.join(f'{self.images_dir}/{img_id}')
        #mask_path = os.path.join(self.masks_dir, img_id.replace('.jpg', '_mask.png'))  # Исправлено для PNG-версии
        image = Image.open(f'{self.images_dir}/{img_id}').convert('RGB')
        mask = Image.open(f'{self.masks_dir}/{img_id.replace('.jpg', '_mask.gif')}').convert('L')  # маска: один канал
        if self.img_transform:
            image = self.img_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()  # Бинарный тензор
        return image, mask
    
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.seq(x)
    
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # Энкодер
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        # Боттлнек
        self.bottleneck = DoubleConv(512, 1024)
        # Декодер (с транспонированными свёртками для увеличения размерности)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(512+512, 512)  # +512 из пропуска
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256+256, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(128+128, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(64+64, 64)
        # Финальный свёрточный слой: выдаём одно значение на пиксель
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Энкодер
        x1 = self.down1(x)            # выход 64 каналов
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x4 = self.down4(self.pool(x3))
        # Боттлнек
        xb = self.bottleneck(self.pool(x4))
        # Декодер с пропусками
        x = self.up4(xb)
        x = torch.cat([x4, x], dim=1)   # соединяем с соответствующим уровнем энкодера
        x = self.conv_up4(x)
        x = self.up3(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv_up3(x)
        x = self.up2(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv_up2(x)
        x = self.up1(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv_up1(x)
        x = self.final(x)
        return x
    
if __name__ == "__main__":
    sizes = [128, 256, 512]  # Размеры для теста
    results = {}
    for size in sizes:
        img_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        mask_transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        # Датасет
        full_dataset = CarvanaDataset(images_dir='./train/train', masks_dir='./train_masks/train_masks',
                                    img_transform=img_transform, mask_transform=mask_transform)

        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size

        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=8, num_workers= 4,persistent_workers=True, pin_memory=True,  shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8,num_workers= 4,  shuffle=False)

        # Модель, loss, optimizer
        model = UNet()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        print('cuda' if torch.cuda.is_available() else 'cpu')
        # Обучение (пример loop)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model.to(device)
        num_epochs = 5  # пример
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for images, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):  # train_loader возвращает (batch,3,H,W) и (batch,1,H,W)
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                optimizer.zero_grad()
                outputs = model(images)           # (batch,1,H,W) логиты
                loss = criterion(outputs, masks)  # поэлементная BCE
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        torch.save(model.state_dict(), f'{size}_UNet.pth')

        iou = JaccardIndex(task='binary').to(device)
        model.eval()
        total_iou, total_dice = 0, 0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader):
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                total_iou += iou(preds, masks)
                total_dice += dice_coefficient(preds, masks)
        IoU = total_iou / len(val_loader)
        Dice = total_dice / len(val_loader)
        print(f'Size {size}, IoU: {IoU}, Dice: {Dice}')
        results[size] = {'IoU': IoU, 'Dice': Dice}
