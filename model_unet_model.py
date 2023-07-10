import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class UNet(pl.LightningModule):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder1 = self.double_conv(in_channels, 64)
        self.encoder2 = self.double_conv(64, 128)
        self.encoder3 = self.double_conv(128, 256)
        self.encoder4 = self.double_conv(256, 512)

        self.decoder1 = self.double_conv(512 + 256, 256)
        self.decoder2 = self.double_conv(256 + 128, 128)
        self.decoder3 = self.double_conv(128 + 64, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(F.max_pool2d(x1, kernel_size=2, stride=2))
        x3 = self.encoder3(F.max_pool2d(x2, kernel_size=2, stride=2))
        x4 = self.encoder4(F.max_pool2d(x3, kernel_size=2, stride=2))

        x = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, x3], dim=1)
        x = self.decoder1(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, x2], dim=1)
        x = self.decoder2(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, x1], dim=1)
        x = self.decoder3(x)

        x = self.final_conv(x)

        return x

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y_true.squeeze(1))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self(x)
        _, predicted_labels = torch.max(y_pred, dim=1)  # Convertir les prédictions en labels prédits
        loss = F.cross_entropy(y_pred, y_true.squeeze(1))
        accuracy = (predicted_labels == y_true.squeeze(1)).float().mean()  # Calculer l'exactitude
        self.log('val_loss', loss)
        self.log('val_accuracy', accuracy)


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['val_accuracy'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss)
        self.log('avg_val_accuracy', avg_accuracy)


