import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def train_nofreeze(model, dataset, optimizer, scheduler, criterion, epochs, batch_size, save_dir, efficiency, device, label_level=None):

    best_train_loss = float('inf')
    best_keep = 0
    patience = 5
    checkpoint = {'model': model.state_dict()}
    best_epoch = -1

    if efficiency*100 >= 1:
        log_dir = os.path.join(save_dir, f'log{int(efficiency*100)}')
    else:
        log_dir = os.path.join(save_dir, f'log{str(efficiency*100)}')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=False, pin_memory=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=False)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        model.train()
        train_loss = 0
        for i, (img_names, imgs, labels) in enumerate(dataloader):
            imgs = imgs.to(device)
            if label_level is not None:
                labels = torch.tensor(labels[label_level]).to(device)
            else:
                labels = torch.tensor(labels).to(device)

            optimizer.zero_grad()
            output = model(imgs)

            loss = criterion(output, labels)
            writer.add_scalar('train_batch_loss', loss.item(), epoch * len(dataloader) + i)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            scheduler.step()

        train_loss /= len(dataloader)
        print(f"------------train loss: {train_loss}")

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_keep = 0

            checkpoint['model'] = model.state_dict()
            best_epoch = epoch
        else:
            best_keep += 1
            if best_keep >= patience:
                print('save final best model weight! ! !')
                if efficiency * 100 >= 1:
                    save_path = os.path.join(save_dir, f'checkpoint{int(efficiency * 100)}', f"epoch{best_epoch}_model.pth")
                    os.makedirs(os.path.join(save_dir, f'checkpoint{int(efficiency * 100)}'), exist_ok=True)
                else:
                    save_path = os.path.join(save_dir, f'checkpoint{str(efficiency * 100)}', f"epoch{best_epoch}_model.pth")
                    os.makedirs(os.path.join(save_dir, f'checkpoint{str(efficiency * 100)}'), exist_ok=True)
                torch.save(checkpoint, save_path)
                print('---------------------early stopping !-------------------')
                return

    print('save final best model weight! ! !')
    if efficiency * 100 >= 1:
        save_path = os.path.join(save_dir, f'checkpoint{int(efficiency * 100)}', f"epoch{best_epoch}_model.pth")
        os.makedirs(os.path.join(save_dir, f'checkpoint{int(efficiency * 100)}'), exist_ok=True)
    else:
        save_path = os.path.join(save_dir, f'checkpoint{str(efficiency * 100)}', f"epoch{best_epoch}_model.pth")
        os.makedirs(os.path.join(save_dir, f'checkpoint{str(efficiency * 100)}'), exist_ok=True)
    torch.save(checkpoint, save_path)


def train_freeze(model, dataset, optimizer, scheduler, criterion, feature_path, epochs, batch_size, save_dir, efficiency, device, label_level=None):

    best_train_loss = float('inf')
    best_keep = 0
    patience = 100
    checkpoint = {'model': model.state_dict()}
    best_epoch = -1

    if efficiency * 100 >= 1:
        log_dir = os.path.join(save_dir, f'log{int(efficiency * 100)}')
    else:
        log_dir = os.path.join(save_dir, f'log{str(efficiency * 100)}')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=False)

    if os.path.exists(feature_path):
        print('loading freeze backbone feature ! ! !')
        features_dict = torch.load(feature_path)
    else:
        print('recording freeze backbone feature ! ! !')
        features_dict = dict()
        model.eval()
        with torch.no_grad():
            for i, (img_names, imgs, labels) in enumerate(dataloader):
                imgs = imgs.to(device)
                features = model.forward_backbone(imgs).cpu()

                for i, img_name in enumerate(img_names):
                    features_dict[img_name] = features[i]

        os.makedirs(os.path.dirname(feature_path), exist_ok=True)
        torch.save(features_dict, feature_path)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        model.train()
        train_loss = 0
        for i, (img_names, imgs, labels) in enumerate(dataloader):
            img_features = torch.stack([features_dict[img_name] for img_name in img_names])
            img_features = img_features.to(device)

            if label_level is not None:
                labels = torch.tensor(labels[label_level]).to(device)
            else:
                labels = torch.tensor(labels).to(device)

            optimizer.zero_grad()
            output = model.forward_classifier(img_features)

            loss = criterion(output, labels)
            writer.add_scalar('train_batch_loss', loss.item(), epoch * len(dataloader) + i)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            scheduler.step()

        train_loss /= len(dataloader)
        print(f"------------train loss: {train_loss}")

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_keep = 0

            checkpoint['model'] = model.state_dict()
            best_epoch = epoch
        else:
            best_keep += 1
            if best_keep >= patience:
                print('save final best model weight! ! !')
                if efficiency * 100 >= 1:
                    save_path = os.path.join(save_dir, f'checkpoint{int(efficiency * 100)}', f"epoch{best_epoch}_model.pth")
                    os.makedirs(os.path.join(save_dir, f'checkpoint{int(efficiency * 100)}'), exist_ok=True)
                else:
                    save_path = os.path.join(save_dir, f'checkpoint{str(efficiency * 100)}', f"epoch{best_epoch}_model.pth")
                    os.makedirs(os.path.join(save_dir, f'checkpoint{str(efficiency * 100)}'), exist_ok=True)
                torch.save(checkpoint, save_path)
                print('---------------------early stopping !-------------------')
                return

    print('save final best model weight! ! !')
    if efficiency * 100 >= 1:
        save_path = os.path.join(save_dir, f'checkpoint{int(efficiency * 100)}', f"epoch{best_epoch}_model.pth")
        os.makedirs(os.path.join(save_dir, f'checkpoint{int(efficiency * 100)}'), exist_ok=True)
    else:
        save_path = os.path.join(save_dir, f'checkpoint{str(efficiency * 100)}', f"epoch{best_epoch}_model.pth")
        os.makedirs(os.path.join(save_dir, f'checkpoint{str(efficiency * 100)}'), exist_ok=True)
    torch.save(checkpoint, save_path)