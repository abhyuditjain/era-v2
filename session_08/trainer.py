from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Trainer:
    def __init__(self, model, train_loader, optimizer, criterion, device) -> None:
        self.losses = []
        self.accuracies = []
        self.epoch_accuracies = []
        self.model = model.to(device)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.lr_history = []

    def train(self, epoch, use_l1=False, lambda_l1=0.01):
        self.model.train()

        lr_trend = []
        correct = 0
        processed = 0
        train_loss = 0

        pbar = tqdm(self.train_loader)

        for batch_id, (inputs, targets) in enumerate(pbar):
            # transfer to device
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Initialize gradients to 0
            self.optimizer.zero_grad()

            # Prediction
            outputs = self.model(inputs)

            # Calculate loss
            loss = self.criterion(outputs, targets)

            l1 = 0
            if use_l1:
                for p in self.model.parameters():
                    l1 = l1 + p.abs().sum()
            loss = loss + lambda_l1 * l1

            self.losses.append(loss.item())

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(targets.view_as(pred)).sum().item()
            processed += len(inputs)

            pbar.set_description(
                desc=f"EPOCH = {epoch} | LR = {self.optimizer.param_groups[0]['lr']} | Loss = {loss.item():3.2f} | Batch = {batch_id} | Accuracy = {100*correct/processed:0.2f}"
            )
            self.accuracies.append(100 * correct / processed)

        # After all the batches are done, append accuracy for epoch
        self.epoch_accuracies.append(100 * correct / processed)

        self.lr_history.extend(lr_trend)
        return 100 * correct / processed, train_loss / len(self.train_loader), lr_trend
