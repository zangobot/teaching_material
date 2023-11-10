# ----- Standard Imports
import os

# ----- Third Party Imports
import torch as tr
from torchvision import models
from tqdm import tqdm

# ----- Library Imports
from utils import evaluate_accuracy


PRETRAINED_MODELS = {
    'alexnet': models.alexnet,
}

class TorchModelWrapper:
    def __init__(self,
                 model_name: str,
                 num_classes: int,
                 ):
        self.model_name = model_name

        # download imagenet-classification pretrained weights
        try:
            self.model = PRETRAINED_MODELS[self.model_name](weights='IMAGENET1K_V1')
        except KeyError:
            raise ValueError(f'Model name {self.model_name} is not included yet')
        
        # adapt final FC layer to the classification task at hand
        self.model.classifier[-1] = tr.nn.Linear(self.model.classifier[-1].in_features, num_classes)

        # freeze feature extraction layers
        for param in self.model.features.parameters():
            param.requires_grad = False
        
    def __call__(self, x):
        return self.model(x)
    
    def train_model(self,
                    train_loader,
                    valid_loader,
                    test_loader,
                    num_epochs: int = 50,
                    device_idx: int = -1,
                    verbose: bool = False,
                    ):
        # optionally move model to GPU
        trn_device = tr.device(f'cuda:{device_idx}' if tr.cuda.is_available() and device_idx != -1 else 'cpu')
        self.model = self.model.to(trn_device)
        
        # define loss function and optimizer
        loss_fn = tr.nn.CrossEntropyLoss(reduction='mean')
        optimizer = tr.optim.Adam(params=self.model.parameters(), lr=1e-3)

        if verbose:
            with tr.no_grad():
                print("Accuracies before Training:")
                trn_acc = evaluate_accuracy(self.model, train_loader)
                print(f"\tAccuracy on train set: {trn_acc:.2f}")
                val_acc = evaluate_accuracy(self.model, valid_loader)
                print(f"\tAccuracy on validation set: {val_acc:.2f}")
                tst_acc = evaluate_accuracy(self.model, test_loader)
                print(f"\tAccuracy on test set: {tst_acc:.2f}")

        self.model.train()

        # for each epoch
        for ep in range(1,num_epochs+1):
            cum_loss = 0
            # for each batch
            batch = tqdm(train_loader) if verbose else train_loader
            for b_idx, (x,y) in enumerate(batch):
                # reset gradients
                optimizer.zero_grad()
                # optionally move to gpu
                x = x.to(trn_device)
                y = y.to(trn_device)
                # compute predictions
                out = self.model(x)
                # compute loss
                loss = loss_fn(out, y)
                # compute gradients
                loss.backward()
                # update model parameters
                optimizer.step()
                if verbose:
                    cum_loss += loss.item()
                    batch.set_description(f"@epoch {ep}/{num_epochs} average train loss: {cum_loss/(b_idx+1):.3f}")

            if verbose:
                with tr.no_grad():
                    # test on train set
                    trn_acc = evaluate_accuracy(self.model, train_loader)
                    print(f"\tAccuracy on train set: {trn_acc:.2f}")
                    # test on validation set
                    val_acc = evaluate_accuracy(self.model, valid_loader)
                    print(f"\tAccuracy on validation set: {val_acc:.2f}")
        
        self.model.eval()

        if verbose:
            with tr.no_grad():
                # test on test set
                print("\nTraining ended")
                tst_acc = evaluate_accuracy(self.model, test_loader)
                print(f"\tAccuracy on test set: {tst_acc:.2f}")
        
        return self.model
    
    def store_model(self, path: str):
        tr.save(self.model.to('cpu').state_dict(),
                os.path.join(path, f'{self.model_name}.pth'))
        
    def load_model(self, path: str):
        self.model.load_state_dict(tr.load(os.path.join(path, f'{self.model_name}.pth')))

    def set_model_gradients(self, value: bool):
        for param in self.model.parameters():
            param.requires_grad = value
