import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim


class PutNet(nn.Module):
    """
    Example of a Neural Network that could be trained price a put option.
    TODO: modify me!
    """

    def __init__(self) -> None:
        super(PutNet, self).__init__()

        self.l1 = nn.Linear(5, 20)
        self.l2 = nn.Linear(20, 20)
        self.l3 = nn.Linear(20, 20)
        self.l4 = nn.Linear(20, 20)
        self.l5 = nn.Linear(20, 20)
        self.out = nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        x = self.out(x)
        return x


def main():
    """Train the model and save the checkpoint"""

    # Create model
    model = PutNet()

    # Load dataset
    df = pd.read_csv("bs-put-1k.csv")

    # Set up training
    x = torch.Tensor(df[["S", "K", "T", "r", "sigma"]].to_numpy())
    y = torch.Tensor(df[["value"]].to_numpy())
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.97)
    # Train for 500 epochs
    for i in range(100000):

        # TODO: Modify to account for dataset size
        xtemp = torch.split(x,[700,300])
        y_hat = model(xtemp[0])
        ytemp=torch.split(y,[700,300])
        ytrain=ytemp[0]
        yver = ytemp[1]
        # Calculate training loss
        training_loss = criterion(y_hat[0], ytrain)

        # Take a step
        optimizer.zero_grad()
        training_loss.backward()
        optimizer.step()

        # Check validation loss
        with torch.no_grad():
            # TODO: use a proper validation set
            validation_loss = criterion(model(xtemp[1]), yver)

        print(f"Iteration: {i} | Training Loss: {training_loss:.4f} | Validation Loss: {validation_loss:.4f} ")
    torch.save(model.state_dict(), "simple-model.pt")
    print("Done!")

if __name__ == "__main__":
    main()
