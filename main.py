import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from fsspec.asyn import running_async
from fsspec.implementations.local import trailing_sep
from torchsummary import summary
import matplotlib


matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
import matplotlib.pyplot as plt


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.c1 = nn.Conv2d(3, 8, 3, padding=1)
        self.c2 = nn.Conv2d(8, 16, 3,padding=1)
        self.c3 = nn.Conv2d(16, 32, 3,padding=1)
        self.c4 = nn.Conv2d(32, 64, 3, padding = 1)
        self.c5 = nn.Conv2d(64, 128, 3, padding = 1)
        self.c6 = nn.Conv2d(128, 256, 3, padding = 1)


        self.pool = nn.MaxPool2d(2, stride = 2)

        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 3)

    def forward(self, x):
        # print("doing shit")
        x = F.relu(self.c1(x)) # 512 8
        x = self.pool(x) # 256 8
        x = F.relu(self.c2(x)) # 256 16
        x = self.pool(x) # 128 16
        x = F.relu(self.c3(x)) # 128 32
        x = self.pool(x)# 64 32
        x = F.relu(self.c4(x))# 64 64
        x = self.pool(x)# 32 64
        x = F.relu(self.c5(x))  # 32 128
        x = self.pool(x) # 16 128
        x = F.relu(self.c6(x))  # 16 256
        x = self.pool(x)  # 8 256
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim = 1)
        return x

if __name__ == "__main__":

    #path = "/Users/parthsharma/.cache/kagglehub/datasets/kacpergregorowicz/house-plant-species/versions/4"

    device = torch.device("mps")
    print(device)

    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    dataset = torchvision.datasets.ImageFolder(root="./4/initial_data", transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)



    classes = []

    # fig, axes = plt.subplots(1, 10, figsize=(12, 3))
    # for i in range(5):
    #     image = train_loader.dataset[i][0].permute(1, 2, 0)
    #     denormalized_image = image / 2 + 0.5
    #     axes[i].imshow(denormalized_image)
    #     axes[i].axis("off")
    #
    # plt.show()
    import time

    net = CNN()
    net.to(device)

    loss_func = nn.NLLLoss()
    optimizer = optim.Adam(net.parameters(),lr=0.001)

    for epoch in range(5):
        start = time.time()
        running_loss = 0.0
        i = 0
        for data in train_loader:
            # print(data)
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_func(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 10 == 9:
                net.eval()
                val_loss = 0.0
                correct = 0
                total = 0

                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = net(inputs)
                        loss = loss_func(outputs, labels)
                        val_loss += loss.item()
                        _, preds = torch.max(outputs, 1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)

                print(f"Validation loss: {val_loss / len(test_loader):.4f}")
                print(f"Validation accuracy: {correct / total:.4%}")

                net.train()

                print(f"{epoch+1}/{10} and loss is {running_loss/100:.3f} and took {(time.time()-start):.4f} seconds")
                start = time.time()
                running_loss = 0
            i+=1

    torch.save(net.state_dict(), "/Users/parthsharma/PycharmProjects/Flower-Detetction/model.pth")
    print("✅ Model saved as model.pth")
    print("done")

    # net.eval()
    #
    #
    # images, _ = next(iter(test_loader))
    #
    # image = images[0]
    #
    # batch_image = image.unsqueeze(0).to(device)
    #
    # with torch.no_grad():
    #     log_probabilities = net(batch_image)
    #
    # plt.imshow(image.permute(1,2,0)/2 + 0.5)
    # print(torch.exp(log_probabilities).squeeze().cpu())












