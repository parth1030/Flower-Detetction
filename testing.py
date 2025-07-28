import torch
import torchvision
import torchvision.transforms as transforms
from main import CNN
import matplotlib
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
import matplotlib.pyplot as plt

# transform = transforms.Compose([
#         transforms.Resize(512),
#         transforms.CenterCrop(512),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
#     ])
#
# device = torch.device("mps")
#
# dataset = torchvision.datasets.ImageFolder(root="./4/initial_data", transform=transform)
#
# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size
#
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
#
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
# test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)
#
# model = CNN()
# model.load_state_dict(torch.load("model.pth", map_location=device))
# model.to(device)
# model.eval()
#
#
# res = next(iter(test_loader))
# print(res)
#
# images = res[0]
#
# image = images[0]
#
# batch_image = image.unsqueeze(0).to(device)
#
# with torch.no_grad():
#     log_probabilities = model(batch_image)
#
# plt.imshow(image.permute(1,2,0)/2 + 0.5)
# print(torch.exp(log_probabilities).squeeze().cpu())
# plt.show()

from PIL import Image

transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

device = torch.device("cpu")

model = CNN()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)

train_nodes, eval_nodes = get_graph_node_names(model)
print(train_nodes)
print()
print(eval_nodes)

return_nodes = {
    'relu': 'relu_0',
    'relu_1': 'relu_1',
    'relu_2': 'relu_2',
    'relu_3': 'relu_3',
    'relu_4': 'relu_4',
    'relu_5': 'relu_5'
}

feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)

Anthurium_6 = transform(Image.open('./4/initial_data/Anthurium (Anthurium andraeanum)/6.jpg'))
Anthurium_300 = transform(Image.open('./4/initial_data/Anthurium (Anthurium andraeanum)/300.jpg'))

AfViolet_67 = transform(Image.open('./4/initial_data/African Violet (Saintpaulia ionantha)/67.jpg'))
AfViolet_276 = transform(Image.open('./4/initial_data/African Violet (Saintpaulia ionantha)/276.jpg'))

images = [Anthurium_6, Anthurium_300, AfViolet_67, AfViolet_276]
fig, axs = plt.subplots(6, 4)



for i in range(6):
    k=0
    for j in range(4):
        extracted_features = feature_extractor(images[k])
        k += 1

        feature_map_array = extracted_features[f'relu_{i}'][0, :, :].detach().numpy()
        axs[i][j].imshow(feature_map_array, cmap='viridis') # Choose a colormap


plt.axis('off') # Optional: turn off axis for cleaner display
plt.show()
