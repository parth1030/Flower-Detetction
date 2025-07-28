# learn about database
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

"""
minimum 90
maximum 8021
x_avg 1307.4662244483552
y_avg 1335.001353729525
"""


dataset = torchvision.datasets.ImageFolder(root="./4/house_plant_species")

print(dataset.class_to_idx)
print(dataset[0], type(dataset[0]))
print(dataset[0][0].size)

# maxim = 0
# minimum = float("inf")
# x_total = 0
# y_total = 0
#
# for image in dataset:
#     if max(image[0].size) > maxim:
#         maxim = max(image[0].size)
#     if min(image[0].size) < minimum:
#         minimum = min(image[0].size)
#     x_total += image[0].size[0]
#     y_total += image[0].size[1]
#
# print(f"minimum {minimum}")
# print(f"maximum {maxim}")
#
# print(f"x_avg {x_total/len(dataset)}")
# print(f"y_avg {y_total/len(dataset)}")

# data_x = [image[0].size[0] for image in dataset]
# data_y = [image[0].size[1] for image in dataset]
#
#
# fig, axes = plt.subplots(1, 2, figsize=(10, 4)) # 1 row, 2 columns
#
# # Plot histogram for data1 on the first subplot
# axes[0].hist(data_x, bins=30, color='skyblue')
# axes[0].set_xlabel('Value')
# axes[0].set_ylabel('Frequency')
# axes[0].set_title('Histogram of Dataset X')
#
# # Plot histogram for data2 on the second subplot
# axes[1].hist(data_y, bins=30, color='lightcoral')
# axes[1].set_xlabel('Value')
# axes[1].set_ylabel('Frequency')
# axes[1].set_title('Histogram of Dataset Y')
#
# plt.tight_layout() # Adjust layout to prevent overlapping titles/labels
# plt.show()

# transform = transforms.Compose([
#     transforms.Resize((5000, 5000)),
#     transforms.ToTensor()
# ])
#
# img = dataset[4][0]
#
# #plt.imshow(img)
#
# tensor = transform(img)
#
# image = tensor.permute(1, 2, 0)
#
#
# plt.imshow(image)
#
# plt.show()


# how much do I shrink my image by => help feature detection
