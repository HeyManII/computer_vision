import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import time

from torch.utils.data import DataLoader
from dataset import load_mnist, load_cifar10, load_fashion_mnist, imshow
from model import LeNet5

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("Device: CUDA")
else:
    device = torch.device('cpu')
    print("Device: CPU")

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

time1 = time.perf_counter()
trainset, testset = load_mnist()
trainloader = DataLoader(trainset, batch_size=8, shuffle=True)
testloader = DataLoader(testset, batch_size=8, shuffle=False)

model1 = LeNet5.to(device)
model1.apply(weights_init)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model1.parameters(), lr=0.001, momentum=0.9)

# loop over the dataset multiple times
num_epoch = 1
model1.train()
for epoch in range(num_epoch):
    running_loss = 0.0
    for i, batch in enumerate(trainloader, 0):
        # get the images; batch is a list of [images, labels]
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()  # zero the parameter gradients

        # get prediction
        outputs = model1(images)
        # compute loss
        loss = loss_fn(outputs, labels)
        # reduce loss
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 500 == 499:  # print every 500 mini-batches
            print('[epoch %2d, batch %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 500))
            running_loss = 0.0

# model_file = 'model.pth'
# torch.save(model.state_dict(), model_file)
# print(f'Model saved to {model_file}.')

print('Finished Training')


# show some prediction result
dataiter = iter(testloader)
# images, labels = dataiter.next()
images, labels = next(dataiter)
images = images.to(device)
labels = labels.to(device)
predictions = model1(images).argmax(1).detach()

classes = trainset.classes
print('GroundTruth: ', ' '.join('%5s' % classes[i] for i in labels))
print('Prediction: ', ' '.join('%5s' % classes[i] for i in predictions))
# imshow(torchvision.utils.make_grid(images.cpu()))


# test
@torch.no_grad()
def accuracy(model, data_loader):
    model.eval()
    correct, total = 0, 0
    for batch in data_loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return correct / total

# train_acc = accuracy(model, trainloader)
SGD_test_acc = accuracy(model1, testloader)
time2 = time.perf_counter()
SGD_time = round(time2 - time1, 5)

# print('Accuracy on the train set: %f %%' % (100 * train_acc))
print('Accuracy of SGD on the test set: %f %%' % (100 * SGD_test_acc))


# -------------------------------------------------------------------------------------------------------------------------------

time1 = time.perf_counter()
trainset2, testset2 = load_mnist()
trainloader2 = DataLoader(trainset2, batch_size=8, shuffle=True)
testloader2 = DataLoader(testset2, batch_size=8, shuffle=False)

model2 = LeNet5.to(device)
model2.apply(weights_init)
loss_fn = nn.CrossEntropyLoss()
optimizer2 = optim.Adam(model2.parameters(), lr=0.001)

# loop over the dataset multiple times
num_epoch = 1
model2.train()
for epoch in range(num_epoch):
    running_loss = 0.0
    for i, batch in enumerate(trainloader2, 0):
        # get the images; batch is a list of [images, labels]
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        optimizer2.zero_grad()  # zero the parameter gradients

        # get prediction
        outputs = model2(images)
        # compute loss
        loss = loss_fn(outputs, labels)
        # reduce loss
        loss.backward()
        optimizer2.step()

        # print statistics
        running_loss += loss.item()
        if i % 500 == 499:  # print every 500 mini-batches
            print('[epoch %2d, batch %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 500))
            running_loss = 0.0

# model_file = 'model.pth'
# torch.save(model.state_dict(), model_file)
# print(f'Model saved to {model_file}.')

print('Finished Training')


# show some prediction result
dataiter2 = iter(testloader2)
# images, labels = dataiter.next()
images, labels = next(dataiter2)
images = images.to(device)
labels = labels.to(device)
predictions = model2(images).argmax(1).detach()

classes = trainset.classes
print('GroundTruth: ', ' '.join('%5s' % classes[i] for i in labels))
print('Prediction: ', ' '.join('%5s' % classes[i] for i in predictions))
# imshow(torchvision.utils.make_grid(images.cpu()))


# test
@torch.no_grad()
def accuracy(model, data_loader):
    model.eval()
    correct, total = 0, 0
    for batch in data_loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return correct / total

# train_acc = accuracy(model, trainloader)
adam_test_acc = accuracy(model2, testloader2)
time2 = time.perf_counter()
adam_time = round(time2 - time1, 5)

# print('Accuracy on the train set: %f %%' % (100 * train_acc))
print('Accuracy of adam on the test set: %f %%' % (100 * SGD_test_acc))


labels = ['SGD', 'Adam']
accuracies = [SGD_test_acc, adam_test_acc]
time_consumption = [SGD_time, adam_time]

plt.bar(labels, accuracies)
plt.xlabel('Optimizer')
plt.ylabel('Accuracy')
plt.title('Comparison of Adam and SGD Accuracy')
plt.show()


plt.bar(labels, accuracies)
plt.xlabel('Optimizer')
plt.ylabel('Accuracy')
plt.title('Comparison of Adam and SGD time consumption')
plt.show()
