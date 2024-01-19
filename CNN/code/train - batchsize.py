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

accuracy_list = []
time_list = []
B = [10, 30, 50, 70, 90, 110, 130, 150, 170, 190]
for b in B:
    time1 = time.perf_counter()
    trainset, testset = load_mnist()
    trainloader = DataLoader(trainset, batch_size=b, shuffle=True)
    testloader = DataLoader(testset, batch_size=8, shuffle=False)

    model = LeNet5.to(device)
    model.apply(weights_init)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # loop over the dataset multiple times
    num_epoch = 1
    model.train()
    for epoch in range(num_epoch):
        running_loss = 0.0
        for i, batch in enumerate(trainloader, 0):
            # get the images; batch is a list of [images, labels]
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()  # zero the parameter gradients

            # get prediction
            outputs = model(images)
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
    predictions = model(images).argmax(1).detach()

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
    test_acc = accuracy(model, testloader)
    accuracy_list.append(100 * test_acc)
    time2 = time.perf_counter()
    time_list.append(round(time2 - time1, 5))

    # print('Accuracy on the train set: %f %%' % (100 * train_acc))
    print('Accuracy on the test set: %f %%' % (100 * test_acc))



plt.plot(B, accuracy_list)
plt.xlabel('batch size')
plt.xticks(B)
plt.ylabel('Accuracy')
plt.title('Accuracy vs. batch size')
plt.show()

plt.plot(B, time_list)
plt.xlabel('batch size')
plt.xticks(B)
plt.ylabel('time')
plt.title('time vs. batch size')
plt.show()