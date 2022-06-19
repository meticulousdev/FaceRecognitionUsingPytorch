import torch
import torchvision
from torchvision import transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# pytorch model to onnx
import torch.onnx

from typing import Sequence


# 각 레이어 구성하는 방법
# layer의 사이즈 및 forward 구성하는 방법
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 15, 5)

        self.fc1 = nn.Linear(15 * 22 * 22, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu((self.conv1(x))))
        x = self.pool(F.relu((self.conv2(x))))
        x = x.view(x.size(0), 15 * 22 * 22)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def model_training(sizeofimage: Sequence, num_batch: int, trainset_path: str, testset_path: str) -> None:
    transform = transforms.Compose([transforms.Resize(sizeofimage),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.ImageFolder(root=trainset_path, transform=transform)
    testset = torchvision.datasets.ImageFolder(root=testset_path, transform=transform)

    classes = trainset.classes
    num_classes: int = len(classes)
    print(f'class list: {classes}')

    # DataLoader
    # 배치 사이즈를 이용해서 test 및 train loader 설정
    print('training and test data load')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=num_batch, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=num_batch, shuffle=True, num_workers=2)

    # model training
    print('model Training')
    net = Net()

    # define a Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # train the network
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1000 == 999:    # print every 1000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0

    print('model training finished')

    PATH = './pytorch_face_recognition.pth'
    torch.save(net.state_dict(), PATH)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('accuracy of the network on the total testset images: %d %%' % (100 * correct / total))

    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(num_batch):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(num_classes):
        print('accuracy of %10s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

    print('export onnx')
    net.eval()

    dummy_input = torch.randn(1, 3, 100, 100)
    torch.onnx.export(net,
                      dummy_input,
                      "./onnx_model/pytorch_face_recognition.onnx",
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})

    print('end of file')


if __name__ == "__main__":
    sizeofimage = (100, 100)
    num_batch: int = 2
    trainset_path: str = './capture/train'
    testset_path: str = './capture/test'

    model_training(sizeofimage, num_batch, trainset_path, testset_path)
