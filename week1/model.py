
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision
import torchsummary


class ResidualBlock(nn.Module):

    def __init__(self, input=64, output=64, stride=1):
        super(ResidualBlock, self).__init__()

        self.input = input
        self.output = output
        self.stride = stride

        self.layer = nn.Sequential(
            nn.Conv2d(input, output, kernel_size=1, bias=False),
            nn.BatchNorm2d(output),
            nn.ReLU(),
            nn.Conv2d(output, output, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(output),
            nn.ReLU(),
            nn.Conv2d(output, output * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(output * 4))

        self.projection = nn.Sequential(
            nn.Conv2d(input, output * 4, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(output * 4))

    def forward(self, x):
        identity = x
        x = self.layer(x)
        if self.input != self.output * 4:
            identity = self.projection(identity)
        x += identity
        return x

class Resnet(nn.Module):

    def __init__(self, num_block, input, num_classes):
        super(Resnet, self).__init__()

        self.input = input
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv2 = self._make_layer(64, 64, num_block[0], 1)
        self.conv3 = self._make_layer(256, 128, num_block[1], 2)
        self.conv4 = self._make_layer(512, 256, num_block[2], 2)
        self.conv5 = self._make_layer(1024, 512, num_block[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # Conv층 초기화
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # kaming_normal 분포로 가중치 초기화
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # init_constant_상수로 초기화
            elif isinstance(m, nn.BatchNorm2d):  # BatchNorm 초기화
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):  # Linear층 초기화
                nn.init.normal_(m.weight, 0, 0.01)  # 평균이 0이고, 표준편차가 0.01 분포로 설
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, input_ch, output, num, stride):
        layers = []
            # 첫 번째 ResidualBlock 생성 시 입력 채널 수와 출력 채널 수를 설정합니다.
        layers.append(ResidualBlock(input_ch, output, stride))
        input_ch = output * 4  # 다음 레이어의 입력 채널 수를 설정하기 위해 업데이트합니다.
        for _ in range(num - 1):
                # 나머지 ResidualBlock들을 생성할 때도 입력 채널 수와 출력 채널 수를 설정합니다.
            layers.append(ResidualBlock(input_ch, output))
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


def ResnetN(x,y,z):
    return Resnet(x,y,z)

if __name__ == '__main__':
    model=ResnetN([3,4,6,3],3,100)
    model.to(device='cuda')
    torchsummary.summary(model, input_size=(3, 32, 32), device='cuda')
