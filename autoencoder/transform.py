import torchvision.transforms as T


class Cifar(object):
    def __init__(self):
        self.transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
