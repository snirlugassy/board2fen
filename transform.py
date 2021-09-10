from torchvision import transforms

transformations = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.RandomCrop((250,250)),
    transforms.ToTensor()
    # transforms.Normalize((0.4, 0.4, 0.4), (0.1, 0.1, 0.1))
])
