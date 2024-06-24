    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.utils
    import torch.utils.data
    import torchvision
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt


    def import_MNIST_dataset(batch_size=64,test=True):
        """
        Downloads the MNIST dataset and loads it into DataLoader objects for training and testing.

        The MNIST dataset consists of 60,000 training images and 10,000 testing images of handwritten digits.
        The images are normalized to have pixel values between -1 and 1.

        :return: A tuple containing the training DataLoader and the testing DataLoader.
        """
        # Define a transform to normalize the data
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        # Download and load the training dataset
        trainset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        if not test:
            return train_loader

        # Download and load the testing dataset
        testset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader



    def import_MNIST_examples(mnist:torch.utils.data.DataLoader):
        re = torch.empty(10,28,28)
        for i in range(10):
            run_ind = 0
            while(mnist.dataset[run_ind][1]!=i):
                run_ind+=1
            re[i]=mnist.dataset[run_ind][0]
        return re.unsqueeze(1)
        

