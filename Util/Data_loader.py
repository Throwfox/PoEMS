from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Set up the dataloader
class Sepsis_onset(Dataset):
    def __init__(self, x_train, y_train):
        self.input = x_train
        self.target = y_train

    def __getitem__(self, index):
        return self.input[index], self.target[index]

    def __len__(self):
        return self.input.shape[0]
def create_dataloader(x_train,y_train,x_test,y_test):
    train_dataset = Sepsis_onset(x_train,y_train)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=128,
                              num_workers=0,
                              shuffle=True)   
    test_dataset = Sepsis_onset(x_test,y_test)
    test_loader = DataLoader(dataset=test_dataset,
                              batch_size=128,
                              num_workers=0,
                              shuffle=True)                             
    loaders = {'train':train_loader,'validation':test_loader}
    return loaders