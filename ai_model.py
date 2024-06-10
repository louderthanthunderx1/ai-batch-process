import torch
import torchvision
from torch import nn
from torchvision.transforms import transforms
import timm

from ProductEncoder.network import Network

class Bot_Beer_Model(nn.Module):
    def __init__(self, num_classes):
        super(Bot_Beer_Model, self).__init__()
        # Load the pre-trained ResNet50 model with weights
        # weights = torchvision.models.ResNet50_Weights.DEFAULT  # .DEFAULT = best available weights
        # self.model = torchvision.models.resnet50(weights=weights) # เขียนโหลด offline model
        self.model = torchvision.models.resnet50(None)

        # Freeze all parameters in the model
        for param in self.model.parameters():
            param.requires_grad = False

        # Number of features in the bottleneck layer
        num_ftrs = self.model.fc.in_features

        # Replace the last fully connected layer
        self.model.fc = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.25),  # Additional dropout layer for regularization
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
                
        # unfreeze some layer
        for param in self.model.layer4.parameters():
            param.requires_grad = True
            
    def forward(self, x):
        return self.model(x)
    
class Can_Beer_Model(nn.Module):
    def __init__(self, num_classes):
        super(Can_Beer_Model, self).__init__()
        # Load the pre-trained ResNet50 model with weights
        # weights = torchvision.models.ResNet50_Weights.DEFAULT  # .DEFAULT = best available weights
        # self.model = torchvision.models.resnet50(weights=weights)
        self.model = torchvision.models.resnet50(None)
        

        # Freeze all parameters in the model
        for param in self.model.parameters():
            param.requires_grad = False

        # Number of features in the bottleneck layer
        num_ftrs = self.model.fc.in_features

        # Replace the last fully connected layer
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5, inplace=True),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
        # unfreeze some layer
        for param in self.model.layer4.parameters():
            param.requires_grad = True
            
    def forward(self, x):
        return self.model(x)
    
    
class Bot_CSD_Model(nn.Module):
    def __init__(self, num_classes):
        super(Bot_CSD_Model, self).__init__()
        # Load the pre-trained ResNet50 model with weights
        # weights = torchvision.models.ResNet50_Weights.DEFAULT  # .DEFAULT = best available weights
        # self.model = torchvision.models.resnet50(weights=weights)
        self.model = torchvision.models.resnet50(None)
        
        # Freeze all parameters in the model
        for param in self.model.parameters():
            param.requires_grad = False

        # unfreeze some layer
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Number of features in the bottleneck layer
        num_ftrs = self.model.fc.in_features
        
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),  # First layer with 512 neurons
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),  # Second layer with 256 neurons
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),  # Third layer with 128 neurons
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),  # Fourth layer with 64 neurons
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)  # Output layer for classification
        )

    def forward(self, x):
        return self.model(x)
    
class Bot_Tea_Model(nn.Module):
    def __init__(self, num_classes):
        super(Bot_Tea_Model, self).__init__()
        # Load the pre-trained ResNet50 model with weights
        # weights = torchvision.models.ResNet50_Weights.DEFAULT  # .DEFAULT = best available weights
        # self.model = torchvision.models.resnet50(weights=weights)
        self.model = torchvision.models.resnet50(None)

        # Freeze all parameters in the model
        for param in self.model.parameters():
            param.requires_grad = False

        # unfreeze some layer
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Number of features in the bottleneck layer
        num_ftrs = self.model.fc.in_features

        # Fully Layers
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),  # First layer with 512 neurons
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),  # Second layer with 256 neurons
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),  # Third layer with 128 neurons
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),  # Fourth layer with 64 neurons
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)  # Output layer for classification
        )

    def forward(self, x):
        return self.model(x)
    
class Bot_Water_Model(nn.Module):
    def __init__(self, num_classes):
        super(Bot_Water_Model, self).__init__()
        # Load the pre-trained ResNet50 model with weights
        # weights = torchvision.models.ResNet50_Weights.DEFAULT  # .DEFAULT = best available weights
        # self.model = torchvision.models.resnet50(weights=weights)
        self.model = torchvision.models.resnet50(None)
        
        # Freeze all parameters in the model
        for param in self.model.parameters():
            param.requires_grad = False

        # unfreeze some layer
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Number of features in the bottleneck layer
        num_ftrs = self.model.fc.in_features
        
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),  # First layer with 512 neurons
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),  # Second layer with 256 neurons
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),  # Third layer with 128 neurons
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),  # Fourth layer with 64 neurons
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)  # Output layer for classification
        )

    def forward(self, x):
        return self.model(x)
    

    
def load_bot_beer_model(model_path, device):
    model = Bot_Beer_Model(num_classes=2)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

def load_can_beer_model(model_path, device):
    model = Can_Beer_Model(num_classes=2)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

def load_bot_csd_model(model_path, device):
    model = Bot_CSD_Model(num_classes=9)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

def load_bot_tea_model(model_path, device):
    model = Bot_Tea_Model(num_classes=4)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

def load_bot_water_model(model_path, device):
    model = Bot_Water_Model(num_classes=3)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

# ฟังก์ชั่นสำหรับโหลด encoder
def get_resnet_encoder(cfg):
    """
    Load pretrained resnet for encoding.
    ต้องใส่ cfg เข้ามาเป็น config ของ model
    """
    # สร้างตัวแปร model
    model = Network(cfg).cuda()

    # ปิดพวกค่าต่างๆที่เอาไว้สำหรับตอน train
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # โหลด Model จาก savepoint
    checkpoint = torch.load(cfg.RESUME_MODEL, map_location='cuda')
    model.load_model(cfg.RESUME_MODEL)

    return model


def get_transfer_model(device, encoder_model_path, cfg):
    # model = torch.load(encoder_model_path, map_location='cuda')

    # ต้อง define ตัว model ให้เหมือนกับตอนที่เราทำ finetune มา ต้อง save แบบ state_dict เท่านั้น !
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    # จำนวน class_num ต้องเท่ากับจำนวนคลาสที่ train ตอนทำ finetune
    class_num = 124
    model = Network(cfg).cuda()
    number_features = model.classifier.in_features
    features = list(model.classifier.children())[:-1]  # Remove last layer
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(2048),
        nn.Dropout(p=0.25),
        nn.Linear(in_features=2048, out_features=2048),
        nn.ReLU(),
        nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1),
        nn.Dropout(p=0.25),
        nn.Linear(number_features, class_num),
    )
    model = model.to(device)

    # model = torch.load(encoder_model_path).cuda()
    model.load_state_dict(torch.load(encoder_model_path))
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    return model


# resnet transform image
def get_resnet_trasforms(image, device):

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        # transforms.Resize(256), # new
        transforms.CenterCrop(224),  # new
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    
    return image
