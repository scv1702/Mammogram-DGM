import torch
from torchvision import transforms
from utils import AttrDict

config = AttrDict()

# Metadata path
config.train_metadata_path = './data/MG_train_path.csv'
config.train_img_dir = './data/MG_train'
config.normal_metadata_path = './data/MG_normal_path.csv'
config.normal_img_save_path = './save/normal'
config.residue_map_save_path = './save/residue_map'
config.r1_save_path = './save/reconstructed_1'
config.r2_save_path = './save/reconstructed_2'
config.graph_path = './save/graph'

# Hyperparameters
config.save_interval = 1
config.log_interval = 10

config.n_epoch = 30
config.learning_rate = 0.00001
config.batch_size = 1

# Adam Momentum Hyperparameters
config.b1 = 0.5
config.b2 = 0.999

# Loss Function Hyperparameters
config.lambda_A = 10
config.lambda_R = 10
config.lambda_TV = 1

config.img_shape = (1, 224, 224)
config.augmentation = transforms.Compose([
    transforms.Resize((config.img_shape[1], config.img_shape[2])),
    transforms.PILToTensor(),
])
config.denormalize = lambda x: (x + 1) * 127.5
config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')