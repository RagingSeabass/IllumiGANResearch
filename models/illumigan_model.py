from models.base_model import BaseModel
from models.nets import GeneratorUNetV1, MiniModel
from models.utils import init_network
from . import utils
import torch
import torch.optim as optim


class IllumiganModel(BaseModel):
    def __init__(self, manager):
        super().__init__(manager)

        self.loss_names = ['G_L1']
        
        self.model_names = ['G']

        norm = 'instance' if manager.get_hyperparams().get('batch_size') == 1 else 'batch'
        
        self.G_net = init_network(GeneratorUNetV1(norm_layer=norm, use_dropout=False), gpu_ids=self.gpus)
        
        if manager.is_train:
            self.criterionL1 = torch.nn.L1Loss()
            self.G_optimizer = torch.optim.Adam(self.G_net.parameters(), lr=manager.get_hyperparams().get('lr'), betas=(0.5, 0.999))
            self.optimizers.append(self.G_optimizer)

            self.schedulers = [utils.get_learning_rate_scheduler(optimizer, manager.get_hyperparams()) for optimizer in self.optimizers]

    
    def set_input(self, x, y):
        """Takes input of form X Y"""

        x = x.permute(0,3,1,2).to(self.device)
        y = y.permute(0,3,1,2).to(self.device)

        self.x = x
        self.y = y
    
    def forward(self):
        self.fake_y = self.G_net(self.x)  # G(X) = fake_y

    def g_backward(self):
        self.loss_G_L1 = self.criterionL1(self.fake_y, self.y)
        self.loss_G_L1.backward()

    def get_loss(self):
        return self.loss_G_L1.item()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights"""
        # Calc G(x)
        self.forward()
        
        # set G's gradients to zero
        self.G_optimizer.zero_grad()        

        # back propagate
        self.g_backward()

        # Update weights
        self.G_optimizer.step()        
    

        




