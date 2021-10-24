import torchvision.models as torchmodels
import torch
import numpy as np

def set_parameter_requires_grad(model, feature_extracting):
    # When loading the models, make sure to call this function to update the weights
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def get_model(num_output, _type='resnet34'):
    if _type == 'resnet34':
        agent = torchmodels.resnet34(pretrained=True)
    elif _type == 'resnet18':
        agent = torchmodels.resnet18(pretrained=True)
    else:
        raise Exception(f"model type '{_type}' not defined")
    # set_parameter_requires_grad(agent, False) # I don't understand what does this function do
    num_ftrs = agent.fc.in_features
    agent.fc = torch.nn.Linear(num_ftrs, num_output)
    return agent
