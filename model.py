import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
       
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(embed_size, 
                            hidden_size, 
                            num_layers, 
                            batch_first=True,
                            dropout = 0.2)
        
        
        self.linear = nn.Linear(hidden_size, vocab_size)
        
          
    def forward(self, features, captions):
       
        captions = captions[:, :-1]
        captions = self.embed(captions)
        
        inputs = torch.cat((features.unsqueeze(1), captions), 1)
        lstm_output, _ = self.lstm(inputs)
       
        outputs = self.linear(lstm_output)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        sentence =  []
        word_item = None
        
        for i in range(max_len):
            
            if(word_item == 1): break
            outputs_lstm, states = self.lstm(inputs, states)
            output = self.linear(outputs_lstm)
            
            prob, word = output.max(2)
            word_item = word.item()
            sentence.append(word_item)
            
            inputs = self.embed(word)
            
        return sentence
        