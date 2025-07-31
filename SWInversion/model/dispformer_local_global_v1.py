import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MaskedConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        kwargs['bias'] = False  # force to remove bias
        super().__init__(*args, **kwargs)
        
    def forward(self, x, mask):
        # mask shape: (B, 1, L), x shape: (B, C_in, L)
        x_masked = x * mask
        # conv operation (conv_out shape: (B, C_out, L))
        conv_out = super().forward(x_masked)
        # use the all-1 kernel to count the number of mask=1 in each conv window
        with torch.no_grad():
            ones_kernel = torch.ones(
                (1, 1, self.kernel_size[0]), device=x.device
            )
            valid_count = F.conv1d(
                mask.float(),  # convert mask from bool to float
                ones_kernel,
                bias=None,
                stride=self.stride[0],
                padding=self.padding[0],
                dilation=self.dilation[0]
            )
            valid_count = valid_count.clamp(min=1e-6)
        # not to be divided by 0
        output = conv_out / valid_count
        return output

class LocalFeatureExtraction(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        
        self.conv1 = MaskedConv1d(model_dim, model_dim, kernel_size=7, padding=7//2)
        self.conv2 = MaskedConv1d(model_dim, model_dim, kernel_size=5, padding=5//2)
        self.conv3 = MaskedConv1d(model_dim, model_dim, kernel_size=3, padding=3//2)
        self.relu  = nn.ReLU()

    def forward(self, x, mask):
        x = self.relu(self.conv1(x, mask.clone()))
        x = self.relu(self.conv2(x, mask.clone()))
        x = self.conv3(x, mask.clone())
        x = self.relu(x)
        return x
    
class DispersionTransformer(nn.Module):
    def __init__(self, 
                 model_dim, 
                 num_heads, 
                 num_layers, 
                 output_dim, 
                 scale_factor=6.5):
        super(DispersionTransformer, self).__init__()
        
        # embedding (transform each data point to the same embedding dimension)
        self.period_embedding = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=model_dim, kernel_size=1, stride=1),
            nn.ReLU()
        )
        self.phase_velocity_encoding = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=model_dim, kernel_size=1, stride=1),
            nn.ReLU()
        )
        self.group_velocity_encoding = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=model_dim, kernel_size=1, stride=1),
            nn.ReLU()
        )

        # local feature extraction (extract the local feature from the velocity data)
        self.local_feature_extraction_phaseVelocity = LocalFeatureExtraction(model_dim=model_dim)
        self.local_feature_extraction_groupVelocity = LocalFeatureExtraction(model_dim=model_dim)
        
        # long-range feature extraction (extract the long-range feature from the local feature)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, 
                                       nhead=num_heads,
                                       dropout=0,
                                       batch_first=True
                                       ),
            num_layers=num_layers
        )
        
        # fully connected layer (fuse the long-range feature and the local feature)
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.max_pooling    = nn.AdaptiveMaxPool1d(1)
        
        self.fc_fuse = nn.Sequential(
            nn.Linear(2*model_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.Sigmoid()
        )
        self.scale_factor = scale_factor

    def forward(self, input_data, mask=None):
        """
            mask for the padding data [period,phase velocity, group velocity] <=0
        """
        # input data: [period,phase velocity, group velocity]
        period_data, phase_velocity, group_velocity = input_data[:, 0, :], input_data[:, 1, :], input_data[:, 2, :]
        phase_velocity_solid_mask = (phase_velocity>0).unsqueeze(1) # (batch_size, 1, seq_length)
        group_velocity_solid_mask = (group_velocity>0).unsqueeze(1) # (batch_size, 1, seq_length)
        period_solid_mask         = (period_data>0).unsqueeze(1)    # (batch_size, 1, seq_length)
        
        # embedding
        period_embedding         = self.period_embedding(period_data.unsqueeze(1))           # (batch_size, model_dim, seq_length)
        phase_velocity_embedding = self.phase_velocity_encoding(phase_velocity.unsqueeze(1)) # (batch_size, model_dim, seq_length)
        group_velocity_embedding = self.group_velocity_encoding(group_velocity.unsqueeze(1)) # (batch_size, model_dim, seq_length)
        
        # local feature extraction
        phase_velocity_embedding = self.local_feature_extraction_phaseVelocity(phase_velocity_embedding, phase_velocity_solid_mask) # (batch_size, model_dim, seq_length)
        group_velocity_embedding = self.local_feature_extraction_groupVelocity(group_velocity_embedding, group_velocity_solid_mask) # (batch_size, model_dim, seq_length)

        # print(group_velocity_solid_mask[0],group_velocity_embedding[0],group_velocity_solid_mask.shape,group_velocity_embedding.shape)
        combined_embedding = period_embedding + phase_velocity_embedding + group_velocity_embedding
        fused_features = combined_embedding.permute(0,2,1) # (batch_size, seq_length, model_dim)

        # Apply transformer encoder with mask to ignore padded positions
        transformer_output = self.transformer_encoder(fused_features, src_key_padding_mask=mask)  # (batch_size, seq_length, model_dim)
        
        # aggregation all the model dim
        output_avgpool = self.global_pooling(transformer_output.permute(0,2,1))  # (batch_size, model_dim, 1)
        output_maxpool = self.max_pooling(transformer_output.permute(0,2,1))  # (batch_size, model_dim, 1)
        output = torch.cat([output_avgpool, output_maxpool], dim=1)  # (batch_size, 2*model_dim, 1)

        
        # Apply fully connected layer for final output
        output = self.fc_fuse(output.squeeze(-1))  # (batch_size, output_dim)
        output = output * self.scale_factor  # Scale the output
        
        return output