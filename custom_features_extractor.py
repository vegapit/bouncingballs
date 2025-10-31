import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.skip(x) + self.block(x)

class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        hero_shape = observation_space['hero'].shape[0]
        num_balls = observation_space['balls_position'].shape[0]
        
        # Per-ball feature dimension
        ball_feature_dim = 64
        embeddings_dim = 16
        
        # Hero encoding
        self.hero_position_encoder = ResidualBlock(hero_shape, 32)
            
        # Ball type embedding
        self.balls_type_embedding = nn.Embedding(3,embeddings_dim)
        
        # Per-ball encoder: combines position, speed, and type
        # Input: abssolute_position(2) + relative_position(2) + speed(2) + type_embedding(embeddings_dim)
        self.ball_encoder = nn.Sequential(
            ResidualBlock(2 + 2 + 2 + embeddings_dim, ball_feature_dim),
            ResidualBlock(ball_feature_dim, ball_feature_dim),
            ResidualBlock(ball_feature_dim, ball_feature_dim),
            ResidualBlock(ball_feature_dim, ball_feature_dim),
            ResidualBlock(ball_feature_dim, ball_feature_dim)
        )
        
        # Final combination
        self.combined_net = ResidualBlock(32 + ball_feature_dim, features_dim)

    def forward(self, observations):
        batch_size = observations['hero'].shape[0]
        
        # Encode hero
        hero_position = observations['hero']  # (batch, 2)
        
        # Encode each ball
        balls_position = observations['balls_position']  # (batch, num_balls, 2)
        balls_speed = observations['balls_speed']  # (batch, num_balls, 2)
        balls_type = observations['balls_type'].long()  # (batch, num_balls)
        
        hero_pos_encoded = self.hero_position_encoder( hero_position )  # (batch, 32)
        
        # Get type embeddings
        type_embeddings = self.balls_type_embedding( balls_type )  # (batch, num_balls, 64)
        
        # Concatenate position, speed, and type for each ball
        ball_inputs = torch.cat([
            balls_position,
            balls_position - hero_position.unsqueeze(1),  # relative position
            balls_speed,
            type_embeddings
        ], dim=-1)  # (batch, num_balls, 2 + 2 + 2 + embeddings_dim)
        
        # Encode each ball
        ball_features = self.ball_encoder( ball_inputs ) # (batch, num_balls, 64)
        
        # Pool ball features
        ball_features_pooled = ball_features.mean(dim=1)  # (batch, 64)

        # Combine all features
        combined = torch.cat([hero_pos_encoded, ball_features_pooled], dim=1)
        return self.combined_net(combined)