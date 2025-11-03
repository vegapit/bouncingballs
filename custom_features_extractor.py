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
        hero_position_dim = 32
        ball_feature_dim = 64
        ball_status_embeddings_dim = 16
        ball_color_embeddings_dim = 16
        
        # Hero encoding
        self.hero_position_encoder = ResidualBlock(hero_shape, hero_position_dim)
            
        # Ball color embedding
        self.balls_color_embedding = nn.Embedding(2, ball_color_embeddings_dim)

        # Ball status embedding
        self.balls_status_embedding = nn.Embedding(2, ball_status_embeddings_dim)
        
        # Per-ball encoder: combines position, speed, color and status
        self.ball_encoder = nn.Sequential(
            ResidualBlock(2 + 2 + 2 + ball_color_embeddings_dim + ball_status_embeddings_dim, ball_feature_dim),
            ResidualBlock(ball_feature_dim, ball_feature_dim),
            ResidualBlock(ball_feature_dim, ball_feature_dim),
            ResidualBlock(ball_feature_dim, ball_feature_dim),
            ResidualBlock(ball_feature_dim, ball_feature_dim)
        )
        
        # Final combination
        self.combined_net = ResidualBlock(hero_position_dim + ball_feature_dim, features_dim)

    def forward(self, observations):
        batch_size = observations['hero'].shape[0]
        
        # Encode hero
        hero_position = observations['hero']  # (batch, 2)
        hero_pos_encoded = self.hero_position_encoder( hero_position )  # (batch, hero_position_dim)

        # Encode each ball
        balls_position = observations['balls_position']  # (batch, num_balls, 2)
        balls_speed = observations['balls_speed']  # (batch, num_balls, 2)
        balls_color = observations['balls_color'].long()  # (batch, num_balls)
        balls_status = observations['balls_status'].long()  # (batch, num_balls)
        
        # Get ball embeddings
        color_embeddings = self.balls_color_embedding( balls_color )  # (batch, num_balls, ball_color_embeddings_dim)
        status_embeddings = self.balls_status_embedding( balls_status )  # (batch, num_balls, ball_status_embeddings_dim)
        
        # Concatenate position, speed, and type for each ball
        ball_inputs = torch.cat([
            balls_position,
            balls_position - hero_position.unsqueeze(1),  # relative position
            balls_speed,
            color_embeddings,
            status_embeddings
        ], dim=-1)  # (batch, num_balls, 2 + 2 + 2 + ball_color_embeddings_dim + ball_status_embeddings_dim)
        
        # Encode each ball
        ball_features = self.ball_encoder( ball_inputs ) # (batch, num_balls, ball_feature_dim)
        
        # Pool ball features
        ball_features_pooled = ball_features.mean(dim=1)  # (batch, ball_feature_dim)

        # Combine all features
        combined = torch.cat([hero_pos_encoded, ball_features_pooled], dim=1)
        return self.combined_net(combined)