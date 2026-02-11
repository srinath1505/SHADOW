import torch
import torch.nn as nn
import numpy as np

class LeadLagTransformer(nn.Module):
    def __init__(self, input_dim=2, d_model=64, nhead=4, num_layers=2, output_dim=1):
        """
        Transformer model to detect Lead/Lag relationships.
        :param input_dim: Number of input features (2: EUR returns, GBP returns).
        :param d_model: Dimension of the transformer model.
        :param nhead: Number of attention heads.
        :param num_layers: Number of transformer encoder layers.
        """
        super(LeadLagTransformer, self).__init__()
        
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, d_model)) # Max seq len 100
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.decoder = nn.Linear(d_model, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        :param x: Input tensor of shape (batch_size, seq_len, input_dim)
        :return: Probability score (batch_size, 1)
        """
        # x shape: [batch, seq, features]
        batch_size, seq_len, _ = x.size()
        
        # Embed
        x = self.embedding(x)
        
        # Add Positional Encoding (broadcast)
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # Transformer Pass
        output = self.transformer_encoder(x)
        
        # We take the last time step's output for prediction
        last_output = output[:, -1, :]
        
        # Decode
        score = self.decoder(last_output)
        return self.sigmoid(score)

class TransformerPredictor:
    def __init__(self, model_path=None):
        self.model = LeadLagTransformer()
        if model_path:
            self.load_model(model_path)
            
    def train(self, eur_series, gbp_series, targets, epochs=10):
        """
        Skeleton for training loop.
        """
        pass

    def predict(self, eur_series, gbp_series):
        """
        Predicts lead/lag score.
        :param eur_series: List or array of EUR returns.
        :param gbp_series: List or array of GBP returns.
        :return: Score (0.0 to 1.0). > 0.5 implies GBP leads EUR (EUR lags)?
        """
        # Prepare input
        # Combine series
        data = np.stack([eur_series, gbp_series], axis=1) # (Seq, 2)
        tensor_x = torch.tensor(data, dtype=torch.float32).unsqueeze(0) # (1, Seq, 2)
        
        self.model.eval()
        with torch.no_grad():
            score = self.model(tensor_x)
            return score.item()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

if __name__ == "__main__":
    # Test
    model = LeadLagTransformer()
    # Dummy input: Batch=1, Seq=60, Feat=2
    dummy_x = torch.randn(1, 60, 2)
    output = model(dummy_x)
    print(f"Output Score: {output.item()}")
