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
            
    def train(self, eur_series, gbp_series, epochs=20, batch_size=32, seq_len=60, lr=0.001):
        """
        Trains the Transformer to predict EURUSD direction based on EUR+GBP history.
        """
        # 1. Prepare Data
        # Ensure lengths match
        min_len = min(len(eur_series), len(gbp_series))
        eur = eur_series[:min_len].values.astype(np.float32)
        gbp = gbp_series[:min_len].values.astype(np.float32)
        
        # Create Targets (1 if Next EUR Close > Current, else 0)
        # We need raw close prices for targets? input is series.
        # Assuming series are Log Returns or Prices? 
        # Best practice: Inputs = Log Returns. Target = Sign of next return.
        
        # Let's assume inputs are Log Returns.
        # y[t] = 1 if eur[t+1] > 0 else 0
        targets = (eur[1:] > 0).astype(np.float32)
        
        # Truncate inputs to match targets
        eur = eur[:-1]
        gbp = gbp[:-1]
        
        # Create Sequences
        X = []
        y = []
        for i in range(len(eur) - seq_len):
            _eur = eur[i:i+seq_len]
            _gbp = gbp[i:i+seq_len]
            _x = np.stack([_eur, _gbp], axis=1) # (Seq, 2)
            _y = targets[i+seq_len] # Target for the step AFTER the sequence
            X.append(_x)
            y.append(_y)
            
        X = np.array(X)
        y = np.array(y)
        
        # Tensorize
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X), 
            torch.tensor(y).unsqueeze(1)
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 2. Setup Loop
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        
        print(f"Training Transformer on {len(dataset)} sequences for {epochs} epochs...")
        print(f"DataLoader has {len(dataloader)} batches.")
        
        for epoch in range(epochs):
            print(f"Starting Epoch {epoch+1}/{epochs}...")
            total_loss = 0
            batch_count = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_x) 
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batch_count += 1
                if batch_count % 10 == 0:
                     print(f"  Processed batch {batch_count}/{len(dataloader)}")
            
            print(f"Epoch {epoch+1}/{epochs} Completed, Loss: {total_loss/len(dataloader):.4f}")
    
    def predict(self, eur_seq, gbp_seq):
        """
        Predicts next candle probability. 
        Input: Sequence of last N returns (List or Array).
        """
        # Ensure input is length 60
        seq_len = 60
        if len(eur_seq) < seq_len or len(gbp_seq) < seq_len:
             # Pad or return neutral 0.5
             return 0.5
             
        # Take last 60
        e = eur_seq[-seq_len:]
        g = gbp_seq[-seq_len:]
        
        data = np.stack([e, g], axis=1) # (60, 2)
        tensor_x = torch.tensor(data, dtype=torch.float32).unsqueeze(0) # (1, 60, 2)
        
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
