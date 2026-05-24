import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class NovaTokenizer:
    """
    Wrapper that handles both custom BPE tokenizer (from HuggingFace tokenizers lib)
    and the fallback tiktoken tokenizer, providing a unified encode/decode interface.
    """
    def __init__(self, tokenizer_path=None):
        self.tokenizer_path = tokenizer_path
        self.is_custom = False
        
        if tokenizer_path and os.path.exists(tokenizer_path):
            from tokenizers import Tokenizer
            self._tokenizer = Tokenizer.from_file(tokenizer_path)
            self.vocab_size = self._tokenizer.get_vocab_size()
            self.is_custom = True
            print(f"  Loaded custom tokenizer: {self.vocab_size:,} tokens")
        else:
            import tiktoken
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
            self.vocab_size = 100277
            print(f"  Using default tokenizer: cl100k_base ({self.vocab_size:,} tokens)")
    
    def encode(self, text: str) -> list:
        if self.is_custom:
            return self._tokenizer.encode(text).ids
        else:
            return self._tokenizer.encode(text)
    
    def decode(self, ids: list) -> str:
        if self.is_custom:
            return self._tokenizer.decode(ids)
        else:
            return self._tokenizer.decode(ids)


class SigmaNOVA(nn.Module):
    EOS_TOKEN = "<|end|>"  # End-of-response marker
    SEP_TOKEN = "<|sep|>"  # Separator between question and answer
    
    def __init__(self, vocab_size=None, context_dim=512, pc_dims=[512, 1024, 512], tokenizer_path=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.context_dim = context_dim
        
        # Initialize tokenizer (custom or fallback)
        self.tokenizer = NovaTokenizer(tokenizer_path)
        
        if vocab_size is not None:
            self.vocab_size = vocab_size
        else:
            self.vocab_size = self.tokenizer.vocab_size
        
        # 1. Hypersphere Embeddings
        self.embedding = nn.Embedding(self.vocab_size, context_dim)
        with torch.no_grad():
            nn.init.normal_(self.embedding.weight)
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=1)
            
        # 2. Gated Context Memory (Like Mamba/RNN)
        self.W_gate = nn.Linear(context_dim * 2, context_dim)
        
        # 3. Predictive Coding Hierarchy
        self.pc_layers = nn.ModuleList()
        in_dim = context_dim
        for out_dim in pc_dims:
            self.pc_layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
            
        # Move to device
        self.to(self.device)
        self.current_context = None

    def forward(self, current_word_ids):
        """
        Forward pass for inference. 
        Note: Training logic has been stripped from this public release.
        """
        batch_size = current_word_ids.shape[0]
        
        if self.current_context is None or self.current_context.shape[0] != batch_size:
            self.current_context = torch.zeros(batch_size, self.context_dim, device=self.device)
            
        # Get word embeddings
        word_embs = self.embedding(current_word_ids)
        word_embs = F.normalize(word_embs, p=2, dim=1)
        
        # Update Gated Context
        gate_input = torch.cat([self.current_context, word_embs], dim=1)
        gate = torch.sigmoid(self.W_gate(gate_input))
        
        new_info = torch.tanh(word_embs)
        ctx = gate * self.current_context + (1 - gate) * new_info
        
        # Spherical Constraint
        ctx_norm = F.layer_norm(ctx, ctx.shape[1:])
        self.current_context = F.normalize(ctx_norm, p=2, dim=1)
        
        # Predictive Coding Hierarchy
        current_state = self.current_context
        for i, layer in enumerate(self.pc_layers):
            raw = layer(current_state)
            normed = F.layer_norm(raw, raw.shape[1:])
            pred = torch.tanh(normed)
            pred = F.normalize(pred, p=2, dim=1)
            
            if i < len(self.pc_layers) - 1:
                current_state = pred
                
        return pred

    def generate(self, prompt: str, n_words: int = 20, temperature: float = 0.3) -> str:
        """Generates a response from the model based on the prompt."""
        self.eval()
        prompt_ids = self.tokenizer.encode(prompt)
        
        # Clear context for new generation
        self.current_context = torch.zeros(1, self.context_dim, device=self.device)
        
        # Feed the prompt through the model to build context
        with torch.no_grad():
            for tid in prompt_ids:
                cur_id = torch.tensor([tid], device=self.device)
                pred = self.forward(cur_id)
            
        # Generation loop - clean sampling with EOS stopping
        generated_ids = []

        for _ in range(n_words):
            with torch.no_grad():
                valid_vocab_size = self.vocab_size
                
                # Math for probability extraction
                logits = torch.matmul(self.embedding.weight[:valid_vocab_size], pred.T).squeeze(1)
                sims = logits * 15.0

                # Apply temperature
                sims = sims / temperature

                # Convert to probabilities
                sims_clipped = torch.clamp(sims, min=-80, max=80)
                probs = torch.softmax(sims_clipped, dim=-1)

                # Sample from the distribution
                if probs.sum() > 0:
                    categorical = torch.distributions.Categorical(probs)
                    chosen_tid = categorical.sample().item()
                else:
                    chosen_tid = torch.argmax(probs).item()

                generated_ids.append(chosen_tid)

                # Check if we just generated the EOS token
                decoded_so_far = self.tokenizer.decode(generated_ids)
                if self.EOS_TOKEN in decoded_so_far:
                    return decoded_so_far.replace(self.EOS_TOKEN, "").strip()

                cur_id = torch.tensor([chosen_tid], device=self.device)
                pred = self.forward(cur_id)
                
        # If we hit max words without EOS, still strip any partial EOS
        result = self.tokenizer.decode(generated_ids)
        return result.replace(self.EOS_TOKEN, "").strip()
        
    def load(self, path: str):
        """Loads weights from a saved file, ignoring missing training optimizer states."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        # We only care about the model_state (which ignores optimizer/scaler states from training)
        if "model_state" in checkpoint:
            self.load_state_dict(checkpoint["model_state"], strict=False)
        else:
            self.load_state_dict(checkpoint, strict=False)
