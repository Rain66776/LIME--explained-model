import torch
import torch.nn.functional as F


def create_predict_fn(model, vocab, max_seq_length, device):
    def predict_proba(texts):
        model.eval()
        if isinstance(texts, str):
            texts = [texts]
        
        from src.preprocess import text_to_indices
        
        indices_list = []
        for text in texts:
            indices = text_to_indices(text, vocab, max_seq_length)
            indices_list.append(indices)
        
        tensors = torch.tensor(indices_list, dtype=torch.long).to(device)
        
        with torch.no_grad():
            outputs = model(tensors)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
        
        return probs
    
    return predict_proba
