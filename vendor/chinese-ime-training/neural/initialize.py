"""Strictly load a chinese-ime-lm safetensors artifact into the upstream model for IR export."""
import json
import torch
from safetensors import safe_open
from model import CharLM, Config, RESERVED


def load_initial_model(path):
    with safe_open(str(path), framework='pt', device='cpu') as artifact:
        metadata = artifact.metadata()
        if metadata.get('format') != 'chinese-ime-lm' or metadata.get('version') != '1':
            raise ValueError('unsupported model artifact')
        if metadata.get('tied_embeddings') != 'true':
            raise ValueError('model must tie embeddings')
        cfg = Config(**json.loads(metadata['config']))
        vocab = json.loads(metadata['vocab'])
        if len(vocab) != cfg.vocab or vocab[:3] != RESERVED or len(set(vocab)) != len(vocab):
            raise ValueError('invalid fixed vocabulary')
        model = CharLM(cfg)
        expected = model.state_dict()
        keys, used, state = set(artifact.keys()), set(), {}
        for name in set(expected) - {'head.weight'}:
            tensor = artifact.get_tensor(name)
            used.add(name)
            if tensor.shape != expected[name].shape:
                raise ValueError(f'{name}: wrong shape')
            if tensor.dtype == torch.int8:
                if tensor.ndim != 2:
                    raise ValueError(f'{name}: int8 tensor is not matrix')
                scales = artifact.get_tensor(name + '.scale')
                used.add(name + '.scale')
                if scales.dtype != torch.float32 or scales.shape != (tensor.shape[0],):
                    raise ValueError(f'{name}: invalid per-row scale')
                if not torch.isfinite(scales).all() or not (scales > 0).all():
                    raise ValueError(f'{name}: invalid scale values')
                tensor = tensor.float() * scales[:, None]
            elif tensor.dtype in (torch.float16, torch.float32):
                tensor = tensor.float()
            else:
                raise ValueError(f'{name}: unsupported dtype')
            if not torch.isfinite(tensor).all():
                raise ValueError(f'{name}: nonfinite weights')
            state[name] = tensor
        if keys != used:
            raise ValueError(f'unexpected model keys: {sorted(keys - used)}')
        state['head.weight'] = state['tok.weight']
        model.load_state_dict(state, strict=True)
        return model, cfg, vocab, metadata
