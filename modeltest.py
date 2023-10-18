from cramming.architectures.crammed_bert import *
from cramming.architectures.attention import SeqFirstSelfAttention
import torch
from cramming.architectures.attention import get_attention_mechanism
from transformers import PretrainedConfig, PreTrainedModel
from typing import Optional
import hydra
from omegaconf import OmegaConf
import os

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2,3"  # Set the GPU 2 to use

cfg = {'arch': {'architectures': ['ScriptableCrammedBERT'], 'num_transformer_layers': 10, 'hidden_size': 768, 'intermed_size': 3072, 'hidden_dropout_prob': 0.1, 'norm': 'LayerNorm', 'norm_eps': 1e-12, 'norm_scheme': 'pre', 'nonlin': 'GELUglu', 'tie_weights': True, 'decoder_bias': False, 'sparse_prediction': '${train.objective.mlm_probability}', 'loss': 'cross-entropy', 'objective_layout': 'MLM', 'embedding': {'vocab_size': None, 'pos_embedding': 'scaled-sinusoidal', 'dropout_prob': 0.1, 'pad_token_id': 0, 'max_seq_length': 128, 'embedding_dim': '${arch.hidden_size}', 'normalization': True, 'stable_low_precision': False}, 'attention': {'type': 'self-attention', 'causal_attention': False, 'num_attention_heads': 12, 'dropout_prob': 0.1, 'skip_output_projection': False, 'qkv_bias': False, 'rotary_embedding': False, 'seq_op_in_fp32': False, 'sequence_op': 'exp'}, 'init': {'type': 'normal', 'std': 0.02}, 'ffn_layer_frequency': 1, 'skip_head_transform': True, 'use_bias': False, 'final_norm': True, 'num_labels': None, 'classification_head': {'pooler': 'avg', 'include_ff_layer': True, 'head_dim': 1024, 'nonlin': 'Tanh', 'classifier_dropout': '${arch.hidden_dropout_prob}'}}, 'data': {'name': 'pile-readymade', 'sources': {'hub': {'provider': 'hub'}}, 'hf_location': 'JonasGeiping/the_pile_WordPiecex32768_2efdb9d060d1ae95faf952ec1a50f020', 'streaming': True, 'vocab_size': 32768, 'seq_length': 128}, 'impl': {'path': 'data', 'local_staging_dir': None, 'forbid_dataset_preprocessing': False, 'temporary_corpus': False, 'max_raw_chunk_size': 8000000.0, 'print_loss_every_nth_step': 1000, 'save_intermediate_checkpoints': False, 'save_every_nth_step': 5000, 'resume_run_after_preempt': True, 'troubleshoot_strategy': 'recover_checkpoint', 'early_termination': {'enabled': False, 'budget': 3, 'loss_threshold': 6.0}, 'microbatch_size': 128, 'threads': 32, 'benchmark': True, 'deterministic': False, 'non_blocking': True, 'tf32_allowed': True, 'matmul_precision': 'high', 'pad_to_multiple_of': 8, 'shuffle_in_dataloader': False, 'pin_memory': True, 'prefetch_factor': 2, 'persistent_workers': True, 'default_precision': 'float', 'dist_backend': 'nccl', 'sharing_strategy': None, 'enable_huggingface_offline_mode': False, 'local_rank': None, 'push_to_huggingface_hub': False, 'hf_directoy_name': 'test-crammedBERT-c5', 'add_env_variables': None, 'name': 'torch-default', 'mixed_precision': True, 'grad_scaling': True, 'mixed_precision_target_dtype': 'float16', 'zero_redundancy_optimizer': False, 'broadcast_buffers': False, 'bucket_cap_mb': 25, 'gradient_as_bucket_view': True, 'static_graph': False, 'foreach_optimizer': False, 'compile_torch': True, 'mode': None, 'dynamic': False, 'fullgraph': True, 'backend': 'inductor', '_inductor_vars': {'triton': {'cudagraphs': True}, 'permute_fusion': True, 'shape_padding': True}, 'enable_mem_efficient_sdp': True, 'enable_math_sdp': True, 'enable_flash_sdp': True}, 'wandb': {'enabled': False, 'entity': None, 'project': None, 'tags': []}, 'train': {'optim': {'type': 'AdamW', 'lr': 0.001, 'betas': [0.9, 0.98], 'eps': 1e-12, 'weight_decay': 0.01, 'amsgrad': False, 'fused': None}, 'optim_mod': {'name': 'none'}, 'name': 'bert-o4', 'limited_decay_keys': ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'norm'], 'warmup_steps': 0, 'cooldown_steps': 0, 'steps': 900000, 'scheduler': 'budget-triangle2', 'batch_size': 8192, 'batch_size_ramp': 0.6, 'gradient_clipping': 0.5, 'pretrain_in_train_mode': False, 'objective': {'name': 'masked-lm', 'mlm_probability': 0.25, 'use_80_20_rule': True, 'disable_mlm': False, 'token_drop': 0.0}, 'reverse_dataset_order': False, 'budget': '${budget}'}, 'base_dir': 'outputs', 'seed': None, 'name': 'dhrho', 'budget': 12, 'dryrun': False}
cfg = dotdict(cfg)
cfg.arch = dotdict(cfg.arch)

# model = construct_crammed_bert(cfg.arch, cfg.data['vocab_size'], downstream_classes=None)