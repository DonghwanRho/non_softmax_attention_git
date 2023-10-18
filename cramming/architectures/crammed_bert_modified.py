"""This rewrite is a simplified version of the proposed changes that actually compiles statically in torch 2.0.

This model is the final, optimized crammed model.
OmegaConf
Not all ablations discussed in the paper are implemented as switches in this version,
for all those, check scriptable_bert.py on the old branch.

"""
import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoModelForTokenClassification

from typing import Optional
from omegaconf import OmegaConf
from termcolor import colored
import os

from .components import (
    _get_norm_fn,
    _get_nonlin_fn,
    EmbeddingComponent_modified,
    PoolingComponent,
    PredictionHeadComponent,
    GLU,
    get_extended_attention_mask,
    _init_module,
)
from .attention_modified import get_attention_mechanism
import matplotlib.pyplot as plt

class crammedBertConfig(PretrainedConfig):
    model_type = "crammedBERT"

    def __init__(self, cfg_arch_container: dict = {}, **kwargs):
        self.arch = cfg_arch_container
        super().__init__(**kwargs)

def construct_crammed_bert_modified(cfg_arch, vocab_size, downstream_classes=None):
    """See the config file for details on what is possible."""
    config = crammedBertConfig(OmegaConf.to_container(cfg_arch, resolve=True))
    config.arch["embedding"]["vocab_size"] = vocab_size
    config.arch["num_labels"] = downstream_classes

    if downstream_classes is None:
        if config.arch["objective_layout"] == "MLM":
            model = ScriptableLMForPreTraining_modified(config)
        elif config.arch["objective_layout"] == "SCRIPT":
            model = ScriptableLMForSCRIPTTraining(config)
        else:
            raise ValueError(f"Invalid layout {config.arch['objective_layout']} of training objective given.")
    else:
        model = ScriptableLMForSequenceClassification_modified(config)
    return model

class FFNComponent(torch.nn.Module):
    """Note: The FF layer is not auto-scaled when using a GLU type activation.
    It actually turned out better not to scale it, so here the block is effectively smaller than may be expected.

    The neox suggestion for approx. equal parameter count is int(4 * 2 / 3 * hidden_size) * 2 [this is ~5.33]
    """

    def __init__(self, hidden_size, intermed_size, nonlin_fn=torch.nn.GELU, use_bias=True):
        super().__init__()
        self.dense_in = torch.nn.Linear(hidden_size, intermed_size, bias=use_bias)
        self.nonlin = nonlin_fn()
        if isinstance(self.nonlin, GLU):
            intermed_output_size = intermed_size // 2
        else:
            intermed_output_size = intermed_size
        self.dense_out = torch.nn.Linear(intermed_output_size, hidden_size, bias=use_bias)

    def forward(self, hidden_states):
        return self.dense_out(self.nonlin(self.dense_in(hidden_states)))

class AttentionComponent_modified(torch.nn.Module):
    def __init__(self, idx, hidden_size, cfg_attention, use_bias=True):
        super().__init__()
        self.self_attention = get_attention_mechanism(idx, hidden_size, cfg_attention)
        if cfg_attention.skip_output_projection:
            self.dense = torch.nn.Identity()
        else:
            self.dense = torch.nn.Linear(self.self_attention.output_dim, hidden_size, bias=use_bias)

        self.LAYOUT = self.self_attention.LAYOUT

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        output, matmul_result = self.self_attention(hidden_states, attention_mask)
        output = self.dense(output)
        return output, matmul_result


class TransformerLayer_modified(torch.nn.Module):
    """A transformer-encoder structure based on the components from above."""

    def __init__(self, idx, cfg_arch):
        super().__init__()
        self.dropout = torch.nn.Dropout(cfg_arch.hidden_dropout_prob, inplace=False)
        self.norm1 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
        self.norm2 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
        self.attn = AttentionComponent_modified(
            idx,
            cfg_arch.hidden_size,
            cfg_arch.attention,
            cfg_arch.use_bias,
        )
        self.LAYOUT = self.attn.LAYOUT

        self.ffn = FFNComponent(
            cfg_arch.hidden_size,
            cfg_arch.intermed_size,
            _get_nonlin_fn(cfg_arch.nonlin),
            cfg_arch.use_bias,
        )  
        
    def forward(self, states, attention_mask: Optional[torch.Tensor] = None):
        states2, matmul_result = self.attn(self.norm1(states), attention_mask)
        states = states + self.dropout(states2)
        states = states + self.dropout(self.ffn(self.norm2(states)))
        
        return states, matmul_result
 
class ScriptableLM_modified(PreTrainedModel):
    """Simplified transformer wrapper."""

    config_class = crammedBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)

        self.embedding = EmbeddingComponent_modified(self.cfg.embedding, self.cfg.norm, self.cfg.norm_eps)
        self.layers = torch.nn.ModuleList([TransformerLayer_modified(idx, self.cfg) for idx in range(self.cfg.num_transformer_layers)])
         
        self.seq_first = True
        self.use_causal_attention = self.cfg.attention.causal_attention

        if self.cfg.final_norm:
            self.final_norm = _get_norm_fn(self.cfg.norm)(self.cfg.hidden_size, eps=self.cfg.norm_eps)
        else:
            self.final_norm = torch.nn.Identity()


    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        matmuls = []
        
        if attention_mask is not None:
            attention_mask = get_extended_attention_mask(attention_mask, input_ids.shape, self.use_causal_attention)
        hidden_states = self.embedding(input_ids)

        if self.seq_first:
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        
        for i, layer_module in enumerate(self.layers):
            hidden_states, matmul = layer_module(hidden_states, attention_mask)
            matmuls.append(matmul)
            
        if self.seq_first:
            hidden_states = hidden_states.transpose(0, 1).contiguous()

        return self.final_norm(hidden_states), matmuls

class ScriptableLMForPreTraining_modified(PreTrainedModel):
    """Pretraining version with optional prediction head and variant for sparse prediction."""

    config_class = crammedBertConfig

    def __init__(self, config):
        super().__init__(config)
        
        self.cfg = OmegaConf.create(config.arch)

        self.encoder = ScriptableLM_modified(config)

        if not self.cfg.skip_head_transform:
            self.prediction_head = PredictionHeadComponent(self.cfg)
        else:
            self.prediction_head = torch.nn.Identity()  # from linear in old version

        self.decoder = torch.nn.Linear(self.cfg.embedding.embedding_dim, self.cfg.embedding.vocab_size, bias=self.cfg.decoder_bias)
        self.decoder.weight = self.encoder.embedding.word_embedding.weight

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.sparse_prediction = self.cfg.sparse_prediction

        self._init_weights()
        
        self.count = 0
        self.x_list = []
        self.loss_list = []
        self.last_100_loss_list = []
        self.matmul_results = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.best_loss = 0

    def _init_weights(self, module=None):
        modules = self.modules() if module is None else [module]
        for module in modules:
            _init_module(
                module,
                self.cfg.init.type,
                self.cfg.init.std,
                self.cfg.hidden_size,
                self.cfg.num_transformer_layers,
            )

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, **kwargs):
        outputs, matmuls_from_enc = self.encoder(input_ids, attention_mask)
        outputs = outputs.view(-1, outputs.shape[-1])

        len_matmuls = len(matmuls_from_enc)

        if self.sparse_prediction and labels is not None:            
            masked_lm_loss = self._forward_sparse(outputs, labels) ### loss
            
            self.count += 1
            self.x_list.append(self.count)
            self.loss_list.append(masked_lm_loss.item())
            
            if self.count < 100:
                last_100_loss = sum(self.loss_list) / len(self.loss_list)
                self.last_100_loss_list.append(last_100_loss)
                print('\nLoss: {}, Last_100_losses: {}, Layers: {}, Count: {}'.format(masked_lm_loss.item(), last_100_loss, self.cfg.num_transformer_layers, self.count))
            else:
                last_100_loss = sum(self.loss_list[-100 :]) / len(self.loss_list[-100 :])
                self.last_100_loss_list.append(last_100_loss)
                if self.best_loss == 0 or last_100_loss < self.best_loss:
                    self.best_loss = last_100_loss
                print('\nLoss: {}, Last_100_losses: {}, Best_100_losses: {}, Layers: {}, Count: {}'.format(masked_lm_loss.item(), last_100_loss, self.best_loss, self.cfg.num_transformer_layers, self.count))
            
            for i in range(len_matmuls):
                norm_i = torch.norm(matmuls_from_enc[i], p=float('inf'))
                print('Matmul_{}: {}'.format(i, norm_i.item()))
                self.matmul_results[i].append(norm_i.item())
            # Impose a norm penalty    
            ###################################### 
                # Loss 추가
                if norm_i > 160:
                    masked_lm_loss += 10 * norm_i
            print('Norm Penalty: O')
            ######################################
                
            # Plot loss and matmuls  
            ######################################
            if self.count % 100 == 0:
                plt.plot(self.x_list, self.loss_list)
                plt.title('Loss')
                plt.xlabel('Steps')
                plt.savefig('losses.png')
                plt.clf()
                plt.plot(self.x_list, self.last_100_loss_list)
                plt.title('Last 100 losses')
                plt.xlabel('Steps')
                plt.savefig('last_100_losses.png')
                plt.clf()
                for i in range(len_matmuls):
                    plt.subplot(7, 4, i + 1)
                    plt.plot(self.x_list, self.matmul_results[i])
                    plt.title('Matmul_{}'.format(i))
                    plt.xlabel('Steps')
                plt.savefig('matmuls.png')
                plt.clf()
            ######################################
            
        else:
            outputs = self.decoder(self.prediction_head(outputs))
            if labels is not None:
                masked_lm_loss = self.loss_fn(outputs, labels.view(-1))
            else:
                masked_lm_loss = outputs.new_zeros((1,))

        return {"loss": masked_lm_loss, "outputs": outputs}

    # Sparse prediction usually has an unpredictable number of entries in each batch
    # but the dataloader was modified so that 25% of the batch is ALWAYS masked.
    # This allows for static compilation. If you modify the dataloader, this function will fill your compile cache
    def _forward_sparse(self, outputs: torch.Tensor, labels: Optional[torch.Tensor] = None):

        labels = labels.view(-1)
        mask_positions = labels.view(-1) != self.loss_fn.ignore_index
        num_masks_guaranteed = round(self.sparse_prediction * labels.shape[0])
        indices = torch.argsort(mask_positions.int())[-num_masks_guaranteed:]  # ugh

        outputs = outputs[indices]  # not allowed as dynamic shape op, but ok with indices
        labels = labels[indices]
        # alternative:
        # outputs = torch.take_along_dim(outputs, indices.view(-1, 1), 0)
        # labels = torch.take(labels, indices)

        outputs = self.decoder(self.prediction_head(outputs))
        masked_lm_loss = self.loss_fn(outputs, labels)
        return masked_lm_loss

class ScriptableLMForSequenceClassification_modified(PreTrainedModel):
    """Classification head and pooler."""

    config_class = crammedBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)
        self.num_labels = self.cfg.num_labels

        self.encoder = ScriptableLM_modified(config)
        self.pooler = PoolingComponent(self.cfg.classification_head, self.cfg.hidden_size)
        self.head = torch.nn.Linear(self.cfg.classification_head.head_dim, self.num_labels)

        self.problem_type = None
        self._init_weights()

        self.count = 0
        self.x_list = []
        self.loss_list = []
        self.last_100_loss_list = []
        self.matmul_results = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.best_loss = 0
        os.makedirs(self.cfg.task_name)

    def _init_weights(self, module=None):
        modules = self.modules() if module is None else [module]
        for module in modules:
            _init_module(
                module,
                self.cfg.init.type,
                self.cfg.init.std,
                self.cfg.hidden_size,
                self.cfg.num_transformer_layers,
            )

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, **kwargs):
        encoder_output,  matmuls_from_enc = self.encoder(input_ids, attention_mask)
        
        logits = self.head(self.pooler(encoder_output))
        
        if labels is not None:
            if self.problem_type is None:  # very much from huggingface
                if self.num_labels == 1:
                    self.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.problem_type = "single_label_classification"
                else:
                    self.problem_type = "multi_label_classification"
            
            if self.problem_type == "regression":
                loss_fct = torch.nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.problem_type == "single_label_classification":
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.problem_type == "multi_label_classification":
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        else:
            loss = logits.new_zeros((1,))

        len_matmuls = len(matmuls_from_enc)
        
        self.count += 1
        self.x_list.append(self.count)
        self.loss_list.append(loss.item())
        
        if self.count < 100:
            last_100_loss = sum(self.loss_list) / len(self.loss_list)
            self.last_100_loss_list.append(last_100_loss)
            # print('\nLoss: {}, Last_100_losses: {}, Layers: {}, Count: {}'.format(masked_lm_loss.item(), last_100_loss, self.cfg.num_transformer_layers, self.count))
        else:
            last_100_loss = sum(self.loss_list[-100 :]) / len(self.loss_list[-100 :])
            self.last_100_loss_list.append(last_100_loss)
            if self.best_loss == 0 or last_100_loss < self.best_loss:
                self.best_loss = last_100_loss
            # print('\nLoss: {}, Last_100_losses: {}, Best_100_losses: {}, Layers: {}, Count: {}'.format(masked_lm_loss.item(), last_100_loss, self.best_loss, self.cfg.num_transformer_layers, self.count))
            
        for i in range(len_matmuls):
            norm_i = torch.norm(matmuls_from_enc[i], p=float('inf'))
            # print('Matmul_{}: {}'.format(i, norm_i.item()))
            self.matmul_results[i].append(norm_i.item())
             
            # # Impose a norm penalty    
            # ######################################
            # if norm_i > 450:
            #     loss += 0.1 * norm_i
            # print('Norm Penalty: O')
            # ######################################
            
        # Plot loss and matmuls  
        ######################################
        if self.count % 100 == 0:
            plt.plot(self.x_list, self.loss_list)
            plt.title('Loss')
            plt.xlabel('Steps')
            plt.savefig(os.path.join(self.cfg.task_name, 'losses.png'))
            plt.clf()
            plt.plot(self.x_list, self.last_100_loss_list)
            plt.title('Last 100 losses')
            plt.xlabel('Steps')
            plt.savefig(os.path.join(self.cfg.task_name, 'last_100_losses.png'))
            plt.clf()
            for i in range(len_matmuls):
                plt.subplot(7, 4, i + 1)
                plt.plot(self.x_list, self.matmul_results[i])
                plt.title('Matmul_{}'.format(i))
                plt.xlabel('Steps')
            plt.savefig(os.path.join(self.cfg.task_name, 'matmuls.png'))
            plt.clf()
        ######################################
                     
        return dict(logits=logits, loss=loss)
