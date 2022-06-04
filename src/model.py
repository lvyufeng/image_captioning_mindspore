import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from .resnet.resnet import resnet101
from .layers import AdaptiveAvgPool2d, Dense
from mindspore.common.initializer import initializer, Uniform, Zero
from mindspore import Tensor, Parameter
from mindspore.ops import constexpr

@constexpr
def arange(start, stop, step, dtype):
    return Tensor(np.arange(start, stop, step), dtype)

def sequence_mask(lengths, maxlen):
    """generate mask matrix by seq_length"""
    range_vector = arange(0, maxlen, 1, lengths.dtype)
    result = range_vector < lengths.view(lengths.shape + (1,))
    return result.astype(lengths.dtype)

def select_by_mask(inputs, mask):
    """mask hiddens by mask matrix"""
    return mask.view(mask.shape + (1,)).expand_as(inputs).astype(mindspore.bool_)  * inputs

def clip_grad(clip_value, grad):
    return ops.clip_by_value(grad, ops.scalar_to_tensor(-clip_value, grad.dtype),
                             ops.scalar_to_tensor(clip_value, grad.dtype))
class Encoder(nn.Cell):
    def __init__(self, encoded_image_size=14):
        super().__init__()
        self.encoded_image_size = encoded_image_size
        resnet = resnet101(pretrained=True)

        modules = list(resnet.cells())[:-2]
        self.resnet = nn.SequentialCell(*modules)

        self.adaptive_pool = AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune()

    def construct(self, images):
        out = self.resnet(images)
        out = self.adaptive_pool(out)
        out = out.transpose(0, 2, 3, 1)
        return out
    
    def fine_tune(self, fine_tune=True):
        for p in self.resnet.get_parameters():
            p.requires_grad = False
        
        for c in list(self.resnet.cells())[5:]:
            for p in c.get_parameters():
                p.requires_grad = fine_tune

class Attention(nn.Cell):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = Dense(encoder_dim, attention_dim)
        self.decoder_att = Dense(decoder_dim, attention_dim)
        self.full_att = Dense(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(axis=1)

    def construct(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.expand_dims(1))).squeeze(2)
        alpha = self.softmax(att)
        att_weight_encode = (encoder_out * alpha.expand_dims(2)).sum(axis=1)
        return att_weight_encode, alpha

class Decoder(nn.Cell):
    def __init__(self, vocab_size, embed_dim, encoder_dim, decoder_dim, attention_dim, dropout=0.5):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim
        self.vocab_size = vocab_size

        self.dropout = nn.Dropout(1-dropout)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.decode_cell = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)

        self.init_h = Dense(encoder_dim, decoder_dim)
        self.init_c = Dense(encoder_dim, decoder_dim)
        self.f_beta= Dense(decoder_dim, encoder_dim)
        self.simgoid = nn.Sigmoid()
        self.fc = Dense(decoder_dim, vocab_size)
        self.init_weights()
    
    def init_weights(self):
        self.embedding.embedding_table.set_data(initializer(Uniform(0.1), self.embedding.embedding_table.shape))
        self.fc.weight.set_data(initializer(Uniform(0.1), self.fc.weight.shape))
        self.fc.bias.set_data(initializer(Zero(), self.fc.bias.shape))

    def init_hidden_state(self, encoder_out):
        encoder_out = encoder_out.mean(axis=1)
        h = self.init_h(encoder_out)
        c = self.init_c(encoder_out)
        return h, c

    def construct(self, encoder_out, captions, caption_lengths):
        batch_size, max_caption_length = captions.shape

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, self.encoder_dim)

        # Embedding
        embeddings = self.embedding(captions)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)

        predictions = []
        alphas = []

        for t in range(max_caption_length):
            attn_weight_encode, alpha = self.attention(encoder_out, h)
            gate = self.simgoid(self.f_beta(h)) # gating scalar
            attn_weight_encode = gate * attn_weight_encode
            h, c = self.decode_cell(ops.Concat(1)([embeddings[:, t, :], attn_weight_encode]),
                                    (h, c))
            preds = self.fc(self.dropout(h))
            predictions.append(preds)
            alphas.append(alpha)

        decode_cap_lens = caption_lengths - 1
        mask = sequence_mask(decode_cap_lens, captions.shape[1])
        predictions = ops.Stack(1)(predictions)
        predictions = select_by_mask(predictions, mask)
        alphas = ops.Stack(1)(alphas)
        alphas = select_by_mask(alphas, mask)

        return predictions, alphas, decode_cap_lens

class Img2Seq(nn.Cell):
    def __init__(self, encoder, decoder, loss, alpha_c=1.0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss = loss
        self.alpha_c = alpha_c

    def construct(self, images, captions, caption_lengths, all_captions=None):
        images = self.encoder(images)
        scores, alphas, decode_cap_lens = self.decoder(images, captions, caption_lengths)
        loss = self.loss(scores[:, :-1].swapaxes(1, 2), captions[:, 1:])
        # Add doubly stochastic attention regularization
        loss += self.alpha_c * ((1.0 - alphas.sum(axis=1)) ** 2).mean()
        # Compute Top5 accuracy
        top5 = self.accuracy(scores[:, :-1], captions[:, 1:], 5, decode_cap_lens)
        if all_captions is not None:
            predictions = scores.argmax(axis=2)
            return loss, predictions, top5, decode_cap_lens.sum()
        return loss, ops.stop_gradient(top5), ops.stop_gradient(decode_cap_lens.sum())

    def accuracy(self, scores, targets, k, cap_lens):
        _, ind = ops.TopK()(scores, k)
        mask = sequence_mask(cap_lens, targets.shape[1])
        correct = ops.equal(ind, targets.expand_dims(2).expand_as(ind))
        correct = select_by_mask(correct.astype(mindspore.float32), mask)
        correct_total = correct.sum()
        return correct_total / cap_lens.sum() * 100

class TrainOneStepCell(nn.Cell):
    def __init__(self, network, encoder_optimizer, decoder_optimizer, grad_clip=None):
        super(TrainOneStepCell, self).__init__()
        self.network = network
        self.network.set_grad()
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.grad_clip = grad_clip
        if self.encoder_optimizer is not None:
            self.weights = self.encoder_optimizer.parameters + self.decoder_optimizer.parameters
            self.grad_split = len(encoder_optimizer.parameters)
        else:
            self.weights = self.decoder_optimizer.parameters
            self.grad_split = 0

        self.grad = ops.GradOperation(get_by_list=True)
        self.hyper_map = ops.HyperMap()

    def construct(self, *inputs):
        loss, top5, cap_lens_sum = self.network(*inputs)
        grads = self.grad(self.network, self.weights)(*inputs)
        if self.grad_clip is not None:
            grads = self.hyper_map(ops.partial(clip_grad, self.grad_clip), grads)

        if self.encoder_optimizer is not None:
            self.encoder_optimizer(grads[:self.grad_split])
            self.decoder_optimizer(grads[self.grad_split:])
        else:
            self.decoder_optimizer(grads)
        return loss, top5, cap_lens_sum
