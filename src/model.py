import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from .resnet.resnet import resnet101
from .layers import AdaptiveAvgPool2d, Dense
from mindspore.common.initializer import initializer, Uniform, Zero

class Encoder(nn.Cell):
    def __init__(self, encoded_image_size=14):
        super().__init__()
        self.encoded_image_size = encoded_image_size
        resnet = resnet101(pretrained=True)

        modules = list(resnet.cells())[:-2]
        self.resnet = nn.SequentialCell(*modules)

        self.adaptive_pool = AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

    def construct(self, images):
        out = self.resnet(images)
        out = self.adaptive_pool(out)
        out = out.transpose(0, 2, 3, 1)
        return out

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
        self.f_beta= Dense(encoder_dim, decoder_dim)
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
        num_pixels = encoder_out.shape[1]

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
        
        predictions = ops.Stack(1)(predictions)
        alphas = ops.Stack(1)(alphas)

        return predictions, alphas

class Img2Seq(nn.Cell):
    def __init__(self, encoder, decoder, loss, alpha_c=1.0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss = loss
        self.alpha_c = alpha_c

    def construct(self, images, captions, caption_lengths):
        images = self.encoder(images)
        scores, alphas = self.decoder(images, captions, caption_lengths)
        loss = self.loss(scores, captions[:, 1:])
        # Add doubly stochastic attention regularization
        loss += self.alpha_c * ((1.0 - alphas.sum(axis=1)) ** 2).mean()
        return loss

