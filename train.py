import os
import json
import mindspore
import mindspore.nn as nn
from tqdm import tqdm
from src.model import Encoder, Decoder, Img2Seq, TrainOneStepCell
from src.layers import CrossEntropyLoss
from src.dataset import create_dataset
from mindspore import context

# context.set_context(mode=context.PYNATIVE_MODE)

# Model parameters
embed_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
mindspore.set_seed(0)

# Training parameters
start_epoch = 0
epochs = 120  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 32
workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?
checkpoint = None  # path to checkpoint, None if none

# Data parameters
data_folder = './data/coco_mindrecord'  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
# Read word map
word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
with open(word_map_file, 'r') as j:
    word_map = json.load(j)

encoder = Encoder()
decoder = Decoder(len(word_map), embed_dim, 2048, decoder_dim, attention_dim, dropout)
loss = CrossEntropyLoss(ignore_index=word_map['<pad>'])
model = Img2Seq(encoder, decoder, loss)

encoder.fine_tune(fine_tune_encoder)
encoder_optimizer = nn.Adam(encoder.trainable_params(), learning_rate=encoder_lr) \
                    if fine_tune_encoder else None
decoder_optimizer = nn.Adam(params=decoder.trainable_params(),learning_rate=decoder_lr)

trainer = TrainOneStepCell(model, encoder_optimizer, decoder_optimizer, grad_clip)

train_dataset = create_dataset('/run/determined/workdir/coco_mindrecord/TRAIN_IMAGES_coco_5_cap_per_img_5_min_word_freq.mindrecord')

def train_one_epoch(model, train_dataset, epoch=0):
    model.set_train()
    total = train_dataset.get_dataset_size()
    loss_total = 0
    step_total = 0
    with tqdm(total=total) as t:
        t.set_description('Epoch %i' % epoch)
        for i in train_dataset.create_tuple_iterator():
            loss = model(*i)
            loss_total += loss.asnumpy()
            step_total += 1
            t.set_postfix(loss=loss_total/step_total)
            t.update(1)

train_one_epoch(trainer, train_dataset)
