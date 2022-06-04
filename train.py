import os
import json
import mindspore
import mindspore.nn as nn
from tqdm import tqdm
from src.model import Encoder, Decoder, Img2Seq, TrainOneStepCell
from src.layers import CrossEntropyLoss
from src.dataset import create_dataset
from src.utils import AverageMeter, get_references, get_hypotheses
from mindspore import context
from nltk.translate.bleu_score import corpus_bleu

# context.set_context(mode=context.PYNATIVE_MODE)

# Model parameters
embed_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
mindspore.set_seed(0)

# Training parameters
start_epoch = 0
epochs = 10  # number of epochs to train for (if early stopping is not triggered)
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
val_dataset = create_dataset('/run/determined/workdir/coco_mindrecord/VAL_IMAGES_coco_5_cap_per_img_5_min_word_freq.mindrecord', 'val')

def train_one_epoch(model, train_dataset, epoch=0):
    model.set_train()
    total = train_dataset.get_dataset_size()
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy
    with tqdm(total=total) as t:
        t.set_description('Epoch %i' % epoch)
        for imgs, caps, caplens in train_dataset.create_tuple_iterator():
            loss, top5, cap_lens_sum = model(imgs, caps, caplens)
            losses.update(loss, cap_lens_sum)
            top5accs.update(top5, cap_lens_sum)
            t.set_postfix(loss='{loss.val} ({loss.avg})'.format(loss=losses),
                          top5='{top5.val} ({top5.avg})'.format(top5=top5accs))
            t.update(1)

def validation(model, val_dataset, epoch=0):
    model.set_train(False)
    total = val_dataset.get_dataset_size()
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    references = []
    hypotheses = []

    with tqdm(total=total) as t:
        t.set_description('Epoch %i' % epoch)
        for imgs, caps, caplens, all_captions in val_dataset.create_tuple_iterator():
            loss, predictions, top5, cap_lens_sum = model(imgs, caps, caplens, all_captions)
            losses.update(loss, cap_lens_sum)
            top5accs.update(top5, cap_lens_sum)
            t.set_postfix(loss='{loss.val} ({loss.avg})'.format(loss=losses),
                          top5='{top5.val} ({top5.avg})'.format(top5=top5accs))
            t.update(1)
            references.extend(get_references(all_captions.asnumpy(), word_map))
            hypotheses.extend(get_hypotheses(predictions.asnumpy(), caplens.asnumpy()-1))
        
            assert len(references) == len(hypotheses)

        bleu4 = corpus_bleu(references, hypotheses)
        print("BLEU-4 - {}\n".format(bleu4))

    return bleu4

for epoch in range(start_epoch, epochs):
    train_one_epoch(trainer, train_dataset, epoch)
    bleu4 = validation(model, val_dataset, epoch)