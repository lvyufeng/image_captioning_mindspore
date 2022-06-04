class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_references(all_captions, word_map):
    references = []
    for j in range(all_captions.shape[0]):
        img_caps = all_captions[j].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
        references.append(img_captions)
    return references

def get_hypotheses(predictions, cap_lens):
    # Hypotheses
    preds = predictions.tolist()
    temp_preds = []
    for j, p in enumerate(preds):
        temp_preds.append(preds[j][:cap_lens[j]])  # remove pads
    return temp_preds
