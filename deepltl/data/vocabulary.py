
# pylint: disable = line-too-long

START_TOKEN = '<start>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'


class CharacterVocabulary():
    """Character level vocabulary that simply maps every character to an integer"""

    def __init__(self, vocab_list):
        """Expects a list of characters that can also contain special tokens (START_TOKEN, EOS_TOKEN, PAD_TOKEN). The index of each element specifies the integer the element is mapped to."""
        self.vocab = vocab_list
        self.start_id = self.vocab.index(START_TOKEN) if START_TOKEN in self.vocab else None
        self.eos_id = self.vocab.index(EOS_TOKEN) if EOS_TOKEN in self.vocab else None
        self.pad_id = self.vocab.index(PAD_TOKEN) if PAD_TOKEN in self.vocab else None

    def encode(self, s, prepend_start_token=True):
        """Encodes a string into a list of integers"""
        if isinstance(s, str):
            s = s.rstrip()
        encoded = [] if (not prepend_start_token) or self.start_id is None else [self.start_id]
        encoded += [self.vocab.index(c) for c in s]
        return encoded if self.eos_id is None else encoded + [self.eos_id]

    def decode(self, ids, as_list=False):
        """Decodes a list of integers into a string"""
        if ids[0] == self.start_id:
            ids = ids[1:]
        if self.eos_id in ids:
            ids = ids[:ids.index(self.eos_id)]
        elif self.pad_id in ids:
            ids = ids[:ids.index(self.pad_id)]
        res = [self.vocab[id] for id in ids]
        return res if as_list else ''.join(res)

    def vocab_size(self):
        return len(self.vocab)


class LTLVocabulary(CharacterVocabulary):

    def __init__(self, aps, consts, ops, start=False, eos=True, pad=True):
        vocab_list = [PAD_TOKEN] if pad else[]
        vocab_list += aps
        vocab_list += consts
        vocab_list += ops
        if start:
            vocab_list += [START_TOKEN]
        if eos:
            vocab_list += [EOS_TOKEN]
        super().__init__(vocab_list)


class TraceVocabulary(CharacterVocabulary):

    def __init__(self, aps, consts, ops, special=[';', '{', '}'], start=True, eos=True, pad=True):
        vocab_list = [PAD_TOKEN] if pad else[]
        vocab_list += aps
        vocab_list += consts
        vocab_list += ops
        vocab_list += special
        if start:
            vocab_list += [START_TOKEN]
        if eos:
            vocab_list += [EOS_TOKEN]
        super().__init__(vocab_list)
