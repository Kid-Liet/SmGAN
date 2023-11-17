import hashlib
import pickle

import torch
from Crypto import Random
from Crypto.Cipher import AES

MODEL_KEY = 'AccuLearning Model Key, Manteia Technologies Co.,Ltd., ZhangWei, written on 2021.07.21'


class AESCipher(object):

    def __init__(self, _key=MODEL_KEY):
        self.bs = 32
        self.key = hashlib.sha256(_key.encode()).digest()

    def encrypt(self, raw):
        raw = self._pad(raw)
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return (iv + cipher.encrypt(raw))

    def decrypt(self, enc):
        iv = enc[:AES.block_size]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return self._unpad(cipher.decrypt(enc[AES.block_size:]))

    def _pad(self, s):
        pad_str = ((self.bs - len(s) % self.bs) *
                   chr(self.bs - len(s) % self.bs)).encode('utf-8')
        return s + pad_str

    @staticmethod
    def _unpad(s):
        return s[:-ord(s[len(s)-1:])]


def encrypt_and_save(state_dict, path):
    cipher = AESCipher()
    nodes_binary_str = pickle.dumps(state_dict)
    nodes_binary_str = cipher.encrypt(nodes_binary_str)
    with open(path, 'wb') as f:
        f.write(nodes_binary_str)


def decrypt_load_from_path(path):
    cipher = AESCipher()
    with open(path, 'rb') as f:
        nodes_binary_str = f.read()
    nodes_binary_str = cipher.decrypt(nodes_binary_str)
    return pickle.loads(nodes_binary_str)


if __name__ == '__main__':

    ckpt = torch.load(r'D:/IMSE-img2img/imse-multimodal2ct-reference/weight/net_new_ct1ct2_others_aug2.pth')
    cipher = AESCipher()
    nodes_binary_str = pickle.dumps(ckpt)
    nodes_binary_str = cipher.encrypt(nodes_binary_str)
    with open(r'D:/IMSE-img2img/imse-multimodal2ct-reference/weight/IMSE_Ref_2ct_others.pth.e', 'wb') as f:
        f.write(nodes_binary_str)
    fff
    with open(r'F:\tmp\AL_api_test\experiment_dir\model\best.pth.e', 'rb') as f:
        nodes_binary_str = f.read()
    nodes_binary_str = cipher.decrypt(nodes_binary_str)
    ckpt_new = pickle.loads(nodes_binary_str)
