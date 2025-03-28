from typing import List
import torch.nn.functional as F
from torch import nn
import torch
import numpy as np
from scipy.stats import ortho_group

SEED = 123


def tensor_reshape(x: torch.Tensor):
    if len(x.shape) == 2:
        x = x.reshape(-1, 1, x.shape[-1])
    elif len(x.shape) == 3:
        x = x.reshape(-1, x.shape[-1])

    return x


def get_ortho_tensor(dim=128, seed=SEED):
    a = np.float32(ortho_group.rvs(dim=dim, random_state=seed))
    ot = torch.Tensor(a)
    return ot


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1, input2):
        cosine_similarity = nn.functional.cosine_similarity(input1, input2, dim=1)
        loss = 1 - cosine_similarity
        return loss.mean()


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        # use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def set_dropout_prob(self, prob):
        self.dropout_prob = prob

    def forward(self, labels, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if use_dropout or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class CombineLinear(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 embed_dim: int,
                 output_dim: int = None,
                 ):
        super().__init__()
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        if output_dim is None:
            self.output_dim = latent_dim
        else:
            self.output_dim = output_dim

        self.combine = nn.Linear(latent_dim + embed_dim, self.output_dim)

    def forward(self, x, c):
        x_cat = torch.cat((x, c), dim=1)
        out = self.combine(x_cat)
        return out


class HiddenLayer(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 embed_dim: int,
                 output_dim: int,
                 dropout: float = 0.,
                 ):
        super().__init__()
        self.dp = nn.Dropout(p=dropout)
        self.comb = CombineLinear(hidden_dim, embed_dim, output_dim)
        self.ln = nn.LayerNorm(normalized_shape=output_dim)
        self.prelu = nn.PReLU()

    def forward(self, x, c):
        x = self.dp(x)
        x = self.comb(x, c)
        x = self.ln(x)
        x = self.prelu(x)

        return x


class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 128,
                 hidden_dim: List[int] = [1024, 1024],
                 dropout: float = 0.,
                 input_dropout: float = 0.,
                 residual: bool = True,
                 embed_dim: int = 256
                 ):
        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.ModuleList()
        self.residual = residual
        self.hidden_dim = hidden_dim

        for i in range(len(hidden_dim)):
            if i == 0:  # input layer
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=input_dropout),
                        nn.Linear(input_dim, hidden_dim[i]),
                        nn.LayerNorm(normalized_shape=hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
            else:  # hidden layers
                self.network.append(HiddenLayer(hidden_dim[i - 1], embed_dim, hidden_dim[i], dropout))

        self.network.append(
            CombineLinear(hidden_dim[-1], embed_dim, latent_dim)
        )

    def forward(self, x, c) -> F.Tensor:
        for i, layer in enumerate(self.network):
            if i == 0:
                x = layer(x)
            else:
                if self.residual and (0 < i < len(self.network) - 1) and self.hidden_dim[i - 1] == self.hidden_dim[i]:
                    x = layer(x, c) + x
                else:
                    x = layer(x, c)

        return F.normalize(x, p=2, dim=1)


class Encoder_cfg(nn.Module):
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 128,
                 hidden_dim: List[int] = [1024, 1024],
                 dropout: float = 0.,
                 input_dropout: float = 0.,
                 residual: bool = True,
                 embed_dim: int = 256
                 ):
        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.ModuleList()
        self.residual = residual
        self.hidden_dim = hidden_dim

        for i in range(len(hidden_dim)):
            if i == 0:  # input layer
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=input_dropout),
                        nn.Linear(input_dim, hidden_dim[i]),
                        nn.LayerNorm(normalized_shape=hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
            else:  # hidden layers
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                        nn.LayerNorm(normalized_shape=hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
        # output layer
        self.network.append(nn.Sequential(
            nn.Linear(hidden_dim[-1], latent_dim),
        ))

    def forward(self, x, c) -> F.Tensor:
        for i, layer in enumerate(self.network):
            if self.residual and (0 < i < len(self.network) - 1) and self.hidden_dim[i - 1] == self.hidden_dim[i]:
                x = layer(x) + x
            else:
                x = layer(x)

        return F.normalize(x, p=2, dim=1)


class Decoder(nn.Module):
    def __init__(
            self,
            input_dim: int,
            latent_dim: int = 128,
            hidden_dim: List[int] = [1024, 1024],
            dropout: float = 0.,
            input_dropout: float = 0.,
            residual: bool = True,
            embed_dim: int = 256
            ):
        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.ModuleList()
        self.residual = residual
        self.hidden_dim = hidden_dim

        for i in range(len(hidden_dim)):
            if i == 0:  # first hidden layer
                self.network.append(
                    HiddenLayer(latent_dim, embed_dim, hidden_dim[i], input_dropout)
                )
            else:  # other hidden layers
                self.network.append(
                    HiddenLayer(hidden_dim[i - 1], embed_dim, hidden_dim[i], dropout)
                )
        # reconstruction layer
        self.network.append(nn.Sequential(
            nn.Dropout(p=input_dropout),
            nn.Linear(hidden_dim[-1], input_dim)
        ))

        self.layer_num = len(self.network)

    def forward(self, x, c, test_layer=None):
        for i, layer in enumerate(self.network):
            if test_layer is not None and test_layer < i:
                break

            if i == self.layer_num - 1:
                x = layer(x)
            else:
                if self.residual and (0 < i < len(self.network) - 1) and self.hidden_dim[i - 1] == self.hidden_dim[i]:
                    x = layer(x, c) + x
                else:
                    x = layer(x, c)
        return x


class Decoder_cfg(nn.Module):
    def __init__(
            self,
            input_dim: int,
            latent_dim: int = 128,
            hidden_dim: List[int] = [1024, 1024],
            dropout: float = 0.,
            input_dropout: float = 0.,
            residual: bool = True,
            embed_dim: int = 256
            ):
        super().__init__()
        self.latent_dim = latent_dim
        self.network = nn.ModuleList()
        self.residual = residual
        self.hidden_dim = hidden_dim

        for i in range(len(hidden_dim)):
            if i == 0:  # first hidden layer
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=input_dropout),
                        nn.Linear(latent_dim, hidden_dim[i]),
                        nn.LayerNorm(normalized_shape=hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
            else:  # other hidden layers
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_dim[i - 1], hidden_dim[i]),
                        nn.LayerNorm(normalized_shape=hidden_dim[i]),
                        nn.PReLU(),
                    )
                )
        # reconstruction layer
        self.network.append(nn.Sequential(
            nn.Dropout(p=input_dropout),
            nn.Linear(hidden_dim[-1], input_dim)
        ))

    def forward(self, x, c, test_layer=None):
        for i, layer in enumerate(self.network):
            if test_layer is not None and test_layer < i:
                break

            if self.residual and (0 < i < len(self.network) - 1) and self.hidden_dim[i - 1] == self.hidden_dim[i]:
                x = layer(x) + x
            else:
                x = layer(x)
        return x


class AutoEncoder(nn.Module):
    """
    VAE base on compositional perturbation autoencoder (CPA)
    """

    def __init__(
            self,
            input_dim,
            hidden_dim=[14080, 14080],
            latent_dim=2048,
            input_dropout=0.4,
            dropout=0.5,
            residual=True,
            reshape=True,
            num_classes=100,
            embed_dim=128,
            class_dropout_prob=0.,
            cfg=True,
            para_devices=None,
            ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.input_dropout = input_dropout
        self.dropout = dropout
        self.residual = residual
        self.reshape = reshape
        self.cfg = cfg
        self.para_devices = para_devices
        self.label_dict = None
        if self.para_devices is not None:
            self.device0 = f"cuda:{para_devices[0]}"
            self.device1 = f"cuda:{para_devices[1]}"
        else:
            self.device0 = None
            self.device1 = None

        if cfg:
            self.encoder = Encoder_cfg(
                self.input_dim,
                self.latent_dim,
                self.hidden_dim,
                self.dropout,
                self.input_dropout,
                self.residual,
                embed_dim=embed_dim
            )
            self.decoder = Decoder_cfg(
                self.input_dim,
                self.latent_dim,
                self.hidden_dim,
                self.dropout,
                self.input_dropout,
                self.residual,
                embed_dim=embed_dim,
            )

        else:
            self.encoder = Encoder(
                self.input_dim,
                self.latent_dim,
                self.hidden_dim,
                self.dropout,
                self.input_dropout,
                self.residual,
                embed_dim=embed_dim
            )
            self.decoder = Decoder(
                self.input_dim,
                self.latent_dim,
                self.hidden_dim,
                self.dropout,
                self.input_dropout,
                self.residual,
                embed_dim=embed_dim,
            )

        self.y_embedder = LabelEmbedder(num_classes, embed_dim, class_dropout_prob)

        self.ortho_tensor = get_ortho_tensor(dim=latent_dim)[:num_classes + 1, :]

        if para_devices:
            self.para(para_devices)

    def para(self, para_devices):
        if para_devices is not None:
            device0 = f"cuda:{para_devices[0]}"
            device1 = f"cuda:{para_devices[1]}"
            self.encoder.to(device0)
            self.y_embedder.to(device0)
            self.decoder.to(device1)
            self.ortho_tensor = self.ortho_tensor.to(device0)
            self.device0 = f"cuda:{para_devices[0]}"
            self.device1 = f"cuda:{para_devices[1]}"

    def get_label_dict(self):
        return self.label_dict

    def set_label_dict(self, new_dict):
        self.label_dict = new_dict

    def set_reshape(self, need_reshape: bool):
        self.reshape = need_reshape

    def forward(self, x, y, enc=False, dec=False):
        if self.reshape:
            x = tensor_reshape(x)

        c = None
        if not self.cfg:
            c = self.y_embedder(y)
        if enc:
            if self.reshape:
                return tensor_reshape(self.encoder(x, c))
            return self.encoder(x, c)
        elif dec:
            if self.reshape:
                return tensor_reshape(self.decoder(x, c))
            return self.decoder(x, c)
        else:
            latent = self.encoder(x, c)
            if self.para_devices is not None:
                latent = latent.to(self.device1)
                if c is not None:
                    c = c.to(self.device1)
            recon = self.decoder(latent, c)
            if self.para_devices is not None:
                recon = recon.to(self.device0)
                latent = latent.to(self.device0)

            if self.reshape:
                return tensor_reshape(latent), tensor_reshape(recon)
            else:
                return latent, recon


if __name__ == '__main__':
    device = 'cuda'
    x = torch.randn((4, 1, 2000)).to(device)
    # c = torch.zeros((4, 1024))
    y = torch.IntTensor([1 for j in range(4)]).to(device)

    # le = LabelEmbedder(num_classes=100, hidden_size=256, dropout_prob=0.)
    # y = le(y_list[0])  # [4, 256]

    # x = tensor_reshape(x)
    # print(x.shape)
    # encoder = Encoder(
    #     input_dim=2000,
    #     latent_dim=128,
    #     hidden_dim=[1024, 1024],
    #     embed_dim=256,
    # )
    #
    # latent = encoder(x, y)
    #
    # decoder = Decoder(
    #     input_dim=2000,
    #     latent_dim=128,
    #     hidden_dim=[1024, 1024],
    #     embed_dim=256,
    # )
    #
    # recon = decoder(latent, y)

    ae = AutoEncoder(
        input_dim=2000,
        latent_dim=256,
        hidden_dim=[1024 for _ in range(2)],
        embed_dim=128,
        reshape=True,
        cfg=True,
        para_devices=[0, 0]
    )

    print(ae(x, y)[0].shape, ae(x, y)[1].shape)

    # params = list(oae.parameters())
    # k = 0
    # for i in params:
    #     l = 1
    #     print("该层的结构：" + str(list(i.size())))
    #     for j in i.size():
    #         l *= j
    #     print("该层参数和：" + str(l))
    #     k = k + l
    # print("总参数数量和：" + str(k))


