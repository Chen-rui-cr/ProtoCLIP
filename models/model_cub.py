from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
from clip.clip_2 import load, tokenize
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from utils.utils import build_cosine_scheduler, cosine_loss, AccRecord

_tokenizer = _Tokenizer()

class calibrated_stacking():
    def __init__(self, test_seen_label, all_label_num, lam):
        self.test_seen_label = list(set(test_seen_label.numpy()))
        # self.test_unseen_label = [idx for idx in range(all_label_num) if idx not in self.test_seen_label]
        self.lam = lam

    def call(self, output):
        """
            output: the output predicted score of size batchsize * 200
            lam: the parameter to control the output score of seen classes.
            self.test_seen_label
            self.test_unseen_label
            :return
        """
        output = output.cpu().numpy()
        # print(np.min(output[:, self.test_seen_label]), np.max(output[:, self.test_seen_label]))
        output[:, self.test_seen_label] = output[:, self.test_seen_label] - self.lam
        return torch.from_numpy(output).cuda()


class PromptLearner(nn.Module):
    def __init__(self, args, attr_name, clip_model, text_encoder=None, n_ctx=36):
        super().__init__()
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.dtype = clip_model.dtype
        self.clip_model = clip_model
        self.args = args
        n_cls = len(attr_name)

        prompt_prefix = ' '.join(['x'] * n_ctx)
        attr_name = [name.replace('_', ' ') for name in attr_name]
        prompts = [prompt_prefix + ' ' + name + '.' for name in attr_name]
        self.name_lens = [len(_tokenizer.encode(name)) for name in attr_name]

        text_prompt = torch.empty(n_cls, n_ctx, ctx_dim, dtype=self.dtype).cuda()
        nn.init.normal_(text_prompt, std=0.02)
        text_prompt = nn.Parameter(text_prompt)
        self.text_prompt = text_prompt  # 可学习

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts])  # 输入一串提示词，输出维度 [n，77]
        self.tokenized_prompts = tokenized_prompts
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts.cuda()).type(self.dtype)  # 嵌入表示 [n, 77, 768]
        self.register_buffer('token_prefix', embedding[:, :1, :])
        self.register_buffer('token_suffix', embedding[:, 1 + (n_ctx):, :])

        # tokenized_prompts = torch.cat([tokenize(p) for p in ["a photo of bird"]])  # 输入一串提示词，输出维度 [n，77]
        # self.tokenized_style_prompts = tokenized_prompts
        # with torch.no_grad():
        #     embedding = clip_model.token_embedding(tokenized_prompts.cuda()).type(self.dtype)  # 嵌入表示 [n, 77, 768]
        # self.style_embedding = embedding.expand(128, -1, -1)
        # self.register_buffer('style_token_prefix', embedding[:, :5, :])
        # self.register_buffer('style_token_suffix', embedding[:, 5:, :])

        # 属性 tokenized
        # self.attr_tokenized_prompts = torch.cat([tokenize(p) for p in attr])  # 输入一串提示词，输出维度 [n，77]
        # with torch.no_grad():
        #     prompts = clip_model.token_embedding(self.attr_tokenized_prompts.cuda()).type(self.dtype)  # 嵌入表示 [n, 77, 768]
        #     self.text_prompt = text_encoder(prompts, self.attr_tokenized_prompts)  # [312, 768]
        #     self.text_prompt = self.text_prompt / self.text_prompt.norm(dim=-1, keepdim=True)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.ctx_dim = ctx_dim

    def forward(self):
        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                self.text_prompt,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, x, tokenized_prompts):
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class ChannelLayerNorm(torch.nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True, unbiased=False)

        return (x - mean) / torch.sqrt(var + self.eps)


class CLIP(nn.Module):
    def __init__(self, args, clip_model, attr, class_names, label_attr_sentence):
        super().__init__()
        self.n_class = len(class_names)
        self.n_attr = len(attr)
        self.args = args
        self.attr = attr

        if args.arch == "RN101":
            self.kernel = 7
            self.feat_dim = 2048
            self.text_dim = 512
        elif args.arch == "RN50x64":
            self.kernel = 14
            self.feat_dim = 4096
            self.text_dim = 1024

        if args.db_name == "CUB":
            self.attr_dim = 312
        elif args.db_name == "SUN":
            self.attr_dim = 102
        elif args.db_name == "AWA2":
            self.attr_dim = 85

        # text enoder
        self.text_encoder = TextEncoder(clip_model)
        # if torch.cuda.device_count() > 1:
        #     self.text_encoder = nn.DataParallel(self.text_encoder)

        self.prompt_learner = PromptLearner(self.args, attr, clip_model, self.text_encoder, n_ctx=args.n_ctx)

        # self.text_quary = nn.Parameter(self.prompt_learner.text_prompt.clone().unsqueeze(0), requires_grad=False).cuda()
        # self.text_quary.data = 2e-4 * (self.text_quary.data / self.text_quary.norm(dim=-1, keepdim=True))

        self.text_quary = nn.Parameter(2e-4 * torch.randn([self.attr_dim, self.text_dim], device="cuda").half(), requires_grad=True)
        # self.text_quary = nn.Parameter(2e-4 * torch.randn([self.attr_dim, self.feat_dim], device="cuda").half(), requires_grad=True)
        self.num_attr = self.n_attr

        # image encoder
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale

        self.cross_attention = nn.MultiheadAttention(embed_dim=self.text_dim, num_heads=8, dtype=clip_model.dtype).cuda()
        width = self.text_dim
        scale = width ** -0.5
        # self.class_embedding = nn.Parameter(scale * torch.randn(width, device="cuda").half(), requires_grad=True)
        # self.positional_embedding = nn.Parameter(scale * torch.randn(self.attr_dim + 1, width, device="cuda").half(), requires_grad=True)  # [rid**2+1, width]
        # self.ln_pre_1 = LayerNorm(width, elementwise_affine=False).cuda()
        self.ln_pre_1 = ChannelLayerNorm().cuda()
        # self.ln_pre_1.track_running_stats = False
        self.ln_pre_2 = LayerNorm(width).cuda()
        # self.ln_pre_2.track_running_stats = False

        # self.train_attr_idx = train_attr_idx
        # self.test_attr_idx = test_attr_idx

        self.mapping = nn.Sequential(nn.Linear(self.text_dim, 1),
                                     # nn.GELU(),
                                     # nn.Linear(4*self.text_dim, 1)
                                     )
        self.mapping = self.mapping.half()
        self.mapping.apply(self.init_weights)
        self.mapping.cuda()
        # self.W = nn.Linear(self.feat_dim, self.text_dim, bias=True)
        # self.W.weight.data = torch.eye(self.feat_dim)
        # self.W.bias.data = torch.zeros(self.text_dim)
        # self.W = self.W.half().cuda()

        # self.W = nn.Parameter(torch.eye(self.feat_dim, device="cuda")[:, :self.text_dim].half(), requires_grad=True)
        # self.W = nn.Parameter(torch.eye(self.text_dim, device="cuda").half(), requires_grad=True)
        # self.W_t = nn.Parameter(torch.eye(self.text_dim, device="cuda").half(), requires_grad=True)

        self.img2style = nn.Sequential(nn.Linear(self.text_dim, self.text_dim//16),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(self.text_dim//16, self.text_dim))
        self.img2style.half()
        self.img2style.cuda()

        self.adapter = nn.Parameter(torch.randn([self.text_dim, self.feat_dim, 1, 1], device="cuda").half(), requires_grad=True)

        self.cos2score = nn.Sequential(nn.Linear(self.attr_dim, self.attr_dim),
                                     # nn.GELU(),
                                     # nn.Linear(4*self.attr_dim, self.attr_dim)
                                     )
        self.cos2score.apply(self.init_weights)
        self.cos2score = self.cos2score.half()
        self.cos2score.cuda()

        tokenized_prompts = torch.cat([tokenize(p, truncate=True) for p in label_attr_sentence])  # 输入一串提示词，输出维度 [n，77]
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts.cuda()).type(self.dtype)  # 嵌入表示 [n, 77, 768]
            text_prompt_nature = self.text_encoder(embedding, tokenized_prompts)  # [312, 768]
        self.class_name_emb = text_prompt_nature / text_prompt_nature.norm(dim=-1, keepdim=True)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)

    def forward(self, image, label=None, a_index=None, epoch=0, test=False):
        if test:
            b, c, h, w = image.shape
            with torch.no_grad():
                patch_features, feat_pool = self.image_encoder(image.type(self.dtype))  # [b, 512, 49]
                patch_features = self.ln_pre_1(patch_features)
                # patch_features = patch_features.permute(1, 2, 0)
                # patch_features = patch_features / patch_features.norm(dim=1, keepdim=True)
                patch_features = patch_features.detach()

            bias = self.img2style(feat_pool).unsqueeze(-1).unsqueeze(-1)

            patch_features = F.conv2d(input=patch_features, weight=self.adapter)  # [b, 1024, 14, 14]
            patch_features = patch_features + bias
            patch_features = patch_features / patch_features.norm(dim=1, keepdim=True)

            text_prompt = self.prompt_learner()
            quary = self.text_encoder(text_prompt, self.prompt_learner.tokenized_prompts)
            # quary = self.text_quary
            quary = quary / quary.norm(dim=1, keepdim=True)
            patch_cos_sim = F.conv2d(input=patch_features, weight=quary.unsqueeze(-1).unsqueeze(-1))  # / (norm1 * norm2)   # [b, 312, 14, 14]
            output = F.max_pool2d(patch_cos_sim, kernel_size=self.kernel).view(b, -1)  # [b, 312]
            output = self.cos2score(output)
            # output = output / output.norm(dim=1, keepdim=True)

            return output

        else:
            b, c, h, w = image.shape
            with torch.no_grad():
                patch_features, feat_pool = self.image_encoder(image.type(self.dtype))  # [b, 2048, 7, 7]
                patch_features = self.ln_pre_1(patch_features)
                # patch_features = patch_features.permute(1, 2, 0)
                # patch_features = patch_features / patch_features.norm(dim=1, keepdim=True)
                patch_features = patch_features.detach()

            bias = self.img2style(feat_pool).unsqueeze(-1).unsqueeze(-1)

            patch_features = F.conv2d(input=patch_features, weight=self.adapter)  # [b, 512, 7, 7]
            patch_features = patch_features + bias
            patch_features = patch_features / patch_features.norm(dim=1, keepdim=True)

            align_text = self.class_name_emb[label]
            loss_align = 1 - F.cosine_similarity(F.avg_pool2d(input=patch_features, kernel_size=self.kernel).squeeze(-1).squeeze(-1), align_text, dim=1).mean()
            # loss_align = F.mse_loss(F.avg_pool2d(input=patch_features, kernel_size=self.kernel).squeeze(-1).squeeze(-1), feat_pool/feat_pool.norm(dim=-1, keepdim=True))

            text_prompt = self.prompt_learner() # [b, 312, 77, 512]
            quary = self.text_encoder(text_prompt, self.prompt_learner.tokenized_prompts)
            # quary = self.text_quary
            quary = quary / quary.norm(dim=1, keepdim=True)

            patch_cos_sim = F.conv2d(input=patch_features, weight=quary.unsqueeze(-1).unsqueeze(-1))  # / (norm1 * norm2)   # [b, 312, 14, 14]
            output = F.max_pool2d(patch_cos_sim, kernel_size=self.kernel).view(b, -1)  # [b, 312]
            output = self.cos2score(output)

            return output, loss_align

    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype


class CoOp:
    def __init__(self, args, attr, train_class_name, test_class_name, label_attr_sentence, all_label_num, test_seen_label,
                 n_ctx=12, use_float32=False, use_grad_checkpoint=False):
        clip_model, _ = load(args.ckpt_path)
        clip_model.eval()
        if use_float32:
            clip_model.float()
        self.clip_model = clip_model
        self.use_grad_checkpoint = use_grad_checkpoint

        self.lr = args.lr * args.train_batch / 20
        self.wd = args.wd
        self.epochs = args.epochs
        self.train_batch = args.train_batch
        self.args = args
        self.dtype = clip_model.dtype

        self.model = CLIP(self.args, self.clip_model, attr, train_class_name, label_attr_sentence)

        self.calibrated_stacking = calibrated_stacking(test_seen_label, all_label_num, lam=0.7)

    def fit(self, train_loader, test_zsl_dl, test_seen_dl, test_unseen_dl, attribute_seen, attribute_zsl,
            attribute_gzsl):
        per_epoch_steps = len(train_loader)
        param_dict = [{'params': self.model.prompt_learner.text_prompt},
                      # {'params': self.model.text_quary},
                    {'params': self.model.img2style.parameters()},
                    {'params': self.model.cos2score.parameters()},
                    {'params': self.model.adapter},
        ]

        self.optimizer = torch.optim.SGD(param_dict, lr=self.lr, weight_decay=self.wd, momentum=0.9)
        self.scheduler = build_cosine_scheduler(
            self.optimizer,
            lr=self.lr,
            total_step=self.epochs * per_epoch_steps)

        self.model.eval()
        acc_record = AccRecord()

        for epoch in range(self.epochs):
            with tqdm(train_loader, unit="it") as tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                for idx, (x, y) in enumerate(tepoch):
                    cur_iter_idx = epoch * per_epoch_steps + idx
                    self.scheduler.step(cur_iter_idx)

                    output, loss_align = self.model(x.cuda(), y.cuda(), epoch=epoch)

                    loss_mse = nn.MSELoss()(output, attribute_seen[y])
                    loss_cls = F.cross_entropy(output.mm(attribute_seen.T), y.cuda())

                    loss = loss_cls + 0.5 * loss_mse + 0.5 * loss_align
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    tepoch.set_postfix(cls=f"{loss_cls.item():.2f}", mse=f"{loss_mse.item():.2f}", align=f"{loss_align.item():.2f}")

                # zsl
                acc = self.accuracy(test_zsl_dl, attribute_zsl)
                acc_record.update(acc)
                acc_record.info()
                # if acc_record.save_zsl_model:
                #     torch.save(self.model.state_dict(), f"./best_zsl_{self.args.db_name}_model")
                #     acc_record.save_zsl_model = False

                acc_GZSL_unseen = self.accuracy(test_unseen_dl, attribute_gzsl, gzsl_flag=True)
                acc_GZSL_seen = self.accuracy(test_seen_dl, attribute_gzsl, gzsl_flag=True)
                acc_record.update([acc_GZSL_unseen, acc_GZSL_seen], zsl_flag=False)
                acc_record.info(zsl_flag=False)
                # if acc_record.save_gzsl_model:
                #     torch.save(self.model.state_dict(), f"./best_gzsl_{self.args.db_name}_model")
                #     acc_record.save_gzsl_model = False


    @torch.no_grad()
    def accuracy(self, loader, attribute, gzsl_flag=False):
        total_count = 0
        acc_count = 0
        for i, (x, y) in enumerate(tqdm(loader)):
            output = self.model(x.cuda(), test=True)
            output = output.mm(attribute.T)

            if gzsl_flag:
                output = self.calibrated_stacking.call(output)

            _, top_labels = output.topk(1, dim=-1)
            acc_count += (top_labels.view(-1) == y.cuda()).sum().cpu().numpy()
            total_count += y.shape[0]
        acc = acc_count * 1.0 / total_count
        acc = acc.item()
        return acc

    @torch.no_grad()
    def inference(self, image, test_class):
        logits = self.model(image, test_class, test=True)
        return logits.float().softmax(dim=-1)
