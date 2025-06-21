import methods.wideresnet as wideresnet
from methods.augtools import HighlyCustomizableAugment, RandAugmentMC
import methods.util as util
import tqdm
from sklearn import metrics
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
import random
from methods.util import AverageMeter
import time
from torchvision.transforms import transforms
from methods.resnet import ResNet
from torchvision import models as torchvision_models


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


class PretrainedResNet(nn.Module):

    def __init__(self, rawname, pretrain_path=None) -> None:
        super().__init__()
        if pretrain_path == 'default':
            self.model = torchvision_models.__dict__[rawname](pretrained=True)
            self.output_dim = self.model.fc.weight.shape[1]
            self.model.fc = nn.Identity()
        else:
            self.model = torchvision_models.__dict__[rawname]()
            self.output_dim = self.model.fc.weight.shape[1]
            self.model.fc = nn.Identity()
            if pretrain_path is not None:
                sd = torch.load(pretrain_path)
                self.model.load_state_dict(sd, strict=True)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        return x


class Backbone(nn.Module):

    def __init__(self, config, inchan):
        super().__init__()

        if config['backbone'] == 'wideresnet28-2':
            self.backbone = wideresnet.WideResNetBackbone(None, 28, 2, 0, config['category_model']['projection_dim'])
        elif config['backbone'] == 'wideresnet40-4':
            self.backbone = wideresnet.WideResNetBackbone(None, 40, 4, 0, config['category_model']['projection_dim'])
        elif config['backbone'] == 'wideresnet16-8':
            self.backbone = wideresnet.WideResNetBackbone(None, 16, 8, 0.4, config['category_model']['projection_dim'])
        elif config['backbone'] == 'wideresnet28-10':
            self.backbone = wideresnet.WideResNetBackbone(None, 28, 10, 0.3, config['category_model']['projection_dim'])
        elif config['backbone'] == 'resnet18':
            self.backbone = ResNet(output_dim=config['category_model']['projection_dim'], inchan=inchan)
        elif config['backbone'] == 'resnet18a':
            self.backbone = ResNet(output_dim=config['category_model']['projection_dim'], resfirststride=2,
                                   inchan=inchan)
        elif config['backbone'] == 'resnet18b':
            self.backbone = ResNet(output_dim=config['category_model']['projection_dim'], resfirststride=2,
                                   inchan=inchan)
        elif config['backbone'] == 'resnet34':
            self.backbone = ResNet(output_dim=config['category_model']['projection_dim'], num_block=[3, 4, 6, 3],
                                   inchan=inchan)
        elif config['backbone'] in ['prt_r18', 'prt_r34', 'prt_r50']:
            self.backbone = PretrainedResNet(
                {'prt_r18': 'resnet18', 'prt_r34': 'resnet34', 'prt_r50': 'resnet50'}[config['backbone']])
        elif config['backbone'] in ['prt_pytorchr18', 'prt_pytorchr34', 'prt_pytorchr50']:
            name, path = {
                'prt_pytorchr18': ('resnet18', 'default'),
                'prt_pytorchr34': ('resnet34', 'default'),
                'prt_pytorchr50': ('resnet50', 'default')
            }[config['backbone']]
            self.backbone = PretrainedResNet(name, path)
        elif config['backbone'] in ['prt_dinor18', 'prt_dinor34', 'prt_dinor50']:
            name, path = {
                'prt_dinor50': ('resnet50', './model_weights/dino_resnet50_pretrain.pth')
            }[config['backbone']]
            self.backbone = PretrainedResNet(name, path)
        else:
            bkb = config['backbone']
            raise Exception(f'Backbone \"{bkb}\" is not defined.')

        # types : ae_softmax_avg , ae_avg_softmax , avg_ae_softmax
        self.output_dim = self.backbone.output_dim
        # self.classifier = CRFClassifier(self.backbone.output_dim,numclss,config)

    def forward(self, x):
        x = self.backbone(x)
        # latent , global prob , logits
        return x


class LinearClassifier(nn.Module):

    def __init__(self, inchannels, num_class, config):
        super().__init__()
        self.gamma = config['gamma']
        self.cls = nn.Conv2d(inchannels, num_class, 1, padding=0, bias=False)

    def forward(self, x):
        x = self.cls(x)
        return x * self.gamma


def sim_conv_layer(input_channel, output_channel, kernel_size=1, padding=0, use_activation=True):
    if use_activation:
        res = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, padding=padding, bias=False),
            nn.Tanh())
    else:
        res = nn.Conv2d(input_channel, output_channel, kernel_size, padding=padding, bias=False)
    return res


class Encoder(nn.Module):

    def __init__(self, inchannel, hidden_layers, latent_chan):
        super().__init__()  # 调用父类构造函数，通常是nn.Module类
        layer_block = sim_conv_layer  # 定义卷积层的模块，假设 sim_conv_layer 已在其他地方定义
        self.latent_size = latent_chan  # 保存潜在通道数
        if latent_chan > 0:
            self.encode_convs = []
            for i in range(len(hidden_layers)):
                h = hidden_layers[i]
                ecv = layer_block(inchannel, h, )
                inchannel = h
                self.encode_convs.append(ecv)
            self.encode_convs = nn.ModuleList(self.encode_convs)
            hidd = 128
            self.extra_layer = layer_block(inchannel, hidd)  # 新增 128 维度
            self.latent_conv = layer_block(hidd, latent_chan)  # 128 → latent_chan

        else:
            self.center = nn.Parameter(torch.rand([inchannel, 1, 1]), True)

    def forward(self, x):
        if self.latent_size > 0:
            output = x
            for cv in self.encode_convs:
                output = cv(output)
            output = self.extra_layer(output)
            latent = self.latent_conv(output)
            return latent
        else:
            return self.center, self.center


class CSSRClassifier(nn.Module):

    def __init__(self, inchannels, num_class, config):
        super().__init__()
        ae_hidden = config['ae_hidden']
        ae_latent = config['ae_latent']
        ae_H_W = config['ae_H_W']
        self.class_enc = []
        for i in range(num_class):
            ae = Encoder(inchannels, ae_hidden, ae_latent)
            self.class_enc.append(ae)
        self.class_enc = nn.ModuleList(self.class_enc)
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.useL1 = config['error_measure'] == 'L1'
        self.reduction = -1 if config['model'] == 'pcssr' else 1
        self.reduction *= config['gamma']
        # 每个类别的原型，初始化为随机值
        self.prototypes = nn.Parameter(torch.randn(num_class, ae_latent, ae_H_W, ae_H_W))
        self.prototypes1 = nn.Parameter(torch.randn(num_class, ae_latent, ae_H_W, ae_H_W))
        self.safe_lr = config['safe_lr']

    def prototypes_error(self, lt, prototype):
        # 扩展 prototypes 的维度，以便与 lt 进行广播（broadcasting）
        prototype_expanded = prototype.unsqueeze(0)
        if self.useL1:
            return torch.norm(lt - prototype_expanded, p=1, dim=1, keepdim=True) * self.reduction
        else:
            return torch.norm(lt - prototype_expanded, p=2, dim=1, keepdim=True) ** 2 * self.reduction

    def prototypes_error1(self, lt, prototype1):
        # 扩展 prototypes 的维度，以便与 lt 进行广播（broadcasting）
        prototype_expanded1 = prototype1.unsqueeze(0)
        torch.norm(lt - prototype_expanded1, p=2, dim=1, keepdim=True) ** 2 * self.reduction
        if self.useL1:
            return torch.norm(lt - prototype_expanded1, p=1, dim=1, keepdim=True) * self.reduction
        else:
            return torch.norm(lt - prototype_expanded1, p=2, dim=1, keepdim=True) ** 2 * self.reduction


    clip_len = 100

    def calculate_pull_loss(self, cls_er_d, y, c, safe_mar,cls_er_d1):
        # 创建掩码，只保留标签等于 c 的样本
        mask = (y == c)
        # 将掩码应用到 cls_er_d，保留符合条件的样本，其它置为 0
        #filtered_cls_er_d1 = torch.sum(cls_er_d, dim=(2, 3)).squeeze() * mask  # 对应的样本位置保留，不符合条件的位置会被置为 0
        filtered_cls_er_d1 = torch.sum(cls_er_d1, dim=(2, 3)).squeeze() * mask
        loss_pull = torch.sum(filtered_cls_er_d1)
        if mask.sum() != 0:
            final_loss_pull = loss_pull / mask.sum()
        else:
            final_loss_pull = 0
        return final_loss_pull

    def calculate_push_loss(self, cls_er_d, y, c, safe_dis):
        mask = (y != c)
        filtered_cls_er_d = torch.sum(cls_er_d, dim=(2, 3)).squeeze()
        # print(self.safe_lr)
        mask_safe_distance1 = (filtered_cls_er_d < (10000)).float()
        combined_mask = (mask.bool() & mask_safe_distance1.bool()).float()
        # 计算损失
        loss_push = torch.sum(filtered_cls_er_d * combined_mask)
        # 计算符合条件的样本数
        num_samples_combine = combined_mask.sum()
        if num_samples_combine != 0:
            final_loss_push = loss_push / num_samples_combine
        else:
            final_loss_push = 0
        return final_loss_push

    def forward(self, x, ycls, safe_margin_all, safe_distance_all):
        cls_ers = []
        cls_ers1 = []
        for c in range(len(self.class_enc)):
            lt = self.class_enc[c](x)
            cls_er = self.prototypes_error(lt, self.prototypes[c])
            cls_er1 = self.prototypes_error1(lt, self.prototypes1[c])
            if CSSRClassifier.clip_len > 0:
                cls_er = torch.clamp(cls_er, -CSSRClassifier.clip_len, CSSRClassifier.clip_len)
            if CSSRClassifier.clip_len > 0:
                cls_er1 = torch.clamp(cls_er1, -CSSRClassifier.clip_len, CSSRClassifier.clip_len)
            cls_ers.append(cls_er)
            cls_ers1.append(cls_er1)
        pull_loss_t = torch.tensor(0)
        push_loss_t = torch.tensor(0)
        if self.training:
            for c in range(len(self.class_enc)):
                pull_loss = self.calculate_pull_loss(cls_ers[c], ycls, c, safe_margin_all[c],cls_ers1[c])
                push_loss = self.calculate_push_loss(cls_ers[c], ycls, c, safe_distance_all[c])
                pull_loss_t = pull_loss_t + pull_loss
                push_loss_t = push_loss_t + push_loss
        logits = torch.cat(cls_ers, dim=1)
        logits1 = torch.cat(cls_ers1, dim=1)
        lossp = [pull_loss_t, push_loss_t]
        return logits,logits1, lossp


class BackboneAndClassifier(nn.Module):

    def __init__(self, num_classes, config):
        super().__init__()
        clsblock = {'linear': LinearClassifier, 'pcssr': CSSRClassifier, 'rcssr': CSSRClassifier}
        self.backbone = Backbone(config, 3)
        cat_config = config['category_model']
        self.cat_cls = clsblock[cat_config['model']](self.backbone.output_dim, num_classes, cat_config)

    def forward(self, x, ycls, safe_margin_all, safe_distance_all, feature_only=False):
        x = self.backbone(x)
        if feature_only:
            return x
        logic,logic1, lossp = self.cat_cls(x, ycls, safe_margin_all, safe_distance_all)
        return x, logic,logic1, lossp


class CSSRModel(nn.Module):

    def __init__(self, num_classes, config, crt):
        super().__init__()
        self.crt = crt
        # ------ New Arch
        self.backbone_cs = BackboneAndClassifier(num_classes, config)
        self.config = config
        self.mins = {i: [] for i in range(num_classes)}
        self.maxs = {i: [] for i in range(num_classes)}
        self.num_classes = num_classes

    powers = [8]

    def forward(self, x, ycls=None, safe_margin_all=None, safe_distance_all=None, reqpredauc=False, prepareTest=False,
                reqfeature=False):

        # ----- New Arch
        x, xcls_raw,xcls_raw1, lossp = self.backbone_cs(x, ycls, safe_margin_all, safe_distance_all, feature_only=reqfeature)
        if reqfeature:
            return x

        def pred_score(xcls):
            score_reduce = lambda x: x.reshape([x.shape[0], -1]).mean(axis=1)
            x_detach = x.detach()
            probs = self.crt(xcls, prob=True).cpu().numpy()
            pred = probs.argmax(axis=1)
            rep_scores = torch.abs(x_detach).mean(dim=1).cpu().numpy()
            return pred, score_reduce(rep_scores)

        if self.training:
            xcloss = self.crt(xcls_raw, ycls)
            if reqpredauc:
                pred, score = pred_score(xcls_raw.detach())
                return xcloss, lossp, pred, score, xcls_raw
        else:
            xcls = xcls_raw
            xcls1 = xcls_raw1
            if reqpredauc:
                pred, score = pred_score(xcls)
                deviations = None
                return pred, score, deviations

        return xcls,xcls1


class CSSRCriterion(nn.Module):
    def get_onehot_label(self, y, clsnum):
        y = torch.reshape(y, [-1, 1]).long()
        return torch.zeros(y.shape[0], clsnum).cuda().scatter_(1, y, 1)

    def __init__(self, avg_order, enable_sigma=True):
        super().__init__()
        self.avg_order = {"avg_softmax": 1, "softmax_avg": 2}[avg_order]
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.enable_sigma = enable_sigma

    def forward(self, x, y=None, prob=False, pred=False):
        if self.avg_order == 1:
            g = self.avg_pool(x).view(x.shape[0], -1)
            g = torch.softmax(g, dim=1)
        elif self.avg_order == 2:
            g = torch.softmax(x, dim=1)
            g = self.avg_pool(g).view(x.size(0), -1)
        if prob: return g
        if pred: return torch.argmax(g, dim=1)
        loss = -torch.sum(self.get_onehot_label(y, g.shape[1]) * torch.log(g), dim=1).mean()
        return loss


def manual_contrast(x):
    s = random.uniform(0.1, 2)
    return x * s


class WrapDataset(data.Dataset):

    def __init__(self, labeled_ds, config, inchan_num=3) -> None:
        super().__init__()
        self.labeled_ds = labeled_ds

        __mean = [0.5, 0.5, 0.5][:inchan_num]
        __std = [0.25, 0.25, 0.25][:inchan_num]

        trans = [transforms.RandomHorizontalFlip()]
        if config['cust_aug_crop_withresize']:
            trans.append(transforms.RandomResizedCrop(size=util.img_size, scale=(0.25, 1)))
        elif util.img_size > 200:
            trans += [transforms.Resize(256), transforms.RandomResizedCrop(util.img_size)]
        else:
            trans.append(transforms.RandomCrop(size=util.img_size,
                                               padding=int(util.img_size * 0.125),
                                               padding_mode='reflect'))
        if config['strong_option'] == 'RA':
            trans.append(RandAugmentMC(n=2, m=10))
        elif config['strong_option'] == 'CUST':
            trans.append(HighlyCustomizableAugment(2, 10, -1, labeled_ds, config))
        elif config['strong_option'] == 'NONE':
            pass
        else:
            raise NotImplementedError()
        trans += [transforms.ToTensor(),
                  transforms.Normalize(mean=__mean, std=__std)]

        if config['manual_contrast']:
            trans.append(manual_contrast)
        strong = transforms.Compose(trans)

        if util.img_size > 200:
            self.simple = [transforms.RandomResizedCrop(util.img_size)]
        else:
            self.simple = [transforms.RandomCrop(size=util.img_size,
                                                 padding=int(util.img_size * 0.125),
                                                 padding_mode='reflect')]
        self.simple = transforms.Compose(([transforms.RandomHorizontalFlip()]) + self.simple + [
            transforms.ToTensor(),
            transforms.Normalize(mean=__mean, std=__std)] + ([manual_contrast] if config['manual_contrast'] else []))

        self.test_normalize = transforms.Compose([
            transforms.CenterCrop(util.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=__mean, std=__std)])

        td = {'strong': strong, 'simple': self.simple}
        self.aug = td[config['cat_augmentation']]
        self.test_mode = False

    def __len__(self) -> int:
        return len(self.labeled_ds)

    def __getitem__(self, index: int):
        img, lb, _ = self.labeled_ds[index]
        if self.test_mode:
            img = self.test_normalize(img)
        else:
            img = self.aug(img)
        return img, lb, index


@util.regmethod('cssr')
class CSSRMethod:

    def get_cfg(self, key, default):
        return self.config[key] if key in self.config else default

    def __init__(self, config, clssnum, train_set) -> None:
        self.config = config
        self.epoch = 0
        self.lr = config['learn_rate']
        self.batch_size = config['batch_size']
        self.clsnum = clssnum
        self.crt = CSSRCriterion(config['arch_type'], False)
        self.model = CSSRModel(self.clsnum, config, self.crt).cuda()
        self.modelopt = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=5e-4)
        self.wrap_ds = WrapDataset(train_set, self.config, inchan_num=3, )
        self.wrap_loader = data.DataLoader(self.wrap_ds,
                                           batch_size=self.config['batch_size'], shuffle=True, pin_memory=True,
                                           num_workers=6)
        self.lr_schedule = util.get_scheduler(self.config, self.wrap_loader)
        self.prepared = -999
        self.safe_distance_all = torch.zeros(self.clsnum)
        self.safe_margin_all = torch.zeros(self.clsnum)
        self.lambda1 = self.config['category_model']['lambda1']
        self.lambda2 = self.config['category_model']['lambda2']

    def train_epoch(self, flag):

        data_time = AverageMeter()
        batch_time = AverageMeter()
        train_acc = AverageMeter()
        running_loss = AverageMeter()
        pull_loss = AverageMeter()
        push_loss = AverageMeter()
        self.model.train()
        endtime = time.time()
        all_logic_distance = []
        all_label = []
        for i, data in enumerate(tqdm.tqdm(self.wrap_loader)):

            data_time.update(time.time() - endtime)

            self.lr = self.lr_schedule.get_lr(self.epoch, i, self.lr)
            util.set_lr([self.modelopt], self.lr)
            sx, lb = data[0].cuda(), data[1].cuda()
            xcloss, lossp, pred, scores, logic_distance = self.model(sx, lb, self.safe_margin_all,
                                                                     self.safe_distance_all, reqpredauc=True)

            # 记录所有样本和特定原型的距离和Label
            all_logic_distance.append(logic_distance.cpu())
            all_label.append(lb.cpu())
            if flag == 1:
                loss = xcloss
            # loss = xcloss + lossp[0]/(lossp[0]/xcloss).detach() + self.lambda2 * lossp[1]/(lossp[0]/xcloss).detach()

            if flag == 0:
                loss = xcloss + self.lambda1 * (lossp[0] + self.lambda2 * lossp[1])
                self.model.backbone_cs.cat_cls.prototypes.requires_grad_(False)
            self.modelopt.zero_grad()
            loss.backward()
            self.modelopt.step()
            nplb = data[1].numpy()
            train_acc.update((pred == nplb).sum() / pred.shape[0], pred.shape[0])
            running_loss.update(loss.item())

            pull_loss.update(lossp[0].item())
            push_loss.update(lossp[1].item())
            batch_time.update(time.time() - endtime)
            endtime = time.time()

        # 拼接所有批次的 logic
        final_logic = torch.cat(all_logic_distance, dim=0)
        final_label = torch.cat(all_label, dim=0)
        final_logic = torch.sum(final_logic, dim=(2, 3)).unsqueeze(-1)
        for c in range(self.clsnum):
            safe_distance = torch.max((final_logic[:, c, :].squeeze()) * ((final_label == c).float()))
            safe_margin = torch.mean((final_logic[:, c, :].squeeze()) * ((final_label == c).float()))
            self.safe_distance_all[c] = safe_distance
            self.safe_margin_all[c] = safe_margin
        print(self.safe_distance_all)

        self.epoch += 1
        training_res = \
            {"Loss": running_loss.avg,
             "pull_loss": pull_loss.avg,
             "push_loss": push_loss.avg,
             "TrainAcc": train_acc.avg,
             "Learn Rate": self.lr,
             "DataTime": data_time.avg,
             "BatchTime": batch_time.avg}
        return training_res

    def known_prediction_test(self, test_loader):
        self.model.eval()
        pred, scores, _, _ = self.scoring(test_loader)
        return pred

    def scoring(self, loader, prepare=False):
        gts = []
        deviations = []
        XXX = []
        XXX1= []
        scores = []
        prediction = []
        with (torch.no_grad()):
            for d in tqdm.tqdm(loader):
                x1 = d[0].cuda(non_blocking=True)
                gt = d[1].numpy()
                pred, scr, dev = self.model(x1, reqpredauc=True, prepareTest=prepare)
                xcloreF,xcloreF1 = self.model(x1, reqpredauc=False, prepareTest=prepare)
                prediction.append(pred)
                scores.append(scr)
                gts.append(gt)
                XXX.append(xcloreF)
                XXX1.append(xcloreF1)

        prediction = np.concatenate(prediction)
        scores = np.concatenate(scores)
        gts = np.concatenate(gts)
        # 解决方案：将 GPU Tensor 转换为 CPU，再转换为 NumPy 数组
        XXX = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in XXX]
        # 执行拼接
        XXX = np.concatenate(XXX)

        XXX1 = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in XXX1]
        # 执行拼接
        XXX1 = np.concatenate(XXX1)

        return prediction, scores, deviations, gts, XXX,XXX1

    def knownpred_unknwonscore_test(self, test_loader):
        self.model.eval()

        def normalize_scores(scores, mean_score, std_score):
            normalized_scores = (scores - mean_score) / (std_score + 1e-8)  # 归一化
            return normalized_scores


     
        score_knows_temp = []
        score_unknows_temp = []
        pred, scores, devs, gts, TestX,TestX1 = self.scoring(test_loader)
        for i in range(0, self.clsnum):
            mask = (pred == i)
            Test_c = TestX[mask, i, :, :]
            sum_per_Test_c = np.sum(Test_c, axis=(1, 2))
            score_know = sum_per_Test_c

            # score_know = abs(sum_per_Test_c - average_Train_cs[i])
            Test_c1 = TestX1[mask, i, :, :]
            sum_per_Test_c1 = np.sum(Test_c1, axis=(1, 2))
            score_unknow = sum_per_Test_c1

            #score_unknow = sum_per_Test_c - average_Train__no_cs[i]
            score_knows_temp.append(score_know)
            score_unknows_temp.append(score_unknow)

        all_score_know = np.concatenate(score_knows_temp)
        all_score_unknow = np.concatenate(score_unknows_temp)

        mean_know = np.mean(all_score_know)
        std_know = np.std(all_score_know)
        mean_unknow = np.mean(all_score_unknow)
        std_know_unknow = np.std(all_score_unknow)
        score_knows = []
        score_unknows = []
        for i in range(0, self.clsnum):
            score_knows.append(normalize_scores(score_knows_temp[i], mean_know, std_know))
            score_unknows.append(normalize_scores(score_unknows_temp[i], mean_unknow, std_know_unknow))

        auroc_weights = []
        aurocZ_array = []
        for j in range(1, 1000):
            weight = 0.001 * j
            aurocs_curr = []
            close_samples_append = []
            zhonghe_scores_append = []
            Knows_scores_append = []
            unKnows_scores_append = []
            for i in range(0, self.clsnum):
                mask = (pred == i)
                gts1 = gts[mask]
                close_samples = gts1 >= 0
                zhonghe_scores = weight * score_knows[i] + (1 - weight) * score_unknows[i]
                close_samples_append = np.concatenate((close_samples_append, close_samples))
                zhonghe_scores_append = np.concatenate((zhonghe_scores_append, zhonghe_scores))
                Knows_scores_append = np.concatenate((Knows_scores_append, score_knows[i]))
                unKnows_scores_append = np.concatenate((unKnows_scores_append, score_unknows[i]))
                if not np.any(close_samples) or np.isnan(zhonghe_scores).any():
                    auroc = 0.99999999
                else:
                    fpr, tpr, thresholds = metrics.roc_curve(close_samples, zhonghe_scores)
                    auroc = metrics.auc(fpr, tpr)
                if auroc != 0.99999999 and not np.isnan(auroc):
                    aurocs_curr.append(auroc)

            temp_auroc = np.mean(aurocs_curr)
            auroc_weights.append(temp_auroc)
            fpr, tpr, thresholds = metrics.roc_curve(close_samples_append, zhonghe_scores_append)
            aurocZ = metrics.auc(fpr, tpr)
            aurocZ_array.append(aurocZ)
        fpr, tpr, thresholds = metrics.roc_curve(close_samples_append, Knows_scores_append)
        auroc_kn = metrics.auc(fpr, tpr)

        fpr, tpr, thresholds = metrics.roc_curve(close_samples_append, unKnows_scores_append)
        auroc_unk = metrics.auc(fpr, tpr)

        max_auroc = max(auroc_weights)
        max_aurocZ = max(aurocZ_array)
        print("max_auroc:", max_auroc, "max_aurocZ:", max_aurocZ, "auroc_kn:", auroc_kn, "auroc_unk:", auroc_unk)

       


        selected = TestX[np.arange(TestX.shape[0]), pred, :, :]  # shape: (10000, 8, 8)

        # 对 8x8 特征图求和，得到 shape: (10000,)
        result = selected.sum(axis=(1, 2))
        aa = 1

        return  auroc_unk,auroc_kn,max_aurocZ

    def save_model(self, path):
        save_dict = {
            'model': self.model.state_dict(),
            'config': self.config,
            'optimzer' : self.modelopt.state_dict(),
            'epoch' : self.epoch,
        }
        torch.save(save_dict,path)

    def load_model(self,path):
        save_dict = torch.load(path)
        self.model.load_state_dict(save_dict['model'])
        if 'optimzer' in save_dict and self.modelopt is not None:
            self.modelopt.load_state_dict(save_dict['optimzer'])
        self.epoch = save_dict['epoch']
