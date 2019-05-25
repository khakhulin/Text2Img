import numpy as np
import torch
import torch.nn as nn


def func_attention(query, context, gamma1):
    """
    query: B x T x D
    context: B x H x W x D
    """
    B, D = query.size(0), query.size(2)
    H, W = context.size(1), context.size(2)
    SR = H * W

    # --> B x SR x D
    context = context.view(B, SR, D)
    # Get attention
    # (B x SR x D)(B x D x T)
    # --> B x SR x T
    attn = torch.bmm(context, query.transpose(1, 2)) # Eq. (7) in AttnGAN paper
    attn = torch.softmax(attn, dim=2) # Eq. (8)

    attn = attn * gamma1
    attn = torch.softmax(attn, dim=1) # Eq. (9)

    # (B x D x SR)(B x SR x T)
    # --> B x D x T
    weightedContext = torch.bmm(context.transpose(1, 2), attn)

    return weightedContext, attn.view(B, H, W, -1)


# ##################Loss for matching text-image###################
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return w12 / (w1 * w2).clamp(min=eps)


def sent_loss(cnn_code, rnn_code, args, class_ids=None, eps=1e-8):
    # ### Mask mis-match samples  ###
    # that come from the same class as the real sample ###
    batch_size = cnn_code.size(0)

    masks = []

    if class_ids is not None:
        for i in range(batch_size):
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        masks = np.concatenate(masks, 0)
        # masks: B x B
        masks = torch.ByteTensor(masks).to(cnn_code.device)

    # cnn_norm / rnn_norm: B x 1
    cnn_norm = torch.norm(cnn_code, dim=1, keepdim=True)
    rnn_norm = torch.norm(rnn_code, dim=1, keepdim=True)
    # scores* / norm*: B x B
    scores0 = cnn_code @ rnn_code.t()
    norm0 = cnn_norm @ rnn_norm.t()
    scores0 = args.gamma3 * scores0 / norm0.clamp(min=eps)
    
    if class_ids is not None:
        scores0.masked_fill_(masks, -float('inf'))

    scores1 = scores0.t()

    labels = torch.LongTensor(range(batch_size)).to(cnn_code.device)
    loss0 = nn.CrossEntropyLoss()(scores0, labels)
    loss1 = nn.CrossEntropyLoss()(scores1, labels)

    return loss0, loss1


def words_loss(img_features, words_emb, cap_lens, args, class_ids=None):
    """
        words_emb(query): batch x nef x seq_len
        img_features(context): batch x nef x 17 x 17
    """
    batch_size = img_features.size(0)

    masks = []
    att_maps = []
    similarities = []
    cap_lens = cap_lens.tolist()

    for i in range(batch_size):
        if class_ids is not None:
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        # Get the i-th text description
        words_num = cap_lens[i]
        # -> 1 x T x D 
        word = words_emb[i, :words_num].unsqueeze(0)
        # -> B x T x D
        word = word.repeat(batch_size, 1, 1)
        # B x H x W x D
        context = img_features
        """
            query: B x T x D
            context: B x H x W x D
            weiContext: B x D x T
            attn: B x H x W x T
        """
        weiContext, attn = func_attention(word, context, args.gamma1)
        att_maps.append(attn[i].unsqueeze(0).contiguous())
        # --> B x T
        row_sim = cosine_similarity(word.transpose(1, 2), weiContext)
        # Eq. (10)
        row_sim.mul_(args.gamma2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)

        # --> B x 1
        # similarities(i, j): the similarity between the i-th image and the j-th text description
        similarities.append(row_sim)

    # B x B
    similarities = torch.cat(similarities, dim=1)
    similarities = similarities * args.gamma3

    if class_ids is not None:
        masks = np.concatenate(masks, 0)
        # masks: B x B
        masks = torch.ByteTensor(masks).to(img_features.device)
        similarities.masked_fill_(masks, -float('inf'))

    similarities1 = similarities.transpose(0, 1)

    labels = torch.LongTensor(range(batch_size)).to(img_features.device)
    loss0 = nn.CrossEntropyLoss()(similarities, labels)
    loss1 = nn.CrossEntropyLoss()(similarities1, labels)

    return loss0, loss1, att_maps


def discriminator_loss(netD, real_imgs, fake_imgs, conditions,
                       real_labels, fake_labels):
    # Forward
    real_features = netD(real_imgs)
    fake_features = netD(fake_imgs.detach())
    # loss
    #
    cond_real_logits = netD.cond_discriminator(real_features, conditions)
    cond_real_err = nn.BCELoss()(cond_real_logits, real_labels)
    cond_fake_logits = netD.cond_discriminator(fake_features, conditions)
    cond_fake_err = nn.BCELoss()(cond_fake_logits, fake_labels)

    batch_size = real_features.size(0)
    cond_wrong_logits = netD.cond_discriminator(real_features[:(batch_size - 1)], conditions[1:batch_size])
    cond_wrong_err = nn.BCELoss()(cond_wrong_logits, fake_labels[1:batch_size])

    if netD.uncond_discriminator is not None:
        real_logits = netD.uncond_discriminator(real_features)
        fake_logits = netD.uncond_discriminator (fake_features)
        real_err = nn.BCELoss()(real_logits, real_labels)
        fake_err = nn.BCELoss()(fake_logits, fake_labels)
        # TODO maybe constant should be parameters
        errD = ((real_err + cond_real_err) / 2. +
                (fake_err + cond_fake_err + cond_wrong_err) / 3.)
    else:
        errD = cond_real_err + (cond_fake_err + cond_wrong_err) / 2.
    return errD


def generator_loss(netsD, image_encoder,
                   fake_images, real_labels,
                   words_embeddings, sentence_embedding,
                    cap_lens, args):
    numDs = len(netsD)
    batch_size = real_labels.size(0)
    logs = ''
    # Forward
    errG_total = 0
    for i in range(numDs):
        features = netsD[i](fake_images[i])
        cond_logits = netsD[i].cond_discriminator(features, sentence_embedding)
        cond_errG = nn.BCELoss()(cond_logits, real_labels)
        if netsD[i].uncond_discriminator is not None:
            logits = netsD[i].uncond_discriminator(features)
            errG = nn.BCELoss()(logits, real_labels)
            g_loss = errG + cond_errG
        else:
            g_loss = cond_errG
        errG_total += g_loss
        # err_img = errG_total.data[0]
        logs += 'generator loss {0}: {1:.5f} '.format(i, g_loss.item())

        # Ranking loss
        if i == (numDs - 1):
            # words_features: batch_size x nef x 17 x 17
            # sent_code: batch_size x nef
            #  TODO unsupported feature
            class_ids = None
            region_features, cnn_code = image_encoder(fake_images[i])
            w_loss0, w_loss1, _ = words_loss(region_features, words_embeddings,  cap_lens, args=args, class_ids=None)
            w_loss = (w_loss0 + w_loss1) * args.smooth_lambda

            s_loss0, s_loss1 = sent_loss(cnn_code, sentence_embedding, args, class_ids)
            s_loss = (s_loss0 + s_loss1) * args.smooth_lambda

            errG_total += w_loss + s_loss
            logs += 'w_loss: {0:.2f} s_loss: {1:.2f} ' .format(w_loss.item(), s_loss.item())
    return errG_total, logs


##################################################################
def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD
