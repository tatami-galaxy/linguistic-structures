# imports
import torch.nn as nn
import torch
import numpy as np
from scipy import stats
from scipy.sparse.csgraph import minimum_spanning_tree as mst
from scipy.sparse import csr_matrix


# distance probe class
class DistanceProbe(nn.Module):
    
    def __init__(self, model_dim, probe_rank):
        super(DistanceProbe, self).__init__()
        self.model_dim = model_dim
        self.probe_rank = probe_rank
        if self.probe_rank is None:  # dxd transform by default
            self.probe_rank = self.model_dim
        self.proj = nn.Parameter(data = torch.zeros(self.model_dim, self.probe_rank))  # projecting transformation 
        nn.init.uniform_(self.proj, -0.05, 0.05) #0.05,0.05 #0.001,0.001


    def del_embeds(self, hidden, ids):
        new_hidden = hidden[:ids[0]]
        prev_id = ids[0]
        for i in range(1, len(ids)):
            new_hidden = torch.cat((new_hidden, hidden[prev_id+1:ids[i]]))
            prev_id = ids[i]
        return new_hidden


    def avg_embed(self, hidden, word_id):
        # get unique counts
        _, c = torch.unique_consecutive(word_id, return_counts=True)
        embed_list = []
        h_i = 0
        for i in range(c.shape[0]):
            if c[i] > 1:
                to_avg = hidden[h_i:h_i+c[i]]
                avgd = torch.mean(to_avg, dim=0) # d dimensional
                embed_list.append(avgd)
                h_i += c[i]
            else:
                embed_list.append(hidden[h_i])
                h_i += 1

        return torch.stack(embed_list)


    def re_pad(self, hidden_state, max_len):
        new_hidden_state = []
        for i in range(len(hidden_state)):
            s = hidden_state[i].shape[0] # length of the current sequence
            h_new = nn.functional.pad(hidden_state[i], (0, 0, 0, max_len-s)) # pad to max_len
            new_hidden_state.append(h_new)

        new_hidden_state = torch.stack(new_hidden_state)
        return new_hidden_state

    
    def forward(self, hidden_state, word_ids, label_mask, max_len):

        # hidden_state -> b, s', d
        # word_ids -> b, s' # -100 for both padding and special tokens
        # attention mask -> b, s' # will not mask out special tokens

        new_hidden_state = []

        for i in range(hidden_state.shape[0]):
            h = hidden_state[i] # s', d
            # del -100 tokens
            word_id = word_ids[i][word_ids[i] != -100]
            # ids for which we dont want the embedding (special tokens and pad)
            del_ids = (word_ids[i] == -100).nonzero(as_tuple=True)[0].tolist()
            # del those embeddings 
            h_new = self.del_embeds(h, del_ids) # s+duplicates, d

            # average over subword embeddings
            h_final = self.avg_embed(h_new, word_id) # s, d (s seq length for the example)

            new_hidden_state.append(h_final)

        # pad to max length in batch
        new_hidden_state = self.re_pad(new_hidden_state, max_len) # b, s, d 

        # squared distance computation
        transformed = torch.matmul(new_hidden_state, self.proj) # b,s,r
        batchlen, seqlen, rank = transformed.size()
        transformed = transformed.unsqueeze(2) # b, s, 1, r
        transformed = transformed.expand(-1, -1, seqlen, -1) # b, s, s, r
        transposed = transformed.transpose(1,2) # b, s, s, r
        diffs = transformed - transposed # b, s, s, r
        squared_diffs = diffs.pow(2) # b, s, s, r
        squared_distances = torch.sum(squared_diffs, -1) # b, s, s

        # mask out useless values to zero (maybe later?)
        squared_distances = squared_distances * label_mask
        

        return squared_distances


    

# depth probe class
class DepthProbe:
    pass



# L1 loss for distance matrices
class L1DistanceLoss(nn.Module):

    def __init__(self, args):
        super(L1DistanceLoss, self).__init__()
        self.args = args
        self.loss = nn.L1Loss(reduction='none') 

    def forward(self, predictions, labels, label_mask, lens):
        # computes L1 loss on distance matrices.

        labels = labels * label_mask
        loss = self.loss(predictions, labels)
        summed_loss = torch.sum(loss, dim=(1,2)) # sum for each sequence
        loss = torch.sum(torch.div(summed_loss, lens.pow(2)))
        return loss



class Metrics:


    def __init__(self, args):
        self.args = args
        self.min_length = args.min_length
        self.max_length = 50
        # spearman dict by sentence length
        # each value list of lists
        # each list within the outer list are spearman coeffs for a sentence
        self.dataset_spear = {}
        self.dataset_uuas = []
        self.results = {}
    
    
    # compute spearman for each sentence in a batch (returns an array with floats and nan)
    # append array to spearman dict, keys are sentence lengths (excluding nans)
    def add_spearman(self, pred_dist, labels, label_mask, sentences):
        # pred_dist, labels, label_mask -> b, s, s
        # sentences -> [{tokens : [token1, token2, ..]}, {tokens : [token1, token2, ..]}, ...]
        labels = labels * label_mask
        for b in range(pred_dist.shape[0]): # each example in batch
            sentence_spear = [] # spearman for a single sentence
            for s in range(pred_dist.shape[1]): # each tokekn in example
                res = stats.spearmanr(pred_dist[b][s].cpu(), labels[b][s].cpu()) 
                sentence_spear.append(res.statistic)  # scalar for each token, nan for mask

            true_len = len(sentences[b]['tokens']) # true length of the sentence excluding nan (correspondds to padding)
            true_spear = sentence_spear[:true_len]
            avg_spear = np.mean(true_spear)

            if true_len in self.dataset_spear:
                self.dataset_spear[true_len].append(avg_spear)
            else:
                self.dataset_spear[true_len] = [avg_spear]


    # reconstruct parse tree from distance between nodes
    # minimum spanning tree with distance matrix
    def add_uuas(self, pred_dist, labels, label_mask, sentences):
        # pred_dist, labels, label_mask -> b, s, s
        # sentences -> [{tokens : [token1, token2, ..]}, {tokens : [token1, token2, ..]}, ...]
        # pred_dist, labels are distance matrices
        # need adjanceny matrices to compute spanning tree
        # for labels, distance = 1 -> edge
        labels = labels * label_mask
        for b in range(pred_dist.shape[0]): # each example in batch
            true_len = len(sentences[b]['tokens']) # true length of the sentence excluding nan (correspondds to padding)
            if true_len < self.min_length or true_len > self.max_length:
                continue
            # remove padding
            pred_mat = csr_matrix(pred_dist[b][:true_len, :true_len].cpu())
            true_mat = csr_matrix(labels[b][:true_len, :true_len].cpu())
            # mst
            true_mst = mst(true_mat).toarray().astype(int)
            pred_mst = np.rint(mst(pred_mat).toarray()).astype(int)
            # edges
            # indices where there is a true edge
            # 1 -> edge
            indices = np.transpose((true_mst==1).nonzero())
            num_true_edges = len(indices)
            pred_edge_count = 0
            for e in range(num_true_edges):
                if pred_mst[indices[e][0], indices[e][1]] == 1:
                    pred_edge_count += 1
            uuas = pred_edge_count / num_true_edges
            self.dataset_uuas.append(uuas)



    def compute_spearman(self):

        avg = []
        for key, val in self.dataset_spear.items():
            # key from min to max (check init)
            # average over each sentence length
            if key >= self.min_length and key <= self.max_length:
                avg.append(np.mean(self.dataset_spear[key]))

        self.results['spearman'] = np.mean(avg)



    def compute_uuas(self):
        self.results['uuas'] = np.mean(self.dataset_uuas)


    

