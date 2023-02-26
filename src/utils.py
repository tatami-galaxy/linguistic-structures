# imports
import torch.nn as nn
import torch



# distance probe class
class DistanceProbe(nn.Module):
    
    def __init__(self, model_dim, probe_rank):
        super(DistanceProbe, self).__init__()
        self.model_dim = model_dim
        self.probe_rank = probe_rank
        if probe_rank is None:  # dxd transform by default
            self.probe_rank = self.model_dim
        self.proj = nn.Parameter(data = torch.zeros(self.model_dim, self.probe_rank))  # projecting transformation # device?
        nn.init.uniform_(self.proj, -0.05, 0.05)


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





    

