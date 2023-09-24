import torch.utils.data as data
from torch.utils.data import Sampler, ConcatDataset, Subset
import torch 
import random 
import numpy as np 

class Dataset(data.Dataset):

    def __init__(self, data, target, static, stayid):
        """
            data: (n, 48, 182),
            target: (n, 48, 1),
            static: (n, 25),
            stayid: (n, 1)
        """

        # self.ti_data = ti_data
        self.data = data
        self.target = target
        self.static = static 
        self.stayid = stayid

    def __getitem__(self, index):

        data, static, target, stayid = self.data[index], self.static[index], self.target[index], self.stayid[index]

        data = np.float32(data)
        target = np.float32(target)
        static = np.float32(static)
        
        return data, static, target, stayid

    def __len__(self):
        return len(self.target)

class BySequenceLengthSampler(Sampler):
    """ 
    A custom Sampler that yields a list of batch indices at a time 
    """
    def __init__(self, data_source,  
                bucket_boundaries, batch_size=64,):
        """
            data_source: list of arrays [(200, 48), (200, 23), (200, 7), ...]
            bucket_boundaries: list of sequence lengths to group by. e.g. [1, 2, 8, ...., 122, 131, 141, 150, 163, 176, 191, 219]
            batch_size: batch size to use
        """
        ind_n_len = []
        for i, p in enumerate(data_source):
            ind_n_len.append( (i, p.shape[1]) )
        self.ind_n_len = ind_n_len
        self.bucket_boundaries = bucket_boundaries
        self.batch_size = batch_size
        
        
    def __iter__(self):
        data_buckets = dict()
        # where p is the id number and seq_len is the length of this id number. 
        for p, seq_len in self.ind_n_len:
            pid = self.element_to_bucket_id(p,seq_len)
            if pid in data_buckets.keys():
                data_buckets[pid].append(p)
            else:
                data_buckets[pid] = [p] 

        for k in data_buckets.keys(): # k is the bucket id 1 to 64 

            data_buckets[k] = np.asarray(data_buckets[k])

        iter_list = []
        for k in data_buckets.keys():
            np.random.shuffle(data_buckets[k])
            # each bucket has to be large enough to have 1 batch
            iter_list += (np.array_split(data_buckets[k]
                           , int(data_buckets[k].shape[0]/self.batch_size)))
        random.shuffle(iter_list) # shuffle all the batches so they arent ordered by bucket
        # size
        for i in iter_list: 
            yield i.tolist() # as it was stored in an array
    
    def __len__(self):
        return len(self.data_source)
    
    def element_to_bucket_id(self, x, seq_length):
        boundaries = list(self.bucket_boundaries)
        buckets_min = [np.iinfo(np.int32).min] + boundaries # [-inf, 1, 2, 3, 4, ..., 191, 219]
        buckets_max = boundaries + [np.iinfo(np.int32).max] # [1, 2, 3, 4, ..., 180, 219, inf]
        # length 1 is in bucket [1, 2) (bucket_id 1), length 216 is in bucket[191, 219) (bucket_id 64)
        conditions_c = np.logical_and(
          np.less_equal(buckets_min, seq_length),
          np.less(seq_length, buckets_max))
        bucket_id = np.min(np.where(conditions_c))
        return bucket_id

def col_fn(batchdata):
    """
    A simple collate fn works like this: 

        def my_collate(batch):
            data = [item[0] for item in batch]
            target = [item[1] for item in batch]
            target = torch.LongTensor(target)
            return [data, target]

        batchdata is a list of (data, static, target, stayid), which is picked by the custom sampler we wrote above
        [[(200, 48), (25), (48, 1), (1)], [(200, 28), (25), (48,1), (1)], [(200, 100), (25), (48, 1), (1)] ....]
    """

    len_data = len(batchdata)  
    seq_len = [batchdata[i][0].shape[-1] for i in range(len_data)]
    # [(48, ), (28, ), (100, )....]
    len_tem = [np.zeros((batchdata[i][0].shape[-1])) for i in range(len_data)]
    max_len = max(seq_len)

    # [(200, 48) ---> (200, 100)]
    padded_td = [np.pad(batchdata[i][0], pad_width=((0, 0), (0, max_len-batchdata[i][0].shape[-1])), \
                mode='constant', constant_values=-3) for i in range(len_data)]
    # [(48, 1) ---> (100, 1)]
    padded_label = [np.pad(batchdata[i][2], pad_width=((0, max_len-batchdata[i][0].shape[-1]), (0, 0)), \
                mode='constant', constant_values=0) for i in range(len_data)]
    static = [batchdata[i][1] for i in range(len_data)]
    stayids = [batchdata[i][3] for i in range(len_data)]
    
    # [(48, ) ---> (100, )]
    mask = [np.pad(len_tem[i], pad_width=((0, max_len-batchdata[i][0].shape[-1])), \
            mode='constant', constant_values=1) for i in range(len_data)]
        
    return torch.from_numpy(np.stack(padded_td)), torch.from_numpy(np.stack(static)), torch.from_numpy(np.asarray(padded_label)), torch.from_numpy(np.asarray(stayids)), torch.from_numpy(np.stack(mask))

def generate_buckets(bs, train_hist):
    """
    Find the boundaries of the buckets to group data 
    Args:
        bs: bucket size, which is >> than batch size, e.g. 3000 
        train_hist: histogram of the training data
    Returns: 
        a list of boundaries, e.g. [1, 2, 3, 4, ..., 180, 219]
    """
    buckets = []
    sum = 0
    s = 0
    for i in range(0, 218): 
        # train_hist length 218, 
        # train_hist[0] is len [0, 1), train_hist[217] is  [217, 218]
        sum +=train_hist[i] 
        if sum>bs:
            buckets.append(i)
            sum = 0 
    # residue is 58 < 128, current bucket is [1, ... 205], then remove 205, attach 219,
    if sum < bs:
        buckets.pop(-1)    
    buckets.append(219) 
    return buckets

def get_data_loader(args, train_head, dev_head, test_head, 
                train_sofa_tail, dev_sofa_tail, test_sofa_tail, 
                train_static = None, dev_static = None, test_static = None, 
                train_id = None, dev_id = None, test_id = None):
    """
    Args:
        args: main arguments
        train_head: list of train head data, e.g. [(200, 48), (200, 15), (200, 9), ...]
        dev_head: list of dev head data, 
        test_head: list of test head data,
        train_sofa_tail: list of tail part SOFA target. 
    """
    
    train_dataset = Dataset(train_head, train_sofa_tail, static = train_static, stayid = train_id)
    val_dataset = Dataset(dev_head, dev_sofa_tail, static = dev_static, stayid = dev_id)
    test_dataset = Dataset(test_head, test_sofa_tail, static = test_static, stayid = test_id)

    train_len = [train_head[i].shape[1] for i in range(len(train_head))]
    val_len = [dev_head[i].shape[1] for i in range(len(dev_head))]
    # max len in the data is 216 
    len_range = [i for i in range(0, 219)]
    # bin is [0, 1), [1, 2), ... [217, 218]
    train_hist, _ = np.histogram(train_len, bins=len_range)
    val_hist, _ = np.histogram(val_len, bins=len_range)

    if args.data_batching == 'random':

        train_dataloader = data.DataLoader(train_dataset, batch_size=args.bs, collate_fn=col_fn,
                                drop_last=False, pin_memory=False)  
        dev_dataloader = data.DataLoader(val_dataset, batch_size=args.bs, collate_fn=col_fn,
                                drop_last=False, pin_memory=False) 
        test_dataloader = data.DataLoader(test_dataset, batch_size=args.bs, collate_fn=col_fn,
                                drop_last=False, pin_memory=False) 
        
    elif args.data_batching == 'same':
        # same means each batch has the same length of time-series data, which constrains the batch size
        batch_sizes=6
        val_batch_sizes = 1
        test_batch_sizes = 1

        bucket_boundaries = [i for i in range(1, 219)]
        val_bucket_boundaries = [i for i in range(len(val_hist)) if val_hist[i]>0 ] + [219]

        sampler = BySequenceLengthSampler(train_head, bucket_boundaries, batch_sizes)
        dev_sampler = BySequenceLengthSampler(dev_head, val_bucket_boundaries, val_batch_sizes)
        test_sampler = BySequenceLengthSampler(test_head, bucket_boundaries, test_batch_sizes)

        train_dataloader = data.DataLoader(train_dataset, batch_size=1, 
                                batch_sampler=sampler, collate_fn=col_fn,
                             drop_last=False, pin_memory=False)
        dev_dataloader = data.DataLoader(val_dataset, batch_size=1, 
                                batch_sampler=dev_sampler, collate_fn=col_fn,
                                drop_last=False, pin_memory=False)
        test_dataloader = data.DataLoader(test_dataset, batch_size=1, 
                                batch_sampler=test_sampler, collate_fn=col_fn,
                                drop_last=False, pin_memory=False)

    elif args.data_batching == 'close':

        batch_sizes= args.bs
        # same bucket boudaries have less data for val and test in one bucket 
        val_batch_sizes = args.bs//8
        test_batch_sizes = args.bs//4

        bucket_boundaries = generate_buckets(args.bucket_size, train_hist)
        
        sampler = BySequenceLengthSampler(train_head, bucket_boundaries, batch_sizes)
        dev_sampler = BySequenceLengthSampler(dev_head, bucket_boundaries, val_batch_sizes)
        test_sampler = BySequenceLengthSampler(test_head, bucket_boundaries, test_batch_sizes)

        train_dataloader = data.DataLoader(train_dataset, batch_size=1, collate_fn=col_fn,
                                batch_sampler=sampler, 
                                drop_last=False, pin_memory=False)
        dev_dataloader = data.DataLoader(val_dataset, batch_size=1, collate_fn=col_fn,
                                batch_sampler=dev_sampler, 
                                drop_last=False, pin_memory=False)
        test_dataloader = data.DataLoader(test_dataset, batch_size=1, collate_fn=col_fn,
                                batch_sampler=test_sampler, 
                                drop_last=False, pin_memory=False)
        
    return train_dataloader, dev_dataloader, test_dataloader

# def get_huge_dataloader(args, train_head, dev_head, test_head, \
#                 train_sofa_tail, dev_sofa_tail, test_sofa_tail):
    
#     total_head = train_head + dev_head + test_head
#     total_target = np.concatenate((train_sofa_tail, dev_sofa_tail, test_sofa_tail), axis=0)
#     train_len = [total_head[i].shape[1] for i in range(len(train_head))]
#     # start_bin 
#     bin_start = 0 
#     len_range = [i for i in range(bin_start, 219+bin_start)]
#     train_hist, _ = np.histogram(train_len, bins=len_range)
    
#     bucket_boundaries = generate_buckets(args.bucket_size, train_hist)

#     train_dataset = Dataset(total_head, total_target)
#     sampler = BySequenceLengthSampler(total_head, bucket_boundaries, args.bs)
#     dataloader = data.DataLoader(train_dataset, batch_size=1, collate_fn=col_fn,
#                                 batch_sampler=sampler, 
#                                 drop_last=False, pin_memory=False)
#     return dataloader
