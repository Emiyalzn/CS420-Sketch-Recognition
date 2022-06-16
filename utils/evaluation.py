import torch


def compute_running_corrects(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = {}
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()
            res[k] = correct_k
        return res

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = {}
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True).cpu().numpy()
            res[k] = (correct_k / batch_size)
        return res

# Quadratic Expansion
def pairwise_distances(x, y=None):
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist

def retrieval_map_evaluation(test_states, test_targets, cate_num, collection_states=None, collection_targets=None, topk=(1,5), batch_size=32):
    collection_is_test = False
    if collection_states is None:
        collection_states = test_states
        collection_targets = test_targets
        collection_is_test = True
    num_test_batch = test_states.size(0) // batch_size + 1
    num_collect_batch = collection_states.size(0) // batch_size + 1
    MAPs = []
    for test_i in range(num_test_batch):
        test_i_distances = []
        for collect_i in range(num_collect_batch):
            test_batch = test_states[test_i*batch_size:(test_i+1)*batch_size,:] #[batch, feat_dim]
            collect_batch = collection_states[collect_i*batch_size:(collect_i+1)*batch_size,:] #[batch, feat_dim]
            tmp_distances = pairwise_distances(test_batch, collect_batch) # [batch, batch]
            if collection_is_test and test_i == collect_i:
                tmp_distances[torch.arange(tmp_distances.size(0)), torch.arange(tmp_distances.size(0))] = 1e+8
            test_i_distances.append(tmp_distances)
        test_i_distances = -torch.cat(test_i_distances, dim=1) #[batch, length]
        _ , pred_indics = test_i_distances.sort(dim=1)

        test_i_target = test_targets[test_i*batch_size:(test_i+1)*batch_size]

        for j in range(len(test_i_target)):
            c = test_i_target[j]
            res = (collection_targets[pred_indics[j]] == c).to(torch.float)
            k, rightk, precision = 0, 0, []
            while rightk < cate_num[c]:
                r = res[k].item()
                if r:
                    precision.append((res[:k + 1]).mean().item())
                    rightk += 1
                k += 1
            MAPs.append(sum(precision) / len(precision))
    MAP = sum(MAPs) / len(MAPs)
    return MAP

def retrieval_accuracy_evaluation(test_states, test_targets, collection_states=None, collection_targets=None, topk=(1,5), batch_size=32):
    collection_is_test = False
    if collection_states is None:
        collection_states = test_states
        collection_targets = test_targets
        collection_is_test = True
    num_test_batch = test_states.size(0) // batch_size + 1
    num_collect_batch = collection_states.size(0) // batch_size + 1
    correct = {k:0 for k in  topk}
    for test_i in range(num_test_batch):
        test_i_distances = []
        for collect_i in range(num_collect_batch):
            test_batch = test_states[test_i*batch_size:(test_i+1)*batch_size,:] #[batch, feat_dim]
            collect_batch = collection_states[collect_i*batch_size:(collect_i+1)*batch_size,:] #[batch, feat_dim]
            tmp_distances = pairwise_distances(test_batch, collect_batch) # [batch, batch]
            if collection_is_test and test_i == collect_i:
                tmp_distances[torch.arange(tmp_distances.size(0)), torch.arange(tmp_distances.size(0))] = 1e+8
            test_i_distances.append(tmp_distances)
        test_i_distances = -torch.cat(test_i_distances, dim=1) #[batch, length]
        _ , pred_indics = test_i_distances.topk(max(topk), dim=1)

        test_i_target = test_targets[test_i*batch_size:(test_i+1)*batch_size]
        for j in range(len(test_i_target)):
            for k in topk:
                if test_i_target[j] in collection_targets[pred_indics[j][:k]]:
                    correct[k] = correct[k] + 1

    return {'retrieval_{}'.format(k):(correct[k] / test_states.size(0)) for k in topk}