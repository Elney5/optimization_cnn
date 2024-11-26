import numpy as np
def getmask(layers, mask, maskflatten, mask_type='min', percentile=0.2, printactivation=False, dropOne=False):

    nodevalues = []
    layermeans = {}

    # if only drop one, then percentile is 0
    if dropOne:
        percentile = 0

    # if only one layer
    if (len(mask) == 1):
        # if mask type is apoz, then sum up the number of zero activations
        if 'apoz' in mask_type:
            layermeans[0] = np.sum(layers == 0, axis=0).ravel()
        else:
            layermeans[0] = np.mean(np.abs(layers), axis=0).ravel()
        nodevalues = np.hstack([nodevalues, layermeans[0]])
        if printactivation:
            print('Layer activations:', layermeans[0])

    # if more than one layer
    else:
        for i in range(len(mask)):
            # if mask type is apoz, then sum up the number of zero activations
            if 'apoz' in mask_type:
                layermeans[i] = np.sum(layers[i] == 0, axis=0).ravel()
            else:
                layermeans[i] = np.mean(np.abs(layers[i]), axis=0).ravel()
            nodevalues = np.hstack([nodevalues, layermeans[i]])
            if printactivation:
                print('Layer activations:', layermeans[i])

    # remove only those in maskindex
    maskflatten = np.ravel(np.where(maskflatten == 1))

    # find out the threshold node/filter value to remove
    if len(maskflatten) > 0:
        # for max mask
        if mask_type == 'max' or 'apoz' in mask_type:
            sortedvalues = -np.sort(-nodevalues[maskflatten])
            index = int((percentile) * len(sortedvalues))
            maxindex = sortedvalues[index]

        # for min or % mask
        else:
            sortedvalues = np.sort(nodevalues[maskflatten])
            index = int(percentile * len(sortedvalues))
            maxindex = sortedvalues[index]

    # Calculate the number of nodes to remove
    nummask = 0

    for v in mask.values():
        nummask += np.sum(v)

    totalnodes = int((percentile) * nummask)

    if dropOne:
        totalnodes = 1

    # remove at least one node
    if (totalnodes == 0):
        totalnodes = 1

    # identify the indices to drop for random mask
    if mask_type == 'random':
        indices = np.random.permutation(maskflatten)
        # take only the first totalnodes number of nodes
        indices = indices[:totalnodes]

        dropmaskindex = {}
        startindex = 0
        # assign nodes/filters to drop for each layer in dropmaskindex
        for k, v in mask.items():
            nummask += np.sum(v)
            dropmaskindex[k] = indices[(indices >= startindex) & (indices < startindex + len(v))] - startindex
            startindex += len(v)

    for i, layermean in layermeans.items():

        # only if there is something to drop in current mask
        if (np.sum(mask[i]) > 0):
            # Have different indices for different masks
            if mask_type == 'apoz':
                indices = np.ravel(np.where(layermean >= maxindex))
                curindices = np.ravel(np.where(mask[i].ravel()))
                indices = [j for j in indices if j in curindices]


        else:
            # default
            indices = np.ravel(np.where(mask[i] == 1))

        # shuffle the indices only if we are not dropping one node/filter
        if (dropOne == False):
            indices = np.random.permutation(indices)

        newmask = mask[i].ravel()

        # for layer masks, total nodes dropped is by percentile of the layer of each mask
        if 'layer' in mask_type:
            initialpercent = np.sum(mask[i]) * 1.0 / len(mask[i].ravel())
            totalnodes = int(initialpercent * (percentile) * len(mask[i].ravel()))

            # remove at least 1 node
            if (totalnodes == 0):
                totalnodes = 1

        if (len(indices) > 0):

            # remove at most total nodes number of nodes
            if (len(indices) > totalnodes):
                indices = indices[:totalnodes]

            # remove nodes
            newmask[indices] = 0

            # updated totalnodes to be removed
            totalnodes = totalnodes - len(indices)

        # reshape to fit new mask
        mask[i] = newmask.reshape(mask[i].shape)

    return mask