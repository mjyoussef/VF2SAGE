import faiss

def indexFlatL2(data, k, nlist):
    """
    data:  the embading graph index
    nlist: how many cells it takes
    """
    d = data.shape[1]
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    return index

def top_k(data, k, xq):
    """
    data: index 
    k: returns the top k vectors closest to our query vector xq
    xq: our query
    return : I: the k closest point
    """
    D, I = data.search(xq, k) 
    return I
