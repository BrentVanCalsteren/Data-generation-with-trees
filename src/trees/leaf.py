import uuid
#default leaf value -> return the samplers
class ReturnIDSandPROB:
    def __init__(self, size):
        self.total_size = size

    def __call__(self, tids):
        sample_list = list(tids)
        leaf_id = uuid.uuid4()
        return {"sample_ids": sample_list,
                'leaf_id': str(leaf_id),
                "rel_prob": len(sample_list) / self.total_size}


#other leaf val functions: NOT USED RIGHT NOW
def empty_val():
    def value(_):
        return 1
    return value
#-------------------------------------------------


