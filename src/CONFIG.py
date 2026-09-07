#chunk sizes

CHUNK_SIZE = 200


#data processing
PREPROCESS_LIST = ['scale', 'rotate_dim',['reduce_feat','nn']]
REDUCE_FEAT = 10
NN_AVG_MIN_ERROR = 0.001
MAX_NN_EPOCHS = 500

TRESHHOLD_GRAIN = 10
DESCRETE_PERCENT = 0.1

#boolean split settings
AVG_BOOL_SPLITS_EACH_FEATURE = 5

#predictor settings
MAX_TIME_PREDICTOR = 120
MAX_DEPTH = 3

#further tree split crits
MIN_SAMPLES_IN_LEAF = 8
MAX_GREED_ITERATIONS = 10


#global vars
GLOBAL_CHUNK_INFO = None
