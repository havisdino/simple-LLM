# ---------- Learning rate scheduler settings ----------
WARMUP_STEP = 180
INIT_LR = 3e-3
PEAK_LR = 3e-3
MIN_LR = 3e-4
DOWN_WEIGHT = 80 # the less the steeper

'''
--- Recommended learning rate schedule for vanilla transformers ---
WARMUP_STEP = 180
INIT_LR = 1e-3
PEAK_LR = 1e-2
MIN_LR = 2e-4
DOWN_WEIGHT = 80
'''

'''
--- Recommended learning rate schedule for rezero transformers ---
WARMUP_STEP = 180
INIT_LR = 3e-3
PEAK_LR = 3e-3
MIN_LR = 3e-4
DOWN_WEIGHT = 80
'''
# ------------------------------------------------------------


# ---------- Transformer settings ----------
D_MODEL = 768
DFF = D_MODEL * 4
N_HEADS = 12
N_BLOCKS = 10
MAXLEN = 512
VOCAB_SIZE = 30000
DROPOUT = 0.1
WEIGHT_STD = 0.05
ARCHITECTURE = 'rezero' # options: 'rezero', 'vanilla'
# ------------------------------------------------------------


# ---------- Training settings ----------
EPOCHS = 100
GLOBAL_BATCH_SIZE = 1600
BATCH_SIZE = 16
GRAD_ACCUM_STEP = GLOBAL_BATCH_SIZE // BATCH_SIZE
DEVICE = 'cuda'
CHECKPOINT_STEP = 10     # Save the model after <CHECKPOINT_STEP> steps of grad accumulation
PREFETCH_FACTOR = 2
USE_AMP = True
TRAIN_LIMIT = None
VAL_LIMIT = 100 * BATCH_SIZE     # number of samples
SAVE_LAST_K_CHECKPOINTS = 5
# ------------------------------------------------------------


# ---------- Data settings ----------
END_TOKEN_ID = 29998
TOKENIZER_PATH = 'tokenizer/byte-level-bpe-tinystories.json'
# ------------------------------------------------------------