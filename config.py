# ---------- Learning rate scheduler settings ----------
WARMUP_STEP = 100
INIT_LR = 1e-3
PEAK_LR = 1e-2
MIN_LR = 2e-4
DOWN_WEIGHT = 80 # the less the steeper
# ------------------------------------------------------------


# ---------- Transformer settings ----------
D_MODEL = 768
DFF = D_MODEL * 4
N_HEADS = 12
N_BLOCKS = 8
MAXLEN = 1024
VOCAB_SIZE = 30000
DROPOUT = 0.1
WEIGHT_STD = 0.02
# ------------------------------------------------------------


# ---------- Training settings ----------
EPOCHS = 100000
GLOBAL_BATCH_SIZE = 1600
BATCH_SIZE = 16
GRAD_ACCUM_STEP = GLOBAL_BATCH_SIZE // BATCH_SIZE
DEVICE = 'cuda'
CHECKPOINT_EPOCH = 1
PREFETCH_FACTOR = 2
# ------------------------------------------------------------


# ---------- Data settings ----------
END_TOKEN_ID = 29998
# ------------------------------------------------------------