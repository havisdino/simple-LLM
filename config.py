# ---------- Learning rate scheduler settings ----------
WARMUP_STEP = 120
INIT_LR = 1e-3
PEAK_LR = 1e-2
MIN_LR = 2e-4
DOWN_WEIGHT = 70 # the less the steeper
# ------------------------------------------------------------


# ---------- Transformer settings ----------
D_MODEL = 768
DFF = D_MODEL * 4
N_HEADS = 12
N_BLOCKS = 8
MAXLEN = 1024
VOCAB_SIZE = 30000
DROPOUT = 0.1
# ------------------------------------------------------------


# ---------- Training settings ----------
EPOCHS = 200
GLOBAL_BATCH_SIZE = 2048
BATCH_SIZE = 32
GRAD_ACCUM_STEP = GLOBAL_BATCH_SIZE // BATCH_SIZE
DEVICE = 'cuda'
CHECKPOINT_EPOCH = 4
PREFETCH_FACTOR = 2
# ------------------------------------------------------------


# ---------- Data settings ----------
END_TOKEN_ID = 29998
# ------------------------------------------------------------