# Paths
BASE_DIRECTORY="."
LLAMA_CPP_DIR="."
HF_MODEL_DIR="neuphonic/neutts-nano-french"  # hf link
MODEL_NAME="neutts-nano-french"
OUT_DIR="$BASE_DIRECTORY/gguf_models/$MODEL_NAME"
# CTX=2048
CALIB_DATA="$BASE_DIRECTORY/single-lang-calibration-data/calibration_data_french.txt"
IMATRIX_THREADS=$(nproc)

# Quantization types to produce
QUANTS=(
  Q4_0
  Q8_0
)

############################
# SETUP
############################

mkdir -p "$OUT_DIR" 
source "$BASE_DIRECTORY/secrets.env" # set HF_TOKEN env variable to allow --remote

CONVERT_PY="$LLAMA_CPP_DIR/convert_hf_to_gguf.py"
IMATRIX_BIN="$LLAMA_CPP_DIR/build/bin/llama-imatrix"
QUANT_BIN="$LLAMA_CPP_DIR/build/bin/llama-quantize"

BF16_GGUF="$OUT_DIR/${MODEL_NAME}-BF16.gguf"
IMATRIX_FILE="$OUT_DIR/${MODEL_NAME}.imatrix"

############################
# 1. HF -> GGUF (BF16)
############################

echo "[1/3] Converting HF model to GGUF (BF16)"
python3 "$CONVERT_PY" \
  "$HF_MODEL_DIR" \
  --remote \
  --outfile "$BF16_GGUF" \
  --outtype bf16

############################
# 2. Build imatrix
############################

echo "[2/3] Building imatrix"
"$IMATRIX_BIN" \
  -m "$BF16_GGUF" \
  -t "$IMATRIX_THREADS" \
  -f "$CALIB_DATA" \
  -o "$IMATRIX_FILE"

############################
# 3. Quantize (all quants)
############################

echo "[3/3] Quantizing"

for Q in "${QUANTS[@]}"; do
  OUT_Q="$OUT_DIR/${MODEL_NAME}-${Q}.gguf"
  echo "  -> $Q"
  "$QUANT_BIN" \
    --imatrix "$IMATRIX_FILE" \
    "$BF16_GGUF" \
    "$OUT_Q" \
    "$Q"
done

echo "Done. Outputs in $OUT_DIR"
