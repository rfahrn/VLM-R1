# Training Issues and Solutions for JSONL-based GRPO Training

## Issues Found

### 1. Model Path Not Found
**Error**: `FileNotFoundError: No such file or directory: "/capstor/scratch/cscs/rfahrni/models/Qwen2.5-VL-7B-Instruct/model-00001-of-00005.safetensors"`

**Root Cause**: The script was configured to load a model from a local path that doesn't exist in the current environment.

**Solution**: Use Hugging Face model identifiers instead of local paths.

### 2. Environment Setup Issues
**Error**: `externally-managed-environment` when trying to install Python packages.

**Root Cause**: The Python environment is externally managed and requires proper virtual environment setup.

## Solutions

### Quick Fix: Use the Corrected Script

I've created a fixed version of your script at `run_scripts/run_new_fixed.sh` with the following key changes:

1. **Model Path**: Changed from local path to Hugging Face identifier:
   ```bash
   # Before (problematic)
   model_path="/capstor/scratch/cscs/rfahrni/models/Qwen2.5-VL-7B-Instruct"
   
   # After (working)
   model_path="Qwen/Qwen2.5-VL-7B-Instruct"
   ```

### Environment Setup Solutions

#### Option 1: Use Docker (Recommended)
The project includes a `Dockerfile`. Build and use the Docker container:

```bash
docker build -t vlm-r1 .
docker run --gpus all -it vlm-r1
```

#### Option 2: Virtual Environment Setup
If you need to set up the environment manually:

```bash
# Create virtual environment
python3 -m venv vlm-r1-env
source vlm-r1-env/bin/activate

# Install dependencies with break-system-packages if needed
cd src/open-r1-multimodal
pip install -e ".[dev]" --break-system-packages

# Install additional dependencies
pip install wandb==0.18.3 tensorboardx qwen_vl_utils torchvision \
           flash-attn babel python-Levenshtein matplotlib \
           pycocotools openai httpx[socks] --break-system-packages
```

### Model Path Options

Based on the analysis of existing scripts, you have several model options:

1. **Hugging Face Models** (Recommended for this environment):
   - `"Qwen/Qwen2.5-VL-7B-Instruct"` (7B model, what you're trying to use)
   - `"Qwen/Qwen2.5-VL-3B-Instruct"` (3B model, smaller/faster)
   - `"OpenGVLab/InternVL2_5-4B-MPO"` (InternVL alternative)

2. **Local Paths** (if you have models locally):
   - Ensure the model directory contains all required files (`.safetensors`, `config.json`, etc.)
   - Verify the path exists and is accessible

### Data Path Verification

Your script references these data paths:
```bash
data_paths="/capstor/scratch/cscs/rfahrni/train_rec_grpo.jsonl:/capstor/scratch/cscs/rfahrni/test_rec_grpo.jsonl"
image_root="/capstor/store/cscs/swissai/a135/RadVLM_project/data/"
```

Make sure these paths exist and are accessible in your environment.

### Usage Instructions

1. **Use the fixed script**:
   ```bash
   chmod +x run_scripts/run_new_fixed.sh
   ./run_scripts/run_new_fixed.sh
   ```

2. **Or update your existing script** by changing the model path:
   ```bash
   # In your run_new.sh, change this line:
   model_path="Qwen/Qwen2.5-VL-7B-Instruct"
   ```

### Additional Considerations

1. **Memory Requirements**: The 7B model requires significant GPU memory. Consider using the 3B model if you encounter OOM errors.

2. **Internet Access**: When using Hugging Face model identifiers, ensure your environment has internet access to download the models.

3. **Caching**: Models will be cached locally after first download, typically in `~/.cache/huggingface/`.

4. **Flash Attention Warning**: The warning about Flash Attention 2.0 is non-critical but indicates the model isn't on GPU during initialization. This is normal for the loading process.

### Alternative Model Paths from Other Scripts

Based on other scripts in the repository, these paths work in different environments:
- `/training/models/Qwen2.5-VL-7B-Instruct`
- `/data9/shz/ckpt/Qwen2.5-VL-3B-Instruct`
- `${REPO_HOME}/Qwen2.5-VL-3B-Instruct`

Choose the appropriate path based on your environment setup.

## Testing the Fix

To test if the environment is properly set up:

```bash
python3 -c "from transformers import AutoTokenizer; print('Environment OK')"
```

If this works, your environment should be ready for training with the corrected script.