# GigaPose Template Generation

GigaPose matches live camera frames against a set of 162 pre-rendered images of an object (the TemplateDB) to estimate pose. This guide covers generating those templates for any custom object.

This is a one-time offline step per object. The output gets committed to the repo.

## Prerequisites

- WSL2 running Ubuntu (required as pinocchio doesn't work on Windows)
- Miniconda installed in WSL2
- Blender installed on Windows

If you don't have WSL2:

```powershell
# PowerShell (Admin)
wsl --install
```

If you don't have Miniconda:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh
# restart terminal after install
```

## Step 1: Clone GigaPose and Set Up the Environment

```bash
mkdir -p ~/6dof && cd ~/6dof
git clone https://github.com/nv-nguyen/gigapose.git
cd gigapose
cp environment.yml environment_modified.yml
```

Open `environment_modified.yml` and change:

```yaml
- pytorch-lightning==1.8.1
```

to:

```yaml
- pytorch-lightning==1.8.6
```

Then:

```bash
conda env create -f environment_modified.yml
conda activate gigapose
pip install torch torchmetrics torchvision pytorch_lightning
pip install seaborn bokeh open3d joblib tqdm ruamel.yaml fvcore simplejson roma pypng scikit-image transforms3d webdataset einops
pip install omegaconf iopath opencv-python pycocotools matplotlib scipy ffmpeg
pip install hydra-colorlog hydra-core gdown pandas ruamel.yaml pyrender wandb peft
pip install --pre --extra-index-url https://archive.panda3d.org/ panda3d==1.11.0.dev3233
pip install nvisii pyrr xatlas
pip install git+https://github.com/megapose6d/megapose6d.git
pip install git+https://github.com/thodan/bop_toolkit.git
pip install -e .
pip install "numpy==1.26.4" --force-reinstall
pip install pin
pip install "numpy==1.26.4" --force-reinstall
```

> numpy gets reinstalled twice because `pin` (pinocchio) upgrades it to 2.x, but `bop_toolkit` needs < 2.0. The final reinstall wins. Ignore the pip conflict warnings.

> Don't use `conda install pinocchio` as it switches the Python interpreter to PyPy and breaks everything. `pip install pin` is the correct way.

Verify:

```bash
python -c 'import pinocchio; import megapose; import bop_toolkit_lib; print("All good!")'
```

## Step 2: Download Model Weights

```bash
mkdir -p ~/6dof/gigapose/gigaPose_datasets/pretrained
wget https://huggingface.co/datasets/nv-nguyen/gigaPose/resolve/main/gigaPose_v1.ckpt \
  -O ~/6dof/gigapose/gigaPose_datasets/pretrained/gigaPose_v1.ckpt
```

## Step 3: Prepare Your 3D Model in Blender

The model must be centred at world origin (0, 0, 0) and scaled in metres. If it's not centred, the renderer points at empty space and produces blank templates.

1. Open your model in Blender
2. Select the object
3. Right-click → **Set Origin → Origin to Geometry**
4. Press **Alt+G** to move it to world origin
5. Check **Scale** in the Item panel (N key) — if it's not `1.0, 1.0, 1.0`, apply it: **Object → Apply → Scale**
6. Check **Dimensions** are correct in metres (e.g. a 66mm wide object should show `0.066m`)
7. Export: **File → Export → Stanford PLY (.ply)**
   - Enable **Apply Transforms** in the export options
   - Name the file `obj_000001.ply` (increment for additional objects: `obj_000002.ply` etc.)
   - To save directly to WSL2, type this in Blender's path bar: `\\wsl$\Ubuntu\home\<username>\6dof\datasets\<object_name>\models\`

## Step 4: Set Up the Dataset Structure

Replace `<object_name>` with your dataset name (e.g. `monster_can`, `red_mug`):

```bash
mkdir -p ~/6dof/datasets/<object_name>/models
```

Create `models_info.json` in that folder. Values are in **millimetres**:

```json
{
  "1": {
    "diameter": <diagonal>,
    "min_x": <-half_width>, "min_y": <-half_depth>, "min_z": <-half_height>,
    "size_x": <width>, "size_y": <depth>, "size_z": <height>
  }
}
```

For example, for a 66mm × 66mm × 168mm can:

```json
{
  "1": {
    "diameter": 181.5,
    "min_x": -33.0, "min_y": -33.0, "min_z": -84.0,
    "size_x": 66.0, "size_y": 66.0, "size_z": 168.0
  }
}
```

`diameter` = `sqrt(size_x² + size_y² + size_z²)`. `min_x/y/z` = half the size values, negative.

Link the dataset into the gigapose directory:

```bash
ln -s ~/6dof/datasets/<object_name> ~/6dof/gigapose/gigaPose_datasets/datasets/<object_name>
```

Check the model is correctly centred:

```bash
cd ~/6dof/gigapose && conda activate gigapose
python3 -c "
import trimesh
m = trimesh.load('gigaPose_datasets/datasets/<object_name>/models/obj_000001.ply')
print('bounds:', m.bounds)
print('extents:', m.extents)
"
```

Bounds should be roughly symmetric around zero. Extents should match real-world dimensions in metres.

## Step 5: Generate the Templates

```bash
cd ~/6dof/gigapose
conda activate gigapose
DISPLAY=:99 python -m src.scripts.render_custom_templates \
  custom_dataset_name=<object_name> \
  data.test.root_dir=$(realpath gigaPose_datasets/datasets)
```

> `DISPLAY=:99` stops Panda3D crashing in WSL2 (no physical display). The `libEGL` warnings are safe to ignore.

> `$(realpath ...)` is required since without it the renderer subprocess can't find the model file even though the path looks correct.

Expected output:

```
[INFO] - Found 1 objects
[INFO] - Start rendering for 1 objects
100%|████████████████████| 1/1
[INFO] - Finished for 1/1 objects
```

---

## Step 6: Verify

```bash
ls ~/6dof/gigapose/gigaPose_datasets/datasets/templates/<object_name>/000001/ | wc -l
```

Should be **324** (162 RGB + 162 depth images). Browse them from Windows at:

```
\\wsl$\Ubuntu\home\<username>\6dof\gigapose\gigaPose_datasets\datasets\templates\<object_name>\000001\
```

RGB images should show the object as a white/grey shape against black, from many different angles. Depth images will appear black due to a WSL2 GPU limitation and doesn't affect the pipeline.

## What Gets Committed

| File | Description |
|------|-------------|
| `templates/<object_name>/000001/*.png` | 162 RGB renders + 162 depth maps |
| `templates/<object_name>/object_poses/000001.npy` | 162 camera viewpoint matrices |
| `models/obj_000001.ply` | Prepared 3D model |
| `models/models_info.json` | Bounding box metadata |
