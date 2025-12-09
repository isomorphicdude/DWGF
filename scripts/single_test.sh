gpu_id=$(gpustat --json | python3 -c "
import sys, json
data = json.load(sys.stdin)
# Ensure that the json output contains a 'gpus' key.
gpus = data.get('gpus', [])
if not gpus:
    sys.exit('No GPUs found in gpustat output')
# Rank by free memory first, then by lowest temperature
best_gpu = max(gpus,
    key=lambda g: (
        g['memory.total'] - g['memory.used'],
        -g['temperature.gpu']
    )
)
print(best_gpu['index'])
")
export CUDA_VISIBLE_DEVICES="${gpu_id}"
echo "Selected GPU: ${gpu_id}"

algo='dwgf'
deg='sr8'
# deg='inp_large_box'
# deg='deblur_gauss'
num_steps=999
w_t=0.15
lr_z=1.0
n_particles=1

python src/main.py exp.load_img_id=True\
                algo.deg=$deg\
                algo.name=$algo\
                exp.num_steps=$num_steps\
                algo.w_t=$w_t\
                algo.lr_z=$lr_z\
                algo.n_particles=$n_particles\
                exp.seed=10\
                exp.save_every=100\
                +algo.mask_idx=15 \
                +exp.img_path="data/samples" \
                +exp.img_id="00003.png"
