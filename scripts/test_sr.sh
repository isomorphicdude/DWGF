# super resolution (8 times)
algo='dwgf'
deg='sr8'
num_steps=999
vjp_coeff=0.1
w_t=0.15   #0.0009, 0.01
lr_z=1.0 #0.8
n_particles=4
use_wandb=false


python src/main.py exp.load_img_id=false \
                algo.name=$algo\
                algo.deg=$deg\
                exp.num_steps=$num_steps\
                algo.w_t=$w_t\
                algo.lr_z=$lr_z\
                algo.n_particles=$n_particles\
                algo.vjp_coeff=$vjp_coeff\
                exp.seed=10\
                exp.save_evolution=false\
                exp.use_wandb=$use_wandb\
                exp.max_num_images=100
