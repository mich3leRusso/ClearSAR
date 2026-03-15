# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate eotdl

# Move to project root
cd /home/michele-russo/eotdl

# Run the main script
python main.py \
    --validation 0.2
