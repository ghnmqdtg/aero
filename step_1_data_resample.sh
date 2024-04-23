SOURCE_DIR=VCTK-48

# For loop to resample 48kHz to 2kHz, 4kHz, 8kHz, 12kHz, 16kHz, 24kHz
for i in 2 4 8 12 16 24
do
    TARGET_DIR=VCTK-$i
    # TARGET_SR = i * 1000
    TARGET_SR=$((i * 1000))
    python data_prep/resample_data.py --data_dir $SOURCE_DIR --out_dir $TARGET_DIR --target_sr $TARGET_SR
done