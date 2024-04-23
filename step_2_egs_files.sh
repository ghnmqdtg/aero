# For each low and high resolution pair, one should create "egs files" twice: for low and high resolution.
# For 16 target, we do 2 -> 16, 4 -> 16, 8 -> 16, 12 -> 16
# For 48 target, we do 8 -> 48, 12 -> 48, 16 -> 48, 24 -> 48
TARGET_SR=(16 48)
for i in ${TARGET_SR[@]}
do
    if [ $i -eq 16 ]; then
        for j in 2 4 8 12
        do
            python data_prep/create_meta_files.py VCTK-$j egs/vctk/$j-$i lr
            python data_prep/create_meta_files.py VCTK-$i egs/vctk/$j-$i hr
        done
    elif [ $i -eq 48 ]; then
        for j in 8 12 16 24
        do
            python data_prep/create_meta_files.py VCTK-$j egs/vctk/$j-$i lr
            python data_prep/create_meta_files.py VCTK-$i egs/vctk/$j-$i hr
        done
    fi
done
