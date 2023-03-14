
data_root='data'
align_dir='raw_style_data_faces'

echo "STEP: start to prepare data for stylegan ..."
cd $data_root
if [ ! -d stylegan ]; then
  mkdir stylegan
fi
cd stylegan
stylegan_data_dir=$(pwd)
if [ ! -d "$(date +"%Y%m%d")" ]; then
  mkdir "$(date +"%Y%m%d")"
fi
cd "$(date +"%Y%m%d")"
cp $align_dir . -r
if [ -d $(echo $align_dir) ]; then
  cp $(echo $align_dir) . -r
fi
src_dir_sg=$(pwd)

cd $data_root/../source
outdir_sg="$(echo $stylegan_data_dir)/traindata_$(echo $stylename)_256_$(date +"%m%d")"
echo $outdir_sg
echo $src_dir_sg
if [ ! -d "$outdir_sg" ]; then
  python prepare_data.py --size 256 --out $outdir_sg $src_dir_sg
fi
echo "prepare data for stylegan finished!"

### train model
#cd $data_root
#cd stylegan
#stylegan_data_dir=$(pwd)
#outdir_sg="$(echo $stylegan_data_dir)/traindata_$(echo $stylename)_256_$(date +"%m%d")"
#echo "STEP:start to train the style learner ..."
#echo $outdir_sg
#exp_name="ffhq_$(echo $stylename)_s256_id01_$(date +"%m%d")"
#cd /data/vdb/qingyao/cartoon/mycode/stylegan2-pytorch
#model_path=face_generation/experiment_stylegan/$(echo $exp_name)/models/001000.pt
#if [ ! -f "$model_path" ]; then
#  CUDA_VISIBLE_DEVICES=6 python train_condition.py --name $exp_name --path $outdir_sg --config config/conf_server_train_condition_shell.json
#fi
#### [training...]
#echo "train the style learner finished!"