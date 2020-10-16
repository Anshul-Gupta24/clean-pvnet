for e in {224..299..5}
do
	#python run.py --type evaluate --cfg_file configs/linemod.yaml test.dataset LinemodTest test.epoch $e model myholepuncher_triplet_mean0.1_conv2 cls_type holepuncher
	python run.py --type evaluate --cfg_file configs/linemod.yaml test.dataset LinemodVal test.epoch $e model myduck_triplet_mean0.1_conv2_final cls_type duck
done
