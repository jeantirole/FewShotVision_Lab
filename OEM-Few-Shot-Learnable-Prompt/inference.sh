python ./inference.py --model-path /disk3/eric/checkpoints/seggpt/seggpt_vit_large.pth \
--prompt-img-dir /disk3/steve/dataset/OpenEarthMap-FSS/trainset/images \
--prompt-label-dir /disk3/steve/dataset/OpenEarthMap-FSS/trainset/labels \
--dataset-dir <path/to/queryset> \
--mapping mappings/test/vit.json \
--outdir <path/where/to/output>