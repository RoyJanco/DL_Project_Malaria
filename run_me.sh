# Train with 2 classes, L2 loss, beta = 0, 1 epochs
python train_main.py -c2 -l l2 -b 0 -e 1

# Train with 2 classes, L2 loss, weight_map, beta = 0, 1 epochs
python train_main.py -c2 -l l2 -w True -b 0 -e 1

# Train with 2 classes, AW loss, beta = 0, 1 epochs
python train_main.py -c2 -l AW -b 0 -e 1

# Train with 2 classes, AW loss, weight_map, beta = 0, 1 epochs
python train_main.py -c2 -l AW -w True -b 0 -e 1