python main.py --dataset=PE --runs=10 --record --exp_name=edge_noise --epochs=150
for edge_noise in $(seq 0.1 0.1 0.9); do
  python main.py --dataset=PE --run=10 --record --exp_name=edge_noise --edge_noise=$edge_noise --epochs=150
done

python main.py --dataset=F2T --runs=10 --record --exp_name=edge_noise --epochs=150
for edge_noise in $(seq 0.1 0.1 0.9); do
  python main.py --dataset=F2T --run=10 --record --exp_name=edge_noise --edge_noise=$edge_noise --epochs=150
done