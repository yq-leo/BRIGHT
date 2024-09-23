#python main.py --dataset=Cora --runs=10 --record --exp_name=attr_noise_new --epochs=150
#for attr_noise in $(seq 0.1 0.1 0.9); do
#  python main.py --dataset=Cora --run=10 --record --exp_name=attr_noise_new --attr_noise=$attr_noise --epochs=150
#done

python main.py --dataset=Douban --runs=5 --record --exp_name=attr_noise_new --epochs=250 --strong_noise
for attr_noise in $(seq 0.1 0.1 0.9); do
  python main.py --dataset=Douban --run=5 --record --exp_name=attr_noise_new --attr_noise=$attr_noise --epochs=250 --strong_noise
done