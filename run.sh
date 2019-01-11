dataset=market1501
#dataset=duke
#dataset=mars
#dataset=DukeMTMC-VideoReID

EF=10
init=-1
loss=ExLoss
#loss=CrossEntropyLoss

fea=2048
momentum=0.5
epochs=70
stepSize=55
batchSize=16
lambda=0.8

logs=logs/$dataset

python3 run.py --dataset $dataset --logs_dir $logs --EF $EF --init $init --loss $loss --fea $fea -m $momentum -e $epochs -s $stepSize -b $batchSize --lamda $lambda
