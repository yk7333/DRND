env_name="hopper"
dataset_name="hopper_random"
actor_lambda="1.0"
critic_lambda="1.0"

while [ "$1" != "" ]; do
    case $1 in
        --env_name ) shift
                     env_name="$1"
                     ;;
        --dataset_name ) shift
                         dataset_name="$1"
                         ;;
        --actor_lambda ) shift
                         actor_lambda="$1"
                         ;;
        --critic_lambda ) shift
                          critic_lambda="$1"
                          ;;
    esac
    shift
done

config_path="configs/$env_name/$dataset_name.yaml"
save_path="data/$dataset_name"

export CUDA_VISIBLE_DEVICES="0"

echo "Running with actor_lambda = $actor_lambda,critic_lambda = $critic_lambda, dataset_name = $dataset_name"
python scripts/sac_drnd.py --config_path="$config_path" --actor_lambda="$actor_lambda" --critic_lambda="$critic_lambda" --save_path="$save_path"
wait