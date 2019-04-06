#!/bin/sh
# ./run_benchmark.sh /path/to/tflite/model.tflite
function run_benchmark() {
    model_name=$1
    cpu_mask=$2
    echo ">>> run_benchmark $1 $2"
    if [ $# -eq 1 ]
    then
        res=$(adb shell /data/local/tmp/benchmark_model_r1.13_official \
            --graph=/data/local/tmp/${model_name} \
            --num_threads=1 \
            --warmup_runs=10 \
            --min_secs=0 \
            --num_runs=50 2>&1 >/dev/null)
    elif [ $# -eq 2 ]
    then
        res=$(adb shell taskset ${cpu_mask} /data/local/tmp/benchmark_model_r1.13_official \
            --graph=/data/local/tmp/${model_name} \
            --num_threads=1 \
            --warmup_runs=10 \
            --min_secs=0 \
            --num_runs=50 2>&1 >/dev/null)
    fi
    echo "${res}"
}

function run_benchmark_summary() {
    model_name=$1
    cpu_mask=$2
    echo ">>> run_benchmark_summary $1 $2"
    res=$(run_benchmark $model_name $cpu_mask | tail -n 3 | head -n 1)
    print_highlighted "${model_name} > ${res}"
}

function print_highlighted() {
    message=$1
    light_green="\033[92m"
    default="\033[0m"
    printf "${light_green}${message}${default}\n"
}

model_path=$1
cpu_mask=$2
adb push benchmark_model_r1.13_official /data/local/tmp/
adb shell 'ls /data/local/tmp/benchmark_model_r1.13_official' | tr -d '\r' | xargs -n1 adb shell chmod +x
adb push ${model_path} /data/local/tmp/

model_name=`basename $model_path`
run_benchmark_summary $model_name $cpu_mask

