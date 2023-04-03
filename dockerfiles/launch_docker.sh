
arg_tag=ubuntu1804:ort_dev
arg_gpus=all
arg_jupyter=0
arg_help=0



while [[ "$#" -gt 0 ]]; do case $1 in
  --tag) arg_tag="$2"; shift;;
  --gpus) arg_gpus="$2"; shift;;
  --jupyter) arg_jupyter="$2"; shift;;
  -h|--help) arg_help=1;;
  *) echo "Unknown parameter passed: $1"; echo "For help type: $0 --help"; exit 1;
esac; shift; done

if [ "$arg_help" -eq "1" ]; then
    echo "Usage: $0 [options]"
    echo " --help or -h         : Print this help menu."
    echo " --tag     <imagetag> : Image name for generated container."
    echo " --gpus    <number>   : Number of GPUs visible in container. Set 'none' to disable, and 'all' to make all visible."
    echo " --jupyter <port>     : Launch Jupyter notebook using the specified port number."
    exit;
fi

extra_args=""
if [ "$arg_gpus" != "none" ]; then
    extra_args="$extra_args --gpus $arg_gpus"
fi

if [ "$arg_jupyter" -ne "0" ]; then
    extra_args+=" -p $arg_jupyter:$arg_jupyter"
fi

docker_args="$extra_args -v ${PWD}/..:/workspace -d --rm -it "

if [ "$arg_jupyter" -ne "0" ]; then
    docker_args+=" jupyter-lab --port=$arg_jupyter --no-browser -p 8011:22 --ip 0.0.0.0 --allow-root"
fi

echo "Launching container:"
# echo "> docker run $docker_args"
echo "> docker run $arg_tag"
nvidia-docker run  -p 8011:22 --ip 0.0.0.0 $docker_args $arg_tag
