#!/bin/bash

num_instances=1
volume="vectordb_data"
type="qdrant"

HOST_START_PORT=6333
HOST_END_PORT=6400
socket=""
membind=""
memory_policy=""
numactl_args=()
created_containers=()

error_cleanup() {
    echo "Performing error cleanup"
    for container in ${created_containers[@]}; do
        echo "Stopping container $container"
        sudo docker stop $container > /dev/null
        echo "Removing container $container"
        sudo docker remove $container > /dev/null
    done
    echo "Removing volume $volume"
    sudo docker volume remove $volume
    exit 1
}

get_next_available_port() {
    #get list of host ports in use by docker
    used_ports=$(sudo docker ps -q | xargs -I {} sudo docker port {} | grep -oP '(?<=:)\d+' | sort --unique | sort -n)
    
    for port in $(seq $HOST_START_PORT $HOST_END_PORT); do
	    if ! echo "$used_ports" | grep -q "^$port$"; then
	        echo $port
	        return
	    fi
    done

    echo "No available ports in the range $start_port-$end_port" >&2
    exit 1
}

set_db_type() {
    case $type in
        qdrant)
            image_name="qdrant/qdrant"
            ;;
        weaviate)
            image_name="chainguard/weaviate:latest"
            ;;
    esac
}

disable_kernel_tpp() {
    local err_state=false

    echo 1 > /proc/sys/kernel/numa_balancing;
    if [[ $? -eq 1 ]]; then
        echo "Failed to disable Kernel Memory Tiering" >&2
        err_state=true
    else
        echo "Successfully disabled Kernel Memory Tiering"
    fi

    echo 0 > /sys/kernel/mm/numa/demotion_enabled;
    if [[ $? -eq 1 ]]; then
        echo "Failed to disable Kernel Memory Tiering Page Demotion" >&2
        err_state=true
    else
        echo "Successfully disabled Kernel Memory Tiering Page Demotion"
    fi

    if [[ $err_state = "true" ]]; then
        echo "Failed to disable Kernel TPP" >&2
        exit 1
    fi
}

enable_kernel_tpp() {
    local error_state=false
    echo 2 > /proc/sys/kernel/numa_balancing;
    if [[ $? -eq 1 ]]; then
        echo "Failed to enable Kernel Memory Tiering. This Kernel may not support tiering." >&2
        error_state=true
    else
        echo "Successfully enabled Kernel Memory Tiering"
	fi
	

    ECHO 1 > /sys/kernel/mm/numa/demotion_enabled;
    if [[ $? -eq 1 ]]; then
        echo "Failed to enable Kernel Memory Tiering Page Demotion" >&2
        err_state=true
    else
        echo "Successfully enabled Kernel Memory Tiering Page Demotion"
    fi

    if [[ $error_state = "true" ]]; then
        echo "Kernel TPP setup failed. Attempting to undo any changes and exiting" >&2
        disable_kernel_tpp
        exit 1
    fi
}

set_numactl_args() {
    [[ -z $memory_policy ]] && return

    #Count the number of values separated by commas for error checks
    IFS=',' read -ra numa_nodes <<< "$membind"
    numa_node_count=${#numa_nodes[@]}

    case $memory_policy in
    "preferred")
        #Check if only one node is specified to be preferred
        if ! [[ "$membind" =~ ^[0-9]+$ ]]; then
	        echo "A single NUMA node must be specified with -m when using the preferred memory policy" >&2
            print_usage
            exit 1
        else
            numactl_args+="--preferred=${membind}"
        fi
        ;;
    "interleave")
        #Check if the variable has two or more values separated by commas
        if [[ $numa_node_count -lt 2 ]]; then
            echo "Two or more NUMA node must be specified with -m when using the interleave memory policy" >&2
            print_usage
            exit 1
        else
            numactl_args+="--interleave=${membind}"
        fi
        ;;
    "weighted-interleave")
        if [[ $numa_node_count -lt 2 ]]; then
            echo "Two or more NUMA node must be specified with -m when using the interleave memory policy" >&2
            print_usage
            exit 1
        else
            numactl_args+="--weighted-interleave=${membind}"
        fi
        ;;
    "kerneltpp")
        if [ "$EUID" -ne 0 ]; then
            echo "Error: Kernel TPP can only be activated by root users" >&2
            exit 1
        else
            enable_kernel_tpp
        fi
        ;;
    *)
        echo "Unknown memory policy specified" >&2
        exit 1
        ;;
    esac
}

#note that preferred will make the node specified by -m to be preferred, interleave will make
#the nodes (plural!) specified by m interleaved. Same for weighted interleave (which also
#assumes weights are already configured)
print_usage() {
    echo "Usage: vectordb_manager.py [options]"
    echo "Options:"
    echo "-n, --num-instances NUM   Specify the number of vector database instances to start. Default: 1"
    echo "-t, --type DB_TYPE        Specify the vector database type (qdrant, weaviate). Default: 'qdrant'"
    echo "-s, --socket SOCKET       Bind vector database(s) to the specified CPU socket."
#    echo "-b, --bindcpus CPUS       Bind vector database(s) to the specified range/list of CPUs (e.g., 0-7)."
    echo "-m, --membind NODE         Bind vector database(s) to the specified NUMA node memory."
    echo "-p, --memory-policy POLICY                            Specify the memory policy (preferred, interleave, weighted-interleave, kerneltpp)."
    echo "-v, --volume PATH         Specify the host volume path for persistent storage. Default: 'vectordb_data'"
    echo "-h, --help                Display this help message and exit."
    echo "Examples:"
    echo "   vectordb_manager.py -n 3 -t Qdrant --socket 0 --membind 1 -v /var/data/vectordb"
    echo " vectordb_manager.py -n 2 -t Weaviate --bindcpus 0-7 --memory-policy kerneltpp"
    echo " Notes:"
    echo " - The --socket option allows vector databases to run only on the specified CPU socket."
    echo " - The --bindcpus option specifies a range and list of CPUs for binding, similar to numactl."
    echo " - The --membind option binds the vector database to the specified NUMA node memory."
    echo " - The --memory-policy option sets the memory policy to preferred, interleave, weighted-interleave, or kerneltpp."
    echo " - The script will configure Linux Kernel Transparent Page Placement when kerneltpp is used."
    echo " - The script will configure and use Kernel weighted interleaving policy when weightedinterleave is used."
    echo " - Warnings will be issued if Kernel TPP or Weighted Interleaving is configured but not specified, with an option to continue or exit."
    echo " - Each vector database instance will run inside a Docker container using a dedicated external port and persistent volume."
}

while [[ $# -gt 0 ]]; do
    case $1 in
	-n|--num-instances)
	    num_instances="$2"
	    shift 2
	    ;;
    -t|--type)
        type="$2"
        shift 2
        ;;
    -s|--socket)
        socket="$2"
        shift 2
        ;;
    -m|--membind)
        membind="$2"
        shift 2
        ;;
    -p|--memory-policy)
        memory_policy="$2"
        shift 2
        ;;
#    -b|--bindcpus)
#        bindcpus="$2"
#        b_flag=true
#        shift 2
#        ;;
    -v|--volume)
        volume="$2"
        shift 2
        ;;
    -h|--help)
        print_usage
        exit 0
        ;;
	-*|--*)
	    echo "Unknown argument $1" >&2
	    exit 1
	    ;;
    esac
done

set_db_type

for i in $(seq 1 $num_instances); do
    host_port=$(get_next_available_port)
    docker_args=("-d"
                 "--name="FMSDemo_db${i}""
                 #"-v $volume:/qdrant/storage"
                 "-p "$host_port":6333")

    #If they were specified, add socket and memory binding arguments to the final command
    [[ -n $socket ]] && docker_args+=("--cpuset-mems=$socket")
    [[ -n $membind ]] && docker_args+=("--cpuset-cpus=${membind}")

    docker_args+=("$image_name") #The image to run has to be passed in after the previous flags

    set_numactl_args
    if [[ -n $numactl_args ]]; then
        container_id=$(numactl ${numactl_args[@]} sudo docker run ${docker_args[@]})
    else
        container_id=$(sudo docker run ${docker_args[@]})
    fi

    if [[ $? -ne 0 ]]; then
        error_cleanup #docker will output its own error messages automatically
    else
        created_containers+=($container_id)
        echo "Created a new $type container using host port $host_port and with id $container_id"
        sleep 3
    fi
done
