#!/bin/bash

# Define the commands and their respective arguments
commands=(
    "python train1.py"
    "python train2.py"
    "python train3.py"
    "python train4.py"
    #"python train5.py"
    #"python train6.py"
    #"python train7.py"
    #"python train8.py"
    #"python train9.py"
    #"python train10.py"
    #"python train2.py"
    #"python train4.py"
    #"python train3.py"
)

# Run the commands sequentially
for command in "${commands[@]}"
do
    echo "Running: $command"
    $command
    if [ $? -eq 0 ]; then
        echo "Success: $command"
    else
        echo "Failed: $command"
    fi
    echo
done

echo "All programs have been executed."
