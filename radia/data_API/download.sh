while getopts d: flag
do
    case "${flag}" in
        d) destination=${OPTARG};;
    esac
done


#!/bin/bash
input="links.txt"
while IFS= read -r line
do
  echo  "$line"
  wget $line
done < "$input"

ls $destination/*.gz |xargs -n1 tar -xzf