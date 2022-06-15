#!/bin/bash

branch_name="$(git symbolic-ref HEAD 2>/dev/null)"
branch_name=${branch_name##refs/heads/}
echo "$branch_name"
main="main"
master="master"
if [ "$branch_name"="$main" ] || [ "$branch_name"="$master" ]; then

   python -m pytest -v ./tests
   status=$?

   if [ $status -eq 1 ]; then
    exit 1
   fi
   if [ $status -eq 0 ]; then
    exit 0
   fi
  echo $status
  exit 2
fi

echo "not on main . Tests wont run by default"
exit 0
