branch_name="$(git symbolic-ref HEAD 2>/dev/null)" ||
branch_name="(unnamed branch)"     # detached HEAD

branch_name=${branch_name##refs/heads/}
if (branch_name="main") || (branch_name="master") then
   python -m black -q ./CheXpert2
   status=$?
  if (status=1) then
    echo "black exited with code 1 ; as always?"
    exit 0
  fi
  if (status=0) then
    exit 0
  fi
  exit 2
fi
echo "Not on main ; black wont run"
exit 0
