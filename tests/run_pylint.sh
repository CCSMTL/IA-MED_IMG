branch_name="$(git symbolic-ref HEAD 2>/dev/null)" ||
branch_name="(unnamed branch)"     # detached HEAD

branch_name=${branch_name##refs/heads/}
if (branch_name="main") || (branch_name="master") then
   python -m pylint  ./CheXpert2
   code=$?
   exit 0
fi

echo "Not on main ; pylint wont run"
exit 0
