branch_name="$(git symbolic-ref HEAD 2>/dev/null)" ||
branch_name="(unnamed branch)"     # detached HEAD

branch_name=${branch_name##refs/heads/}
main="main"
master="master"
if ["$branch_name" = main || "$branch_name" = master] ; then

   python -m pytest -v .
   status=$?
   if (status=1) then
    exit 1
   fi
   if (status=0) then
    exit 0
   fi

exit 2
fi

echo "not on main . Tests wont run by default"
exit 0
