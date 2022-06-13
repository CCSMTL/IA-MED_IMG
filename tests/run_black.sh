branch_name="$(git symbolic-ref HEAD 2>/dev/null)"

if (branch_name="main") || (branch_name="master") then

   python -m black
fi
exit 0
