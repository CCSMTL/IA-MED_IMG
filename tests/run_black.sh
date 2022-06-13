branch_name="$(git symbolic-ref HEAD 2>/dev/null)"

if (branch_name="main") || (branch_name="master") then

   python -m black -q ./CheXpert2
   status=$?

fi
#if (status=1) then
#  exit 1
#fi
#if (status=0) then
#  exit 0
#fi
#exit 2
exit 0
