# cd to folder LTSLtest/
# Put this there named recompile.sh
# do $chmod u+x recompile.sh
# Now, you can recompile with $./recompile.sh
# Then go back to your work directory

pushd ../../../libsrc_cu/
rm bin/*
make
popd
rm ltslmain, ltslmain.o
make -f Makefile_lib.workstation
