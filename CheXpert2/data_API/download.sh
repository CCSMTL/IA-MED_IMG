
cd ..
sudo mkdir data
cd data
curl -x http://tcr06.proxy.mtl.rtss.qc.ca:8080 ... $(cat ../data_API/links.txt)
ls ../data/*.gz |xargs -n1 tar -xzf