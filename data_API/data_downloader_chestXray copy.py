#!/usr/bin/env python3
# Download the 56 zip files in Images_png in batches
import argparse
import requests
import shutil
import os

#if you are running on your computer, ny downoad 1 link
#https://www.geeksforgeeks.org/command-line-arguments-in-python/
parser = argparse.ArgumentParser()
parser.add_argument(
    "-s","--size",
    default=None,
    required=False,
    help="To reduce the size of the dataset enter : reduce"
    )
parser.add_argument(
    "-p","--proxy",
    default=None,
    required=False,
    help="Enter the proxy"
    )
args = parser.parse_args()
if args.size == "reduce":
    print("You chose the lite version download")
    links = [
    'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz']
# URLs for the zip files
else:
    links = [
        'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',#1
        'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',#2
        'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',#3
        'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',#4
        'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',#5
        'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',#6
        'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',#7
        'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',#8
        'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',#9
        'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',#10
        'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',#11
        'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'#12
    ]


if not os.path.exists("data") :
    os.mkdir("data")
#create the object, assign it to a variable
proxy = None#{'https': 'http://ccsmtl.proxy.mtl.rtss.qc.ca:8080'}
def download_file(url,local_filename):

    with requests.get(url, stream=True,proxies=proxy) as r:
        with open(local_filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    return



for idx, link in enumerate(links):
    fn = os.path.join(os.getcwd(),'data/images_%02d.tar.gz' % (idx+1))
    if not os.path.exists(fn) :
        print('downloading'+fn+'...')
        download_file(link, fn)  # download the zip file

print("Download complete. Please check the checksums")


#This part requires to be on linux os
query=f"ls data/*.gz |xargs -n1 tar -xzf"
os.system(query)