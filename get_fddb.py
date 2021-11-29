import os
import urllib.request as req

# downloads dataset from http://vis-www.cs.umass.edu/fddb/

def download(force=False, path='.'):
    print('Downloading FDDB into new folder ' + path + '/fddb/')
    if os.path.exists(path + '/fddb') and not force:
        print('FDDB already downloaded.')
        return

    print('This may take several minutes...')
    req.urlretrieve('http://vis-www.cs.umass.edu/fddb/originalPics.tar.gz', path + '/tmp.tar.gz')

    print('Extracting tar file...')

    os.system('mkdir ' + path + '/fddb')
    os.system('tar xzf ' + path + '/tmp.tar.gz -C ' + path + '/fddb')

    os.system('rm ' + path + '/tmp.tar.gz')

    print('Done!')

if __name__ == "__main__":
    path = '.'
    if os.path.isfile('./dataset_path'):
        with open('./dataset_path') as f:
            path = f.readline()[:-1]
        
    download(force=True, path=path)
