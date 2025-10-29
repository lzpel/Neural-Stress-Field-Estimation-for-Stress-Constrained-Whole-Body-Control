generate:
	pipenv install
	mkdir -p out
	cd out && curl -O https://storage.googleapis.com/lzpelshare/imagenet/n02107312.tar.gz
	cd out && curl -O https://storage.googleapis.com/lzpelshare/imagenet/n02704792.tar.gz
test:
	pipenv run pytest cuda_test.py basic_vae.py -s
clean:
	pipenv --rm