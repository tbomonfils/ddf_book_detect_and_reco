docker build -t ddf_book_detect_and_reco:dev -f dockerfile.dev .

#Docker Augmentor
docker run -it \
	-p 8888:8888 -p 5000:5000 -p 6006:6006 --rm \
	-v //c/Users/Thibaud/Desktop/D2F/ddf_book_detect_and_reco:/home/app/ \
	-e DISPLAY=192.168.99.1:0 -v /tmp/.X11-unix:/tmp/.X11-unix \
	ddf_video_comp:2.4_dev \
	jupyter notebook --ip=0.0.0.0 --allow-root &
	
#Docker noPillow
docker run -it \
	-p 8888:8888 -p 5000:5000 -p 6006:6006 --rm \
	-v //c/Users/Thibaud/Desktop/D2F/ddf_book_detect_and_reco:/home/app/ \
	-e DISPLAY=192.168.99.1:0 -v /tmp/.X11-unix:/tmp/.X11-unix \
	ddf_book_detect_and_reco_nopillow:dev \
	jupyter notebook --ip=0.0.0.0 --allow-root &
	
#Docker ref
docker run -it \
	-p 8888:8888 -p 5000:5000 -p 6006:6006 --rm \
	-v //c/Users/Thibaud/Desktop/D2F/ddf_book_detect_and_reco:/home/servier_app \
	-e DISPLAY=192.168.99.1:0 -v /tmp/.X11-unix:/tmp/.X11-unix \
	test_servier_ml_dev:2.0 \
	jupyter notebook --ip=0.0.0.0 --allow-root &